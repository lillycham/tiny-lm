"""The playground's server: prompts, sampling, attention and the HTTP endpoints.

The HTTP tests start a real server on a free port, with a tiny model in a temporary
checkpoints folder, so they run in a second or two on the CPU.
"""
import json
import math
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from web import chat_sft
from core import gpt
from stories import instruct_sft
from tools import playground as pg
from stories import sft
from core.word_bpe import EOT_ID, TOKENISER, VOCAB_SIZE, WordBPE


def tiny(**options):
    torch.manual_seed(0)
    return gpt.GPT(block=16, emb=12, heads=3, layers=2, dropout=0.0, vocab=VOCAB_SIZE, **options).eval()

class Fixed(nn.Module):
    """A stand-in model that always predicts the tokens in `plan`, one after another."""
    block = 16

    def __init__(self, plan, start):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.plan, self.start = plan, start

    def forward(self, x, last_only=False):
        logits = torch.full((1, x.shape[1], VOCAB_SIZE), -1e9)
        logits[0, -1, self.plan[x.shape[1] - self.start]] = 0
        return logits

@pytest.fixture(scope="module")
def tok():
    if not TOKENISER.exists():
        pytest.skip("needs the tokeniser from word_bpe.py")
    return WordBPE.load(TOKENISER)

# ---------- models and prompts ----------
def test_list_models(tmp_path):
    for name in ["stories_gpt.pt", "stories_sft_384w6l.pt", "stories_instruct.pt", "stories_gpt.resume.pt",
                 "gpt.pt", "snapshots/stories_gpt_step1024.pt", "snapshots/stories_gpt_step64.pt"]:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).touch()
    listed = [(m["path"], m["kind"], m["group"]) for m in pg.list_models(tmp_path)]
    assert listed == [("stories_gpt.pt", "story", "models"),
                      ("stories_instruct.pt", "instruct", "models"),
                      ("stories_sft_384w6l.pt", "name", "models"),
                      ("snapshots/stories_gpt_step64.pt", "story", "snapshots"),    # by step, not as text
                      ("snapshots/stories_gpt_step1024.pt", "story", "snapshots")]

def test_prompts_match_training():
    """The fine-tuned models only follow requests in exactly their training format."""
    assert pg.build_prompt("story", text="Once") == ("Once", True)
    assert pg.build_prompt("name", name=" Zork ") == (sft.prompt("Zork"), False)
    got, after_eot = pg.build_prompt("instruct", words="dragon, soup ,happy", features=["Dialogue"])
    assert got == instruct_sft.prompt(instruct_sft.request(["dragon", "soup", "happy"], ["Dialogue"]))
    assert not after_eot
    assert pg.build_prompt("chat", question="Why?", context="Text.") == (chat_sft.prompt("Why?", "Text."), False)
    got, _ = pg.build_prompt("chat", question="And Spain?", history=[["Capital of France?", "", " Paris. "]])
    assert got == "User: Capital of France?\nAssistant: Paris.\nUser: And Spain?\nAssistant: "
    got, _ = pg.build_prompt("chat", question="Hi", system=" Be brief. ")
    assert got == "System: Be brief.\nUser: Hi\nAssistant: "
    for bad in [dict(mode="name"), dict(mode="instruct", words=" , "), dict(mode="chat", question=" "),
                dict(mode="chat", question="Hi", history=[["only two", "items"]]),
                dict(mode="instruct", words="cat", features=["Sad"]), dict(mode="poem")]:
        with pytest.raises(ValueError):
            pg.build_prompt(**bad)

def test_models_stay_inside_their_folder(tmp_path):
    gpt.save(tiny(), tmp_path / "stories_gpt_tiny.pt")
    models = pg.Models(tmp_path)
    assert models.get("stories_gpt_tiny.pt") is models.get("stories_gpt_tiny.pt")     # loaded once
    for bad in ["../stories_gpt_tiny.pt", "/etc/passwd", "missing.pt", "stories_bpe.json"]:
        with pytest.raises(ValueError):
            models.get(bad)

# ---------- sampling ----------
def test_sample_is_repeatable_and_greedy_at_zero(tok):
    model, ids = tiny(), tok.encode("Once upon a time")
    one = [t["id"] for t in pg.sample(model, tok, ids, 20, 1.0, seed=3)]
    assert one == [t["id"] for t in pg.sample(model, tok, ids, 20, 1.0, seed=3)]
    assert one != [t["id"] for t in pg.sample(model, tok, ids, 20, 1.0, seed=4)]
    greedy = list(pg.sample(model, tok, ids, 5, temperature=0))
    first = model(torch.tensor([ids]))[0, -1]
    assert greedy[0]["id"] == first.argmax().item()
    assert greedy[0]["p"] == pytest.approx(torch.softmax(first, -1).max().item(), abs=1e-5)
    assert greedy[0]["top"][0][1] == pytest.approx(greedy[0]["p"], abs=1e-6)

def test_sample_joins_characters_split_across_tokens(tok):
    """é is two bytes, 0xC3 0xA9. As two tokens, the first gives no text and the second all of it."""
    ids = tok.encode("Hi")
    out = list(pg.sample(Fixed([0xC3, 0xA9, EOT_ID], len(ids)), tok, ids, 10))
    assert [t["text"] for t in out] == ["", "é", ""]
    assert out[0]["piece"] == "�"                          # the half character, on its own
    assert out[-1]["id"] == EOT_ID and len(out) == 3        # stops at <|endoftext|>, and says so

# ---------- attention ----------
@pytest.mark.parametrize("smear", [False, True])
def test_attention_from(smear):
    model = tiny(smear=smear)
    ids = list(range(40))
    offset, w = pg.attention_from(model, ids, 30)
    assert offset == 30 + 1 - model.block                   # only the last block tokens are in view
    assert len(w) == 2 and len(w[0]) == 3 and len(w[0][0]) == model.block
    assert all(abs(sum(head) - 1) < 1e-3 for layer in w for head in layer)
    assert pg.attention_from(model, ids, 0) == (0, [[[1.0]] * 3] * 2)   # the first token sees only itself
    with pytest.raises(ValueError):
        pg.attention_from(model, ids, 40)

# ---------- the server ----------
@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not TOKENISER.exists():
        pytest.skip("needs the tokeniser from word_bpe.py")
    root = tmp_path_factory.mktemp("checkpoints")
    gpt.save(tiny(), root / "stories_gpt_tiny.pt")
    srv = pg.make_server(root, "cpu", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()

def call(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method="POST" if body else "GET")
    with urllib.request.urlopen(req) as r:
        return r.read().decode()

def test_page_and_model_list(server):
    assert "tiny-lm playground" in call(server + "/")
    listed = json.loads(call(server + "/api/models"))
    assert [m["path"] for m in listed["models"]] == ["stories_gpt_tiny.pt"]
    assert listed["features"] == list(instruct_sft.FEATURES)

def test_generate_streams_tokens(server, tok):
    lines = [json.loads(l) for l in call(server + "/api/generate", {
        "model": "stories_gpt_tiny.pt", "mode": "instruct", "words": "cat", "max_tokens": 5, "seed": 1}).splitlines()]
    start, tokens, end = lines[0], lines[1:-1], lines[-1]
    assert start["type"] == "start" and start["prompt"] == "User: Tell me a story that uses the words cat.\nAssistant: "
    assert start["prompt_ids"] == tok.encode(start["prompt"])      # instruct: no <|endoftext|> first
    assert start["params"] == sum(p.numel() for p in tiny().parameters())
    assert [t["type"] for t in tokens] == ["token"] * end["tokens"] and 1 <= end["tokens"] <= 5
    assert end["type"] == "end" and end["reason"] in ("eot", "length")

    story = json.loads(call(server + "/api/generate", {"model": "stories_gpt_tiny.pt", "text": "Hi", "max_tokens": 1}).splitlines()[0])
    assert story["prompt_ids"] == [EOT_ID] + tok.encode("Hi")       # story: after <|endoftext|>

def test_attention_endpoint(server):
    got = json.loads(call(server + "/api/attention", {"model": "stories_gpt_tiny.pt", "ids": [1, 2, 3, 4], "t": 2}))
    assert got["offset"] == 0 and len(got["weights"][0][0]) == 3

@pytest.mark.parametrize("path, body", [
    ("/api/generate", {"model": "../stories_gpt_tiny.pt"}),
    ("/api/generate", {"model": "stories_gpt_tiny.pt", "mode": "name"}),
    ("/api/attention", {"model": "stories_gpt_tiny.pt", "ids": [1, 2], "t": 5}),
])
def test_bad_requests_get_400(server, path, body):
    with pytest.raises(urllib.error.HTTPError) as e:
        call(server + path, body)
    assert e.value.code == 400 and json.loads(e.value.read())["error"]

def test_web_models_are_listed_with_their_family(tmp_path):
    (tmp_path / "snapshots").mkdir()
    for name in ["stories_gpt_384w6l.pt", "web_gpt_768w12l.pt", "web_gpt_768w12l_anneal.pt",
                 "web_gpt_768w12l.resume.pt", "snapshots/web_gpt_768w12l_step64.pt"]:
        (tmp_path / name).touch()
    got = {m["path"]: (m["family"], m["group"]) for m in pg.list_models(tmp_path)}
    assert got == {"stories_gpt_384w6l.pt": ("stories", "models"),
                   "web_gpt_768w12l.pt": ("web", "models"),
                   "web_gpt_768w12l_anneal.pt": ("web", "models"),
                   "snapshots/web_gpt_768w12l_step64.pt": ("web", "snapshots")}

def test_pieces_use_the_tokenisers_own_end_of_text():
    """A bigger tokeniser has a real token at the story EOT_ID, 4095."""
    texts = ["the cat sat on the mat and the dog ran"] * 5
    tok = WordBPE.train(texts, 262)
    assert tok.eot_id == 256 + len(tok.merges)
    assert pg.piece(tok, tok.eot_id) == "<|endoftext|>" and pg.token_bytes(tok, tok.eot_id) == b""
    assert pg.piece(tok, 256) == tok.vocab[256].decode()

def test_story_tokeniser_keeps_its_end_of_text_id(tok):
    assert tok.eot_id == EOT_ID == VOCAB_SIZE - 1

def test_chat_models_open_the_chat_tab(tmp_path):
    for name in ["web_gpt_768w12l_anneal25_chat.pt", "web_gpt_768w12l_anneal25_chat_s250.pt", "web_gpt_768w12l.pt"]:
        (tmp_path / name).touch()
    kinds = {m["path"]: (m["kind"], m["family"]) for m in pg.list_models(tmp_path)}
    assert kinds == {"web_gpt_768w12l.pt": ("story", "web"),
                     "web_gpt_768w12l_anneal25_chat.pt": ("chat", "web"),
                     "web_gpt_768w12l_anneal25_chat_s250.pt": ("chat", "web")}

def test_models_fine_tuned_from_the_web_model_use_its_tokeniser():
    for name in ["stories_instruct_web_gpt_768w12l_anneal_300mb_6k.pt", "web_gpt_768w12l_anneal25_chat.pt"]:
        assert pg.family(Path(name)) == "web"
    assert pg.family(Path("stories_instruct_384w6l_full.pt")) == "stories"
    assert pg.kind(Path("stories_instruct_web_gpt_768w12l_anneal_300mb.pt")) == "instruct"

def test_an_error_while_streaming_is_one_more_line(server, monkeypatch):
    """Tokens are already sent, so the error can't be a new HTTP response."""
    def broken(*args, **kwargs):
        yield {"id": 1, "piece": "a", "text": "a", "p": 0.5, "top": []}
        raise KeyError(6245)
    monkeypatch.setattr(pg, "sample", broken)
    lines = [json.loads(l) for l in call(server + "/api/generate", {"model": "stories_gpt_tiny.pt", "text": "Hi"}).splitlines()]
    assert [m["type"] for m in lines] == ["start", "token", "error"]
    assert "KeyError" in lines[-1]["error"]

@torch.no_grad()
@pytest.mark.parametrize("smear", [False, True])
def test_attention_row_matches_the_full_recorder(smear):
    """attention_from computes only the clicked token's row; it must equal that row of
    interp.Recorder's full T x T maps."""
    from tools.interp import Recorder
    model, ids = tiny(smear=smear), [5, 9, 2, 7, 7, 1, 3]
    offset, rows = pg.attention_from(model, ids, 4)
    with Recorder(model) as rec:
        model(torch.tensor([ids[:5]]))
    for i in range(len(model.blocks)):
        assert torch.allclose(torch.tensor(rows[i]), rec.attn[i][0, :, -1], atol=1e-4)
    assert offset == 0

@torch.no_grad()
def test_last_only_logits_match_the_full_forward():
    model = tiny()
    X = torch.randint(0, VOCAB_SIZE, (2, 10))
    assert torch.allclose(model(X, last_only=True), model(X)[:, -1:], atol=1e-6)

def test_only_the_two_most_recent_models_stay_loaded(tmp_path):
    for n in "abc":
        gpt.save(tiny(), tmp_path / f"stories_gpt_{n}.pt")
    models = pg.Models(tmp_path)
    a = models.get("stories_gpt_a.pt")
    models.get("stories_gpt_b.pt")
    assert models.get("stories_gpt_a.pt") is a          # used again, so the most recent
    models.get("stories_gpt_c.pt")                       # b is dropped, not a
    assert [p.name for p in models.cache] == ["stories_gpt_a.pt", "stories_gpt_c.pt"]

# ---------- visualisations ----------
@torch.no_grad()
def test_entropy_runs_from_certain_to_uniform():
    assert pg.entropy(torch.tensor([1.0, 0.0, 0.0])) == pytest.approx(0, abs=1e-6)
    assert pg.entropy(torch.full((8,), 1 / 8)) == pytest.approx(math.log(8))

@torch.no_grad()
def test_logit_lens_last_row_is_the_models_prediction(tok):
    model, ids = tiny(), [5, 9, 2, 7, 7, 1, 3, 8]
    start, rows = pg.logit_lens(model, tok, ids, 2, len(ids))
    assert start == 2 and len(rows) == len(model.blocks) + 1
    p = torch.softmax(model(torch.tensor([ids]))[0], dim=-1)
    for j, t in enumerate(range(2, len(ids))):
        assert rows[-1]["top"][j] == pg.piece(tok, p[t].argmax().item())
        assert rows[-1]["p"][j] == pytest.approx(p[t].max().item(), abs=1e-3)
    assert rows[-1]["next"][0] == pytest.approx(p[2, ids[3]].item(), abs=1e-3)
    assert rows[0]["next"][-1] is None                      # nothing came after the last token

@torch.no_grad()
@pytest.mark.parametrize("smear", [False, True])
def test_attention_matrix_matches_the_recorder(smear):
    from tools.interp import Recorder
    model, ids = tiny(smear=smear), [5, 9, 2, 7, 7, 1, 3]
    with Recorder(model) as rec:
        model(torch.tensor([ids]))
    start, matrix, outside = pg.attention_matrix(model, ids, 1, 2, 0, len(ids))
    assert torch.allclose(torch.tensor(matrix), rec.attn[1][0, 2], atol=1e-3)
    assert max(outside) < 1e-3
    start, matrix, outside = pg.attention_matrix(model, ids, 1, 2, 3, len(ids))       # a window
    assert start == 3 and len(matrix) == len(matrix[0]) == 4
    for row, out in zip(matrix, outside):
        assert sum(row) + out == pytest.approx(1, abs=1e-3)

@torch.no_grad()
def test_head_map_grids():
    model = tiny()
    offset, grids, repeats = pg.head_map(model, [5, 9, 2, 7, 5, 9, 2, 7])
    assert offset == 0 and repeats == 4                     # the whole second copy: 5 9 2 7
    for g in grids.values():
        assert len(g) == len(model.blocks) and len(g[0]) == model.blocks[0].attn.n_head
        assert all(0 <= v <= 1 for row in g for v in row)
    _, grids, repeats = pg.head_map(model, [1, 2, 3, 4])
    assert repeats == 0 and grids["induction"][0][0] is None

def test_visualisation_endpoints(server):
    base = {"model": "stories_gpt_tiny.pt", "ids": [1, 2, 3, 1, 2, 3, 4]}
    lens = json.loads(call(server + "/api/lens", {**base, "start": 2, "end": 7}))
    assert lens["start"] == 2 and len(lens["rows"][0]["top"]) == 5
    m = json.loads(call(server + "/api/matrix", {**base, "layer": 0, "head": 1, "start": 0, "end": 7}))
    assert len(m["matrix"]) == 7
    hm = json.loads(call(server + "/api/headmap", base))
    assert set(hm) == {"offset", "prev", "first", "induction", "repeats"}
    with pytest.raises(urllib.error.HTTPError):
        call(server + "/api/matrix", {**base, "layer": 9, "head": 0, "start": 0, "end": 7})

# ---------- ablation ----------
def test_parse_heads_checks_the_model():
    model = tiny()
    assert pg.parse_heads([[1, 2], [0, 0], [1, 2]], model) == [(0, 0), (1, 2)]
    assert pg.parse_heads(None, model) == []
    for bad in ([[2, 0]], [[0, 3]], [[-1, 0]]):
        with pytest.raises(ValueError):
            pg.parse_heads(bad, model)

@torch.no_grad()
def test_ablation_changes_the_output_and_leaves_no_hooks(tok):
    model, ids = tiny(), tok.encode("Once upon a time")
    normal = [t["id"] for t in pg.sample(model, tok, ids, 8, temperature=0)]
    heads = [(l, h) for l in range(2) for h in range(3)]
    off = [t["id"] for t in pg.sample(model, tok, ids, 8, temperature=0, ablate=heads)]
    assert off != normal
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules())
    assert [t["id"] for t in pg.sample(model, tok, ids, 8, temperature=0)] == normal    # nothing left on

@torch.no_grad()
def test_ablation_reaches_the_views(tok):
    model, ids = tiny(), [5, 9, 2, 7, 5, 9, 2, 7]
    _, normal = pg.logit_lens(model, tok, ids, 0, len(ids))
    _, off = pg.logit_lens(model, tok, ids, 0, len(ids), ablate=[(0, 0), (0, 1), (0, 2)])
    assert normal[-1]["p"] != off[-1]["p"] and normal[0] == off[0]    # the embeddings don't change
    _, g_normal, _ = pg.head_map(model, ids)
    _, g_off, _ = pg.head_map(model, ids, ablate=[(0, 0), (0, 1), (0, 2)])
    assert g_normal["prev"][0] == g_off["prev"][0] and g_normal["prev"][1] != g_off["prev"][1]

def test_ablation_through_the_server(server):
    base = {"model": "stories_gpt_tiny.pt", "text": "Hi", "max_tokens": 3, "temperature": 0}
    start = json.loads(call(server + "/api/generate", {**base, "ablate": [[1, 2], [0, 1]]}).splitlines()[0])
    assert start["ablate"] == [[0, 1], [1, 2]]
    with pytest.raises(urllib.error.HTTPError):
        call(server + "/api/generate", {**base, "ablate": [[5, 0]]})
    with pytest.raises(urllib.error.HTTPError):
        call(server + "/api/headmap", {"model": "stories_gpt_tiny.pt", "ids": [1, 2, 3], "ablate": [[0, 9]]})
