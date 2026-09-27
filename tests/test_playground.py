"""The playground's server: prompts, sampling, attention and the HTTP endpoints.

The HTTP tests start a real server on a free port, with a tiny model in a temporary
checkpoints folder, so they run in a second or two on the CPU.
"""
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import chat_sft
import gpt
import instruct_sft
import playground as pg
import sft
from word_bpe import EOT_ID, TOKENISER, VOCAB_SIZE, WordBPE


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

    def forward(self, x):
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
