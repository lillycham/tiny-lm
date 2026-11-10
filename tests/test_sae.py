"""Fast checks of the sparse autoencoder (tools/sae.py): tiny sizes on the CPU, a few seconds in all."""
import json

import numpy as np
import pytest
import torch

from core import gpt
from tools import sae as S

TINY = dict(block=16, emb=12, heads=3, layers=2, dropout=0.0, vocab=50)

def tiny_model():
    torch.manual_seed(0)
    return gpt.GPT(**TINY).eval()

def test_topk_keeps_at_most_k_features_and_none_below_zero():
    """TopK is the whole sparsity: if it kept more than k, the SAE would not be sparse."""
    torch.manual_seed(0)
    sae = S.SAE(C=8, M=64, k=5)
    f = sae.encode(torch.randn(100, 8))
    assert ((f != 0).sum(-1) <= 5).all()
    assert (f >= 0).all()

def test_decoder_columns_stay_length_one():
    """Without this, the SAE can shrink f and grow the columns, and feature sizes mean nothing."""
    torch.manual_seed(0)
    sae = S.SAE(C=8, M=32, k=4)
    assert torch.allclose(sae.dec.weight.norm(dim=0), torch.ones(32), atol=1e-5)
    with torch.no_grad():
        sae.dec.weight.mul_(3.0)
    sae.normalise_decoder()
    assert torch.allclose(sae.dec.weight.norm(dim=0), torch.ones(32), atol=1e-5)

def test_scale_makes_the_average_length_sqrt_c_and_does_not_change_the_units():
    """The caller gives and gets the model's own vectors: scale works only inside."""
    torch.manual_seed(0)
    sae = S.SAE(C=16, M=64, k=64)                 # (k = M: nothing is dropped)
    x = 50 * torch.randn(200, 16)
    sae.set_scale(x)
    assert (x * sae.scale).norm(dim=-1).mean().item() == pytest.approx(4.0, rel=1e-4)
    f = sae.encode(x)
    assert torch.allclose(sae.decode(f), (sae.dec(f)) / sae.scale)

def test_it_finds_a_planted_sparse_code():
    """Vectors made of 2 out of 24 known directions: a working SAE explains nearly all of them."""
    torch.manual_seed(0)
    C, M, k = 16, 24, 2
    truth = torch.nn.functional.normalize(torch.randn(M, C), dim=-1)
    def batch(n=512):
        idx = torch.rand(n, M).argsort(-1)[:, :k]
        coef = torch.zeros(n, M).scatter(-1, idx, 1 + torch.rand(n, k))
        return coef @ truth
    sae = S.SAE(C, 2 * M, k)
    sae.set_scale(batch())
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    for _ in range(1500):
        x = batch()
        x_hat, _ = sae(x)
        loss = (x - x_hat).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalise_decoder()
    assert S.variance_explained(sae, batch(2048)) > 0.9

def test_residuals_match_the_model_up_to_the_block():
    """residuals() runs the blocks by hand: it must give what the model computes."""
    model = tiny_model()
    X = torch.randint(0, 50, (2, 10))
    seen = {}
    handle = model.blocks[0].register_forward_hook(lambda m, i, out: seen.setdefault("x", out))
    model(X)
    handle.remove()
    assert torch.allclose(S.residuals(model, 0, X), seen["x"], atol=1e-6)

def test_stream_leaves_out_position_zero():
    model = tiny_model()
    tokens = np.arange(1000) % 50
    x = next(S.stream(model, 1, tokens, B=3, T=8, device="cpu"))
    assert x.shape == (3 * 7, 12)

def test_splicing_in_the_identity_changes_nothing():
    """The hook plumbing: replacing a block's output with itself must give the clean loss."""
    model = tiny_model()
    X, Y = torch.randint(0, 50, (2, 10)), torch.randint(0, 50, (2, 10))
    clean = gpt.lm_loss(model, X, Y).item()
    assert S.spliced_loss(model, 0, X, Y, lambda out: out) == pytest.approx(clean, abs=1e-6)
    assert S.spliced_loss(model, 0, X, Y, lambda out: torch.zeros_like(out)) != pytest.approx(clean, abs=1e-3)

def test_top_contexts_point_at_the_tokens_where_each_feature_fired():
    """The (window, position) bookkeeping: each stored place must hold the stored value."""
    model = tiny_model()
    torch.manual_seed(0)
    sae = S.SAE(C=12, M=20, k=4)
    tokens = np.arange(5000) % 50
    scan = S.top_contexts(model, sae, 1, tokens, "cpu", n_tokens=200, top=3, B=4)
    T = model.block
    _, f = sae(S.residuals(model, 1, scan.windows))
    for i in range(20):
        for v, flat in zip(scan.values[:, i].tolist(), scan.where[:, i].tolist()):
            if v > 0:
                w, t = divmod(flat, T)
                assert f[w, t, i].item() == pytest.approx(v, rel=1e-4)
    f[:, 0] = 0                                          # (position 0: not counted, see stream())
    assert scan.seen == len(scan.windows) * (T - 1)
    assert scan.counts.tolist() == (f > 0).sum(dim=(0, 1)).tolist()

def test_shortlist_leaves_out_dead_very_common_and_token_features():
    """A feature on half the tokens is usually a common token, not an idea; a dead one
    says nothing; and one whose top contexts are all the same token is just that token.
    The rest come strongest first."""
    same = [[10.0, "", "x", ""]] * 3
    diverse = [[4.0, "", t, ""] for t in "xyz"]
    data = {"features": [[[5.0, "", "a", ""]], [], [[1.0, "", "b", ""]], [[9.0, "", "c", ""]], same, diverse],
            "counts": [500, 0, 1, 5000, 100, 100], "seen": 10_000}
    assert S.shortlist(data, n=10, min_rate=1e-4, max_rate=0.3) == [0, 5, 2]
    assert S.shortlist(data, n=10, min_distinct=1) == [4, 0, 5, 2]          # (feature 3 is still too common)
    assert S.shortlist({"features": data["features"]}, n=10) == [3, 0, 5, 2]

def test_labels_survive_a_round_trip_and_an_empty_label_clears(tmp_path):
    path = tmp_path / "sae" / "s.pt"
    path.parent.mkdir()
    path.touch()
    assert S.load_labels(path) == {}
    S.set_label(path, 7, " animals ")
    assert S.load_labels(path) == {"7": "animals"}
    S.set_label(path, 2, "furniture")
    assert S.load_labels(path) == {"2": "furniture", "7": "animals"}
    S.set_label(path, 7, "")
    assert S.load_labels(path) == {"2": "furniture"}
    assert json.loads(S.labels_path(path).read_text()) == {"2": "furniture"}

def test_ranges_of_the_shortlist_give_out_different_features():
    """Splitting the work over several passes is only safe if two ranges cannot overlap,
    and if the ranges together are the one list a single pass would have had."""
    rows = [[[float(9 - i), "", t, ""] for t in "xyz"] for i in range(6)]
    data = {"features": rows, "counts": [10] * 6, "seen": 10_000}
    assert S.slice_shortlist(data, 0, 2) == [0, 1]
    assert S.slice_shortlist(data, 2, 2) == [2, 3]
    assert S.slice_shortlist(data, 0, 4) == S.slice_shortlist(data, 0, 2) + S.slice_shortlist(data, 2, 2)
    assert S.slice_shortlist(data, 8, 2) == []                 # (past the end: no work to hand out)

def test_a_slice_file_is_the_whole_briefing(tmp_path):
    """What a pass is given: the contexts, the rate, and the exact file to write. It is
    not told anything else, so this file has to carry the style guide too."""
    path = tmp_path / "s.pt"
    data = {"features": [[[3.0, "the ", "dog", " barked"]], [[1.0, "", "x", ""]]],
            "counts": [5, 1], "seen": 10_000}
    out = S.write_slice(path, data, [0, 1], "try")
    assert out == S.work_path(path, "try")
    text = out.read_text()
    assert text.index("feature 0  peak 3.00") < text.index("feature 1  peak 1.00")
    assert "…the [dog] barked" in text
    assert "fires on 0.05% of 10,000 tokens" in text
    assert str(S.staged_path(path, "try")) in text              # (where the names go)
    assert "unclear (a or b)" in text                           # (the style guide, whole)

def test_merge_takes_what_agrees_and_leaves_the_rest_to_a_person(tmp_path):
    """Two passes that agree are one answer. Two that do not, or one that disagrees with
    a name already there, are a question: nothing is written for them unless forced."""
    path = tmp_path / "s.pt"
    path.touch()
    S.examples_path(path).write_text(json.dumps({"features": [[] for _ in range(10)]}))
    S.staging_dir(path).mkdir()
    S.set_label(path, 1, "already here")
    S.staged_path(path, "a").write_text(json.dumps({"1": "something else", "2": "flowers", "3": "questions"}))
    S.staged_path(path, "b").write_text(json.dumps({"3": "questions", "4": "one", "5": "x"}))
    S.staged_path(path, "c").write_text(json.dumps({"4": "two"}))

    merged = S.merge_staged(path)
    assert S.load_labels(path) == {"1": "already here", "2": "flowers", "3": "questions", "5": "x"}
    assert merged.added == {"2": "flowers", "3": "questions", "5": "x"}
    assert merged.same == set()
    assert merged.conflicts == {"1": ["already here", "something else"], "4": ["one", "two"]}

    S.staged_path(path, "later").write_text(json.dumps({"9": "only from later"}))
    assert S.merge_staged(path, ["later"]).added == {"9": "only from later"}   # (one pass at a time)

    merged = S.merge_staged(path, force=True)
    assert merged.added == {"1": "something else", "4": "one"}   # (the first pass in name order)
    assert S.load_labels(path)["4"] == "one"

    S.staged_path(path, "d").write_text(json.dumps({"11": "outside"}))
    with pytest.raises(ValueError, match="outside"):
        S.merge_staged(path)
    S.staged_path(path, "d").unlink()
    S.staged_path(path, "e").write_text(json.dumps({"not a number": "x"}))
    with pytest.raises(ValueError, match="not a feature number"):
        S.merge_staged(path)
    assert "11" not in S.load_labels(path)                      # (a broken file writes nothing)

def test_an_audit_shows_a_name_beside_contexts_it_was_not_written_from(tmp_path):
    """The point of checking rather than labelling again: the contexts are found in other
    text at another seed, so a name that only fits its own examples shows up as wrong."""
    model, path = tiny_model(), tmp_path / "s.pt"
    torch.manual_seed(0)
    sae = S.SAE(C=TINY["emb"], M=64, k=8)
    tok = type("Tok", (), {"decode": lambda self, ids: "".join(chr(97 + i % 26) for i in ids)})()
    S.set_label(path, 3, "letters")

    out, features = S.write_audit(path, model, sae, 0, np.zeros(4096, dtype=np.int64), "cpu",
                                  n_tokens=1024, offset=0, tok=tok)
    assert features == [3] and out == S.work_path(path, "audit")
    text = out.read_text()
    assert "label: letters" in text
    assert str(S.audit_path(path, "audit")) in text             # (where the verdicts go)
    assert '"verdict": "disagrees"' in text                     # (the shape to answer in)

def test_verdicts_are_read_back_with_what_they_suggest(tmp_path):
    path = tmp_path / "s.pt"
    path.touch()
    S.staging_dir(path).mkdir()
    S.audit_path(path, "one").write_text(json.dumps({
        "1": {"verdict": "agrees"},
        "2": {"verdict": "disagrees", "was": "old", "label": " verbs "},
        "3": "partly"}))
    assert S.read_audits(path) == {"1": ("agrees", "", ""), "2": ("disagrees", "verbs", "old"),
                                  "3": ("partly", "", "")}
    S.audit_path(path, "bad").write_text(json.dumps({"4": {"verdict": "maybe"}}))
    with pytest.raises(ValueError, match="maybe"):
        S.read_audits(path)

def test_a_verdict_is_done_once_its_name_or_the_suggestion_has_moved_on():
    """A check that keeps reporting a fix already made is noise, and so is one about a name
    that has since been rewritten: neither says anything about the name there now."""
    verdicts = {"1": ("disagrees", "verbs", ""),          # (suggestion taken)
                "2": ("partly", "nouns", ""),             # (still open)
                "3": ("agrees", "", ""),                  # (agreed)
                "4": ("disagrees", "nouns", "old name"),  # (name rewritten since)
                "5": ("disagrees", "nouns", "here")}      # (name unchanged: still open)
    labels = {"1": "verbs", "2": "nouns + verbs", "3": "anything", "4": "something else", "5": "here"}
    assert S.outstanding(verdicts, labels) == {"2": ("partly", "nouns", ""),
                                               "5": ("disagrees", "nouns", "here")}

def test_save_and_load_give_the_same_sae(tmp_path):
    torch.manual_seed(0)
    sae = S.SAE(C=8, M=16, k=3)
    sae.set_scale(torch.randn(10, 8))
    S.save(sae, tmp_path / "s.pt", layer=1)
    back, info = S.load(tmp_path / "s.pt")
    x = torch.randn(5, 8)
    assert torch.equal(sae(x)[0], back(x)[0]) and info == {"layer": 1}
