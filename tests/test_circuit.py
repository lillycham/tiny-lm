"""Fast checks of the name-circuit test (tools/circuit.py): prefixes, ablation and scores, on tiny models."""
import pytest
import torch

from core import gpt
from core.word_bpe import TOKENISER, WordBPE
from tools import circuit as C
from tools.interp import Ablate

TINY = dict(block=64, emb=12, heads=3, layers=2, dropout=0.0, vocab=50)

def tiny():
    torch.manual_seed(0)
    return gpt.GPT(**TINY).eval()

@pytest.fixture(scope="module")
def tok():
    if not TOKENISER.exists():
        pytest.skip("needs the tokeniser from word_bpe.py")
    return WordBPE.load(TOKENISER)

def test_stutter_adds_one_gle():
    assert C.stutter("Glegley") == "Gleglegley" and C.stutter("Gegley") == "Geglegley"

def test_prefixes_stop_before_the_last_y_and_count_mentions():
    p = C.prefix("Glegley", 3, False, "One day,")
    assert p == ("Once upon a time, there was a person named Glegley. Glegley was very kind. "
                 "Glegley liked to play in the park. One day, Glegle")
    assert C.prefix("Glegley", 1, True, "Then").startswith("Once upon a time, there was a person named Gleglegley.")
    assert len(C.conditions()) == 4 * 3 * 2 * len(C.FRAMES)

def test_every_prefix_ends_on_le(tok):
    """The score is read at the last "le": a prefix that ends elsewhere measures something else."""
    y, g = C.end_ids(tok)
    assert tok.decode([y]) == "y" and tok.decode([g]) == "g"
    for _, name, m, p, f in C.conditions():
        assert tok.decode(C.encode(tok, C.prefix(name, m, p, f))[-1:]) == "le"

def test_names_differ_only_in_the_repeated_piece(tok):
    """The comparison: repeated names have "le" twice, unique names once; same length, same ending."""
    for group, names in C.NAMES.items():
        for name in names:
            pieces = [tok.decode([i]) for i in tok.encode(" " + name)]
            assert len(pieces) == 5 and pieces[-2:] == ["le", "y"]
            assert pieces.count("le") == (2 if group == "repeated" else 1)

def test_mean_ablation_replaces_only_that_heads_slice():
    """What attn.proj receives: head 1's slice is the mean at every position, the other heads unchanged."""
    model, X = tiny(), torch.randint(0, 50, (2, 10))
    seen = {}
    capture = lambda m, i: seen.__setitem__("x", i[0].clone())
    means = torch.randn(2, 12)
    proj = model.blocks[0].attn.proj
    with C.MeanAblate(model, means, 0, 1):
        handle = proj.register_forward_pre_hook(capture)        # (after MeanAblate's, so it sees the result)
        model(X)
        handle.remove()
    ablated = seen["x"]
    handle = proj.register_forward_pre_hook(capture)
    model(X)
    handle.remove()
    assert torch.equal(ablated[:, :, 4:8], means[0, 4:8].expand(2, 10, 4))
    assert torch.equal(ablated[:, :, :4], seen["x"][:, :, :4]) and torch.equal(ablated[:, :, 8:], seen["x"][:, :, 8:])

def test_head_means_are_the_average_input_to_proj():
    model = tiny()
    tokens = torch.arange(500) % 50
    means = C.head_means(model, tokens.numpy(), chunks=4)
    assert means.shape == (2, 12) and torch.isfinite(means).all()

def test_mean_and_zero_ablation_change_the_output_and_leave_no_hooks():
    model, X = tiny(), torch.randint(0, 50, (2, 10))
    clean = model(X)
    means = torch.randn(2, 12)
    for how in ("mean", "zero"):
        with C.switched_off(model, how, means, [(0, 1), (1, 2)]):
            assert not torch.allclose(model(X), clean)
        assert torch.allclose(model(X), clean)

def test_scores_batch_by_length_and_match_one_at_a_time():
    model = tiny()
    prefixes = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11, 12]]
    got = C.scores(model, prefixes, 5, 7)
    for i, ids in enumerate(prefixes):
        logits = model(torch.tensor([ids]))[0, -1]
        assert got[i].item() == pytest.approx((logits[5] - logits[7]).item(), abs=1e-5)

def test_y_attention_sums_the_weight_on_y_positions():
    """Against the full attention of every head, from playground.attention_from."""
    from tools.playground import attention_from
    model = tiny()
    ids = [3, 7, 9, 7, 2, 7, 4]
    got = C.y_attention(model, [ids], 7)[0]                 # (layers, heads)
    _, weights = attention_from(model, ids, len(ids) - 1)
    for l in range(2):
        for h in range(3):
            want = sum(w for w, t in zip(weights[l][h], ids) if t == 7)
            assert got[l, h].item() == pytest.approx(want, abs=1e-3)

def test_report_writes_a_page(tmp_path):
    heads = [f"L{l}H{h}" for l in range(2) for h in range(6)]
    result = {"columns": ["a", "b"], "heads": heads, "clean": [1.0, -1.0],
              "heads_mean": [[0.1, -0.2]] * 12, "heads_zero": [[0.0, 0.3]] * 12,
              "y_attention": [0.1] * 12, "composition": [[0.0] * 12] * 12}
    C.report(result, tmp_path / "r.html")
    page = (tmp_path / "r.html").read_text()
    assert "Head x condition (mean-ablation)" in page and "L1H5" in page
