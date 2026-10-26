"""Fast checks of the sparse autoencoder (tools/sae.py): tiny sizes on the CPU, a few seconds in all."""
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
    values, where, windows = S.top_contexts(model, sae, 1, tokens, "cpu", n_tokens=200, top=3, B=4)
    T = model.block
    _, f = sae(S.residuals(model, 1, windows))
    for i in range(20):
        for v, flat in zip(values[:, i].tolist(), where[:, i].tolist()):
            if v > 0:
                w, t = divmod(flat, T)
                assert f[w, t, i].item() == pytest.approx(v, rel=1e-4)

def test_save_and_load_give_the_same_sae(tmp_path):
    torch.manual_seed(0)
    sae = S.SAE(C=8, M=16, k=3)
    sae.set_scale(torch.randn(10, 8))
    S.save(sae, tmp_path / "s.pt", layer=1)
    back, info = S.load(tmp_path / "s.pt")
    x = torch.randn(5, 8)
    assert torch.equal(sae(x)[0], back(x)[0]) and info == {"layer": 1}
