"""tools/gpt2.py: reading safetensors, converting GPT-2's names, and its tokeniser.

The conversion is checked against reference(): GPT-2's forward pass written straight
on GPT-2's own tensors, as OpenAI's code does it. Random weights make every mistake
show, including a missed transpose of a square matrix, where the shapes still fit.
"""
import json
import struct

import pytest
import torch
import torch.nn.functional as F

from core import gpt
from tools import gpt2

TINY = dict(gpt2.CONFIG, block=16, emb=12, heads=3, layers=2, vocab=50)

def fake_gpt2(block=16, C=12, layers=2, vocab=50):
    """Random tensors with GPT-2's names and shapes, for a tiny size."""
    torch.manual_seed(0)
    r = lambda *shape: torch.randn(*shape) * 0.3
    hf = {"wte.weight": r(vocab, C), "wpe.weight": r(block, C), "ln_f.weight": 1 + r(C), "ln_f.bias": r(C)}
    for i in range(layers):
        h = f"h.{i}."
        hf |= {h + "ln_1.weight": 1 + r(C), h + "ln_1.bias": r(C), h + "ln_2.weight": 1 + r(C), h + "ln_2.bias": r(C),
               h + "attn.c_attn.weight": r(C, 3 * C), h + "attn.c_attn.bias": r(3 * C),
               h + "attn.c_proj.weight": r(C, C), h + "attn.c_proj.bias": r(C),
               h + "mlp.c_fc.weight": r(C, 4 * C), h + "mlp.c_fc.bias": r(4 * C),
               h + "mlp.c_proj.weight": r(4 * C, C), h + "mlp.c_proj.bias": r(C),
               h + "attn.bias": torch.tril(torch.ones(block, block)).view(1, 1, block, block)}
    return hf

def reference(hf, ids, heads, layers):
    """GPT-2's forward pass on its own tensors: Conv1D is x @ W + b."""
    B, T = ids.shape
    C = hf["wte.weight"].shape[1]
    ln = lambda x, name: F.layer_norm(x, (C,), hf[name + ".weight"], hf[name + ".bias"])
    conv1d = lambda x, name: x @ hf[name + ".weight"] + hf[name + ".bias"]
    split = lambda t: t.view(B, T, heads, C // heads).transpose(1, 2)
    x = hf["wte.weight"][ids] + hf["wpe.weight"][:T]
    for i in range(layers):
        h = f"h.{i}."
        q, k, v = conv1d(ln(x, h + "ln_1"), h + "attn.c_attn").split(C, dim=-1)
        a = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
        x = x + conv1d(a.transpose(1, 2).reshape(B, T, C), h + "attn.c_proj")
        x = x + conv1d(F.gelu(conv1d(ln(x, h + "ln_2"), h + "mlp.c_fc"), approximate="tanh"), h + "mlp.c_proj")
    return ln(x, "ln_f") @ hf["wte.weight"].T

def write_safetensors(path, tensors):
    """The format, written by hand: an 8-byte length, a JSON header, then the raw floats."""
    header, data = {"__metadata__": {"format": "pt"}}, b""
    for name, t in tensors.items():
        raw = t.contiguous().numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + data)

# ---------- safetensors ----------
def test_read_safetensors(tmp_path):
    tensors = {"a": torch.arange(6.0).view(2, 3), "b.weight": torch.randn(4), "c": torch.randn(3, 1, 2)}
    write_safetensors(tmp_path / "t.safetensors", tensors)
    got = gpt2.read_safetensors(tmp_path / "t.safetensors")
    assert set(got) == set(tensors)               # no "__metadata__"
    for name, t in tensors.items():
        assert got[name].dtype == torch.float32 and torch.equal(got[name], t)

# ---------- the conversion ----------
@torch.no_grad()
def test_convert_matches_gpt2s_own_forward_pass():
    hf = fake_gpt2()
    model = gpt.GPT(**TINY).eval()
    model.load_state_dict(gpt2.convert(hf, layers=2))     # strict: no key missing, none left over
    ids = torch.randint(0, 50, (2, 16))
    assert torch.allclose(model(ids), reference(hf, ids, heads=3, layers=2), atol=1e-4)

@torch.no_grad()
def test_convert_keeps_the_embeddings_tied():
    model = gpt.GPT(**TINY)
    model.load_state_dict(gpt2.convert(fake_gpt2(), layers=2))
    assert model.out.weight is model.tok.weight

def test_convert_leaves_out_the_masks():
    assert not any("bias" in k and "attn" in k and "qkv" not in k and "proj" not in k
                   for k in gpt2.convert(fake_gpt2(), layers=2))

# ---------- the real files ----------
needs = lambda name: pytest.mark.skipif(not (gpt2.DIR / name).exists(), reason=f"no {gpt2.DIR / name}")

@pytest.fixture(scope="module")
def tok():
    return gpt2.GPT2Tokenizer()

@needs("merges.txt")
def test_tokeniser_matches_gpt2s_ids(tok):
    assert tok.encode("Hello world") == [15496, 995]
    assert tok.eot_id == 50256 and sorted(tok.to_gpt2) == list(range(50257))
    s = "Tom's cat, née Müller, costs £3.50!\n\n\tDone."
    assert tok.decode(tok.encode(s)) == s

@needs("model.safetensors")
def test_gpt2_knows_paris(tok):
    model = gpt2.load()
    top = [t for t, _ in gpt2.top_next(model, tok, "The capital of France is the city of", k=3)]
    assert " Paris" in top
