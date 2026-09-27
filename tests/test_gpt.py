"""Fast checks of the model code: tiny models on the CPU, a few seconds in all.

Each test guards against a real bug or near-miss, named in its docstring.

    pytest              # from the repo root
    pytest -k smear     # only the tests with smear in their name or options
"""
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import gpt
from interp import Ablate, Recorder
from stories_gpt import fine_tuned_path

TINY = dict(block=16, emb=12, heads=3, layers=2, dropout=0.0, vocab=50)
OPTIONS = [dict(gelu=g, tied=t, smear=s) for g, t, s in itertools.product([False, True], repeat=3)]
CHECKPOINTS = Path(__file__).parent.parent / "checkpoints"

def name(options):
    """A readable test ID, like gelu-tied or plain."""
    return "-".join(k for k, v in options.items() if v) or "plain"

def tiny(**options):
    torch.manual_seed(0)
    return gpt.GPT(**TINY, **options)

def tokens(B=2):
    torch.manual_seed(1)
    return torch.randint(0, TINY["vocab"], (B, TINY["block"]))

with_options = pytest.mark.parametrize("options", OPTIONS, ids=name)

# ---------- building the model ----------
@with_options
def test_forward_shape(options):
    X = tokens()
    assert tiny(**options)(X).shape == (*X.shape, TINY["vocab"])

@with_options
def test_options_build_that_model(options):
    """smear was once saved in the config but never passed on to the blocks."""
    model = tiny(**options)
    assert model.config == dict(TINY, **options)
    smear_weights = [n for n, _ in model.named_parameters() if n.endswith("attn.a")]
    assert len(smear_weights) == (TINY["layers"] if options["smear"] else 0)
    acts = [type(m) for m in model.modules() if isinstance(m, (nn.GELU, nn.ReLU))]
    assert acts == [nn.GELU if options["gelu"] else nn.ReLU] * TINY["layers"]
    assert (model.out.weight is model.tok.weight) == options["tied"]
    assert (model.out.bias is None) == options["tied"]

def test_old_configs_still_build():
    """Checkpoints from before vocab, gelu, tied and smear have only the first five keys."""
    old = {k: TINY[k] for k in ("block", "emb", "heads", "layers", "dropout")}
    model = gpt.GPT(**old)
    assert model.config == dict(old, vocab=gpt.V, gelu=False, tied=False, smear=False)

# ---------- training ----------
@with_options
def test_every_parameter_gets_a_gradient(options):
    """A parameter made in forward() instead of __init__ is never trained."""
    model = tiny(**options)
    X = tokens()
    gpt.lm_loss(model, X, X).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing

def test_smear_weight_learns():
    model = tiny(smear=True)
    opt = torch.optim.AdamW(gpt.param_groups(model, 0.1), lr=0.1)
    before = model.blocks[0].attn.a.detach().clone()
    X = tokens()
    gpt.lm_loss(model, X, X).backward()
    opt.step()
    assert not torch.equal(before, model.blocks[0].attn.a)

@with_options
def test_param_groups(options):
    """A (n_head, 1, 1) smear weight would have 3 dimensions, and get weight decay."""
    model = tiny(**options)
    decay, no_decay = gpt.param_groups(model, 0.1)
    ids = [id(p) for g in (decay, no_decay) for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())    # each one, exactly once
    # Weight decay for the matrices of the Linear and Embedding layers, and nothing else:
    # not biases, LayerNorms or smear weights, whatever their shape.
    matrices = {id(m.weight) for m in model.modules() if isinstance(m, (nn.Linear, nn.Embedding))}
    assert {id(p) for p in decay["params"]} == matrices
    assert decay["weight_decay"] > 0 and no_decay["weight_decay"] == 0

# ---------- behaviour ----------
@with_options
@torch.no_grad()
def test_causal(options):
    """Changing token t must not change any output before t. Smeared keys mix in t - 1, never t + 1."""
    model = tiny(**options).eval()
    X = tokens(B=1)
    before = model(X)
    X[0, 6] = (X[0, 6] + 1) % TINY["vocab"]
    after = model(X)
    assert torch.allclose(before[0, :6], after[0, :6])
    assert not torch.allclose(before[0, 6], after[0, 6])

@with_options
@torch.no_grad()
def test_save_load_round_trip(options, tmp_path):
    model = tiny(**options).eval()
    path = tmp_path / "model.pt"
    gpt.save(model, path)
    loaded = gpt.load("cpu", path)
    X = tokens()
    assert loaded.config == model.config
    assert torch.equal(loaded(X), model(X))
    if options["tied"]:
        assert loaded.out.weight is loaded.tok.weight    # still one matrix after loading

@with_options
@torch.no_grad()
def test_recorder_rebuilds_attention(options):
    """The hook once computed the keys without smearing, so it recorded the wrong weights."""
    model = tiny(**options).eval()
    with Recorder(model) as rec:
        model(tokens())
    for i, block in enumerate(model.blocks):
        attn, w = block.attn, rec.attn[i]
        x = block.ln1(rec.resid[i])
        v = attn.split_heads(attn.qkv(x).split(x.shape[-1], dim=-1)[2])
        assert torch.allclose(attn.proj(attn.join_heads(w @ v)), attn(x), atol=1e-6)
    assert all(len(m._forward_hooks) == 0 for m in model.modules())

def test_batch_from_numpy_and_torch():
    ids = np.arange(100, dtype=np.uint16)
    X, Y = gpt.batch(ids, 4, 8, "cpu")
    assert X.dtype == torch.int64 and X.shape == (4, 8)
    assert torch.equal(Y, X + 1)                # the targets are the next tokens
    X, Y = gpt.batch(torch.arange(100), 4, 8, "cpu")
    assert torch.equal(Y, X + 1)

def test_fine_tuned_path():
    sft = Path("checkpoints/stories_sft.pt")
    assert fine_tuned_path(sft, Path("checkpoints/stories_gpt.pt")) == sft
    assert fine_tuned_path(sft, Path("checkpoints/stories_gpt_384w6l.pt")).name == "stories_sft_384w6l.pt"

# ---------- the real checkpoints ----------
@pytest.mark.parametrize("path", sorted(p for p in CHECKPOINTS.glob("*.pt") if not p.name.endswith(".resume.pt")),
                         ids=lambda p: p.name)
def test_checkpoint_loads(path):
    """New options once gave every model a parameter that old checkpoints don't have."""
    gpt.load("cpu", path)

# ---------- run info and snapshots ----------
def test_run_info_saved(tmp_path):
    path = tmp_path / "model.pt"
    gpt.save(tiny(), path)
    info = gpt.load_info(path)
    assert len(info["commit"]) == 40 and isinstance(info["dirty"], bool) and info["argv"]
    assert info["dirty"] == bool(info["diff"])

def test_snapshot_steps():
    assert [n for n in range(1, 1100) if gpt.is_snapshot_step(n)] == [64, 128, 256, 512, 1024]

def test_training_saves_snapshots(tmp_path):
    model = tiny()
    ids = torch.randint(0, TINY["vocab"], (500,))
    gpt.train_model(model, "cpu", STEPS=130, B=2, WARMUP=10, log_every=0,
                    train_ids=ids, val_ids=ids, snapshots=tmp_path / "snaps" / "tiny")
    assert sorted(p.name for p in (tmp_path / "snaps").iterdir()) == ["tiny_step128.pt", "tiny_step64.pt"]
    gpt.load("cpu", tmp_path / "snaps" / "tiny_step64.pt")

# ---------- head ablation ----------
@with_options
@torch.no_grad()
def test_ablation_changes_output_then_goes_away(options):
    model = tiny(**options).eval()
    X = tokens()
    before = model(X)
    with Ablate(model, layer=1, head=2):
        during = model(X)
    after = model(X)
    assert not torch.allclose(before, during)
    assert torch.equal(before, after)                     # the hook is gone again
    assert all(len(m._forward_pre_hooks) == 0 for m in model.modules())

@pytest.mark.parametrize("layer, head", [(0, 0), (1, 1), (1, 2)])
def test_ablation_zeroes_only_that_head(layer, head):
    """A slice from len(x) (the batch size) would zero the wrong channels, or none."""
    model = tiny()
    hs = TINY["emb"] // TINY["heads"]
    x = torch.randn(2, 5, TINY["emb"])
    with Ablate(model, layer, head) as abl:
        out = abl.zero_head(model.blocks[layer].attn.proj, (x,))
    mine = slice(head * hs, (head + 1) * hs)
    assert (out[..., mine] == 0).all()
    others = torch.ones(TINY["emb"], dtype=torch.bool)
    others[mine] = False
    assert torch.equal(out[..., others], x[..., others])
    assert not (x[..., mine] == 0).all()                  # x itself unchanged: the hook copies first

@torch.no_grad()
def test_ablating_every_head_leaves_only_the_bias():
    model = tiny().eval()
    proj = model.blocks[0].attn.proj
    seen = []
    handle = proj.register_forward_hook(lambda m, i, o: seen.append(o))
    ablations = [Ablate(model, 0, h) for h in range(TINY["heads"])]
    for a in ablations:
        a.__enter__()
    model(tokens())
    for a in ablations:
        a.__exit__()
    handle.remove()
    assert torch.allclose(seen[0], proj.bias.expand_as(seen[0]))
