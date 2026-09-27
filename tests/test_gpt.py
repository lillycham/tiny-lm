"""Fast checks of the model code: tiny models on the CPU, a few seconds in all.

Each test guards against a real bug or near-miss, named in its docstring.

    pytest              # from the repo root
    pytest -k smear     # only the tests with smear in their name or options
"""
import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import gpt
from interp import Ablate, Recorder, ablation_table, val_chunks, val_scores
import instruct_sft
from instruct_sft import output_path as instruct_path
from stories_gpt import TOKENS, fine_tuned_path

TINY = dict(block=16, emb=12, heads=3, layers=2, dropout=0.0, vocab=50)
OPTIONS = [dict(gelu=g, tied=t, smear=s, scaled_init=i) for g, t, s, i in itertools.product([False, True], repeat=4)]
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
    assert model.config == dict(old, vocab=gpt.V, gelu=False, tied=False, smear=False, scaled_init=False)

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

def test_instruct_output_path():
    """A bigger run once had no tag, so it would have replaced the default run's model."""
    assert instruct_path(Path("checkpoints/stories_gpt.pt")).name == "stories_instruct.pt"
    full = Path("checkpoints/stories_gpt_384w6l_full.pt")
    assert instruct_path(full).name == "stories_instruct_384w6l_full.pt"
    assert instruct_path(full, "300mb").name == "stories_instruct_384w6l_full_300mb.pt"

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

# ---------- validation-loss test ----------
has_val_tokens = pytest.mark.skipif(not Path(str(TOKENS).format(split="val")).exists(),
                                    reason="needs the val tokens from word_bpe.py")

@has_val_tokens
def test_val_chunks():
    X, Y = val_chunks(16, n=5)
    assert X.shape == Y.shape == (5, 16) and X.dtype == torch.int64
    assert torch.equal(X[:, 1:], Y[:, :-1])               # Y is X shifted by one
    again, _ = val_chunks(16, n=5)
    assert torch.equal(X, again)                          # the same chunks every call
    assert not torch.equal(X, val_chunks(16, n=5, seed=1)[0])

@torch.no_grad()
def test_val_scores():
    """-loss.mean(-1).tolist() negates a list; the mean without dim gives one number."""
    model = tiny().eval()
    X = tokens(B=4)
    Y = torch.roll(X, -1, dims=1)
    scores = val_scores(model, X, Y)
    assert isinstance(scores, list) and len(scores) == 4
    for i, s in enumerate(scores):                        # one chunk at a time
        assert s == pytest.approx(-gpt.lm_loss(model, X[i:i + 1], Y[i:i + 1]).item(), abs=1e-5)
    assert all(s < 0 for s in scores)                     # minus a loss: a head that helps makes it drop

@torch.no_grad()
def test_val_scores_in_ablation_table():
    model = tiny().eval()
    X = tokens(B=3)
    Y = torch.roll(X, -1, dims=1)
    tables = ablation_table(model, {"val loss": lambda m: val_scores(m, X, Y)})
    assert tables["val loss"].shape == (TINY["layers"], TINY["heads"])
    assert (tables["val loss"] != 0).any()

# ---------- resuming ----------
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
                  + (["cuda"] if torch.cuda.is_available() else [])

@pytest.mark.parametrize("device", DEVICES)
def test_resume(device, tmp_path, capsys):
    """torch.load(map_location=device) moved the saved RNG state to the GPU too, and
       set_rng_state only takes a CPU tensor. Resuming failed on every GPU."""
    ids = torch.randint(0, TINY["vocab"], (500,))
    resume = tmp_path / "tiny.resume.pt"
    def run():
        gpt.train_model(tiny().to(device), device, STEPS=20, B=2, WARMUP=2, log_every=0,
                        train_ids=ids, val_ids=ids, resume=resume, resume_minutes=0)
    run()
    assert resume.exists()
    run()                                                 # the second run carries on from the file
    assert "Carrying on from step" in capsys.readouterr().out

def test_default_device():
    device = gpt.default_device()
    assert device in gpt.DEVICES
    torch.zeros(1, device=device)                         # it really is there
    assert device == DEVICES[-1]                          # and it is the fastest one here

@pytest.mark.parametrize("device", DEVICES)
def test_device_option(device):
    import argparse
    parser = argparse.ArgumentParser()
    gpt.add_device_option(parser)
    assert parser.parse_args([]).device == gpt.default_device()
    assert parser.parse_args(["--device", device]).device == device
    gpt.synchronize(device)

# ---------- torch.compile ----------
@pytest.mark.parametrize("snapshot", [False, True])
def test_compiled_training_saves_plain_weights(tmp_path, snapshot, monkeypatch):
    """torch.compile wraps the model, and the wrapper's state_dict names every weight
       _orig_mod.<name>. Saved from the wrapper, a checkpoint won't load into a plain GPT."""
    # A torch.compile that records its use, with a backend that runs the graph as it is:
    # the same wrapper and the same names, but no C++ compiler, so the test is fast.
    real_compile, compiled, ran = torch.compile, [], []
    def backend(graph, example_inputs):
        ran.append(1)
        return graph.forward
    def recording_compile(m, **options):
        compiled.append(m)
        return real_compile(m, backend=backend)
    monkeypatch.setattr(torch, "compile", recording_compile)

    model = tiny()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ids = torch.randint(0, TINY["vocab"], (500,))
    resume = tmp_path / "tiny.resume.pt"
    gpt.train_model(model, "cpu", STEPS=70, B=2, WARMUP=2, log_every=0, train_ids=ids, val_ids=ids,
                    resume=resume, resume_minutes=0, compile=True,
                    snapshots=tmp_path / "snaps" / "tiny" if snapshot else None)
    assert compiled == [model] and ran                    # it trained through the compiled model
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters())   # and model learned
    names = set(model.state_dict())
    assert set(torch.load(resume, weights_only=False)["weights"]) == names
    if snapshot:
        assert set(torch.load(tmp_path / "snaps" / "tiny_step64.pt")["weights"]) == names
    gpt.save(model, tmp_path / "model.pt")
    gpt.load("cpu", tmp_path / "model.pt")

def test_no_compile_by_default(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda *a, **k: pytest.fail("compiled without compile=True"))
    ids = torch.randint(0, TINY["vocab"], (500,))
    gpt.train_model(tiny(), "cpu", STEPS=5, B=2, WARMUP=2, log_every=0, train_ids=ids, val_ids=ids)

def test_trains_on_shakespeare_by_default():
    """Without train_ids, train_model uses gpt.train, the Shakespeare text. A local
       variable named train in train_model once hid it."""
    model = gpt.GPT(**dict(TINY, vocab=gpt.V))
    gpt.train_model(model, "cpu", STEPS=3, B=2, WARMUP=1, log_every=0)

# ---------- SFT data ----------
@pytest.mark.skipif(not instruct_sft.VAL_FILE.exists() or not TOKENS.parent.joinpath("..", "checkpoints", "stories_bpe.json").exists(),
                    reason="needs TinyStories-Instruct-valid.txt and the tokeniser")
def test_instruct_dataset_is_compact_and_unchanged():
    """The full instruct file as Python lists needed ~17 GB, too much for a 32 GB machine."""
    from word_bpe import TOKENISER, WordBPE
    import sft
    tok = WordBPE.load(TOKENISER)
    X, Y = instruct_sft.dataset(tok, instruct_sft.VAL_FILE, limit=20)
    assert X.dtype == Y.dtype == torch.int16 and X.shape == Y.shape == (20, sft.BLOCK)
    lists = [sft.make_example(tok, p, story) for _, _, p, story in instruct_sft.conversations(instruct_sft.VAL_FILE)]
    lists = [ex for ex in lists if ex][:20]
    assert torch.equal(X.long(), torch.tensor([x for x, _ in lists]))
    assert torch.equal(Y.long(), torch.tensor([y for _, y in lists]))       # -100 survives int16

def test_fine_tune_takes_int16():
    import sft
    model = gpt.GPT(**dict(TINY, block=sft.BLOCK, vocab=4096))
    torch.manual_seed(2)
    X = torch.randint(0, 4096, (8, sft.BLOCK))
    Y = X.roll(-1, dims=1)
    Y[:, :100] = sft.IGNORE
    small = (X.to(torch.int16), Y.to(torch.int16))
    assert sft.answer_loss(model, *small, "cpu") == pytest.approx(sft.answer_loss(model, X, Y, "cpu"))
    sft.fine_tune(model, *small, "cpu", STEPS=3, B=4)

def test_instruct_examples_read_line_by_line(tmp_path):
    """examples() once read the whole file into memory: ~6 GB for the full file."""
    ex = "Features: Dialogue\nWords: cat, hat, mat\nSummary: A cat.\nStory: \n\nThe cat sat.\n\nThe end.\n"
    path = tmp_path / "instruct.txt"
    path.write_text(ex + "<|endoftext|>\n" + ex.replace("cat,", "dog,") + "<|endoftext|>" + "Words: cut off, in the mid")
    got = list(instruct_sft.examples(path))
    assert len(got) == 2                                   # the cut-off one is dropped
    assert got[0] == {"Features": "Dialogue", "Words": "cat, hat, mat", "Summary": "A cat.",
                      "Story": "The cat sat.\n\nThe end."}
    assert got[1]["Words"] == "dog, hat, mat"

# ---------- scaled init (GPT-2) ----------
DEEP = dict(block=16, emb=64, heads=4, layers=8, dropout=0.0, vocab=50)   # big enough to measure a std

def residual_writers(model):
    """The layers that add to the residual stream: attention's proj, and the feed-forward's second Linear."""
    return [layer for b in model.blocks for layer in (b.attn.proj, b.ffwd.net[2])]

def test_scaled_init_shrinks_the_residual_writers():
    """std 0.02 / sqrt(2 x layers): 24 writes into the stream in a 12-layer model."""
    torch.manual_seed(0)
    model = gpt.GPT(**DEEP, scaled_init=True)
    want = 0.02 / math.sqrt(2 * DEEP["layers"])                       # 0.005
    for layer in residual_writers(model):
        assert layer.weight.std().item() == pytest.approx(want, rel=0.1)
        assert layer.weight.mean().abs().item() < want / 5

def test_scaled_init_changes_nothing_else():
    """The re-init comes after everything is built, so with the same seed every other
       weight is the same as without it."""
    torch.manual_seed(0)
    plain = gpt.GPT(**DEEP)
    torch.manual_seed(0)
    scaled = gpt.GPT(**DEEP, scaled_init=True)
    changed = {id(l.weight) for l in residual_writers(scaled)} | {id(l.bias) for l in residual_writers(scaled)}
    for (name, p), q in zip(scaled.named_parameters(), plain.parameters()):
        if id(p) not in changed:
            assert torch.equal(p, q), name
    assert not torch.equal(residual_writers(scaled)[0].weight, residual_writers(plain)[0].weight)

def test_scaled_init_is_off_by_default():
    """Old checkpoints have no scaled_init in their config, so off must be the default."""
    assert gpt.GPT(**DEEP).config["scaled_init"] is False
    torch.manual_seed(0)
    plain = gpt.GPT(**DEEP)
    for layer in residual_writers(plain):                             # PyTorch's own init: much wider
        assert layer.weight.std().item() > 0.02

# ---------- gradient accumulation ----------
def fixed_batches(monkeypatch, X, Y):
    """Make gpt.batch hand out the rows of X and Y in order, B at a time."""
    rows = iter(range(0, len(X), 1))
    def batch(ids, B, block, device):
        take = [next(rows) for _ in range(B)]
        return X[take], Y[take]
    monkeypatch.setattr(gpt, "batch", batch)

def one_step(monkeypatch, X, Y, B, accum):
    fixed_batches(monkeypatch, X, Y)
    model = tiny()
    gpt.train_model(model, "cpu", STEPS=1, B=B, WARMUP=1, log_every=0, train_ids=X, val_ids=X, accum=accum)
    return model

def test_accumulation_matches_one_big_batch(monkeypatch):
    """2 micro-batches of 2 must step like 1 batch of 4. Without dividing each loss
       by accum, the step would be twice too big."""
    X = tokens(B=4)
    Y = X.roll(-1, dims=1)
    big = one_step(monkeypatch, X, Y, B=4, accum=1)
    small = one_step(monkeypatch, X, Y, B=2, accum=2)
    for (name, p), q in zip(small.named_parameters(), big.parameters()):
        assert torch.allclose(p, q, atol=1e-6), name

def test_accumulation_logs_the_mean_loss(monkeypatch, capsys):
    X = tokens(B=4)
    Y = X.roll(-1, dims=1)
    with torch.no_grad():
        mean = gpt.lm_loss(tiny(), X, Y).item()             # the loss before the step, on all 4 rows
    fixed_batches(monkeypatch, X, Y)
    monkeypatch.setattr(gpt, "val_loss", lambda *a, **k: 0.0)
    gpt.train_model(tiny(), "cpu", STEPS=1, B=2, WARMUP=1, log_every=1, train_ids=X, val_ids=X, accum=2)
    logged = float(capsys.readouterr().out.split("train loss")[1].split()[0])
    assert logged == pytest.approx(mean, abs=1e-4)

def test_resume_refuses_another_accum(tmp_path):
    ids = torch.randint(0, TINY["vocab"], (500,))
    resume = tmp_path / "tiny.resume.pt"
    gpt.train_model(tiny(), "cpu", STEPS=4, B=2, WARMUP=1, log_every=0, train_ids=ids, val_ids=ids,
                    resume=resume, resume_minutes=0, accum=2)
    with pytest.raises(ValueError):
        gpt.train_model(tiny(), "cpu", STEPS=4, B=2, WARMUP=1, log_every=0, train_ids=ids, val_ids=ids,
                        resume=resume, resume_minutes=0, accum=1)
