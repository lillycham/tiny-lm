"""A sparse autoencoder (SAE) on the story GPT's residual stream.

The residual stream after a block is a C-dimensional vector per token, and each
direction in it can mean many things at once. An SAE rewrites that vector as a
sum of a few directions out of many more (M = C x expansion), learned so that
each one, hopefully, means one thing:

    f  = TopK(ReLU(W_enc (x - b_dec) + b_enc))     M numbers, at most k of them not 0
    x̂ = W_dec f + b_dec                            back to C numbers

Column i of W_dec is feature i's direction. TopK keeps the k largest numbers of
each token and sets the rest to 0, so the sparsity is fixed at k: there is no L1
weight to tune, and the loss is only how well x̂ matches x.

    python -m tools.sae train --layer 3                       # the 384-wide story model, after block 3
    python -m tools.sae train --layer 3 --steps 200           # a quick try
    python -m tools.sae show checkpoints/sae/stories_gpt_384w6l_full_L3_x16_k32.pt
    python -m tools.sae show checkpoints/sae/stories_gpt_384w6l_full_L3_x16_k32.pt --features 12,345
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import gpt
from core.word_bpe import TOKENISER, WordBPE
from stories.stories_gpt import load_tokens

MODEL = Path("checkpoints/stories/stories_gpt_384w6l_full.pt")
OUT = Path("checkpoints/sae")

# ---------- the SAE ----------
class SAE(nn.Module):
    """x (..., C) -> (x̂ (..., C), f (..., M)). See the module docstring.

    scale is a fixed number, set once from the data before training (set_scale): it
    makes the average |x| equal sqrt(C), so the learning rate does not depend on how
    large one layer's vectors happen to be. encode() and decode() apply it, so the
    caller always works with the model's own vectors."""

    def __init__(self, C, M, k):
        super().__init__()
        self.config = dict(C=C, M=M, k=k)
        self.k = k
        self.enc = nn.Linear(C, M)
        self.dec = nn.Linear(M, C)               # dec.weight is (C, M): one column per feature
        self.register_buffer("scale", torch.tensor(1.0))
        with torch.no_grad():
            # Start the decoder with random unit columns, and the encoder as its
            # transpose: then every feature starts out reading the same direction it
            # writes, which leaves fewer features that never fire ("dead").
            self.dec.weight.normal_()
            self.normalise_decoder()
            self.enc.weight.copy_(self.dec.weight.T)
            self.enc.bias.zero_()
            self.dec.bias.zero_()

    def encode(self, x):
        pre = F.relu(self.enc(x * self.scale - self.dec.bias))
        values, indices = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, values)

    def decode(self, f):
        return self.dec(f) / self.scale

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def normalise_decoder(self):
        """Make every decoder column length 1. Without this the SAE could make f smaller
        and the columns longer, and the sizes of the features would mean nothing."""
        self.dec.weight /= self.dec.weight.norm(dim=0, keepdim=True)

    @torch.no_grad()
    def set_scale(self, x):
        self.scale.fill_(math.sqrt(x.shape[-1]) / x.norm(dim=-1).mean().item())

def save(sae, path, **info):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": sae.config, "weights": sae.state_dict(), "info": info}, path)

def load(path, device="cpu"):
    saved = torch.load(path, map_location=device)
    sae = SAE(**saved["config"]).to(device)
    sae.load_state_dict(saved["weights"])
    return sae, saved["info"]

# ---------- activations ----------
def chunks(tokens, B, T, rng):
    """B random windows of T tokens from a 1-D array of token IDs, as a (B, T) tensor."""
    starts = rng.integers(0, len(tokens) - T, size=B)
    return torch.from_numpy(np.stack([np.asarray(tokens[s:s + T], dtype=np.int64) for s in starts]))

@torch.no_grad()
def residuals(model, layer, X):
    """The residual stream after block `layer` (0 = the first block) for X (B, T):
    (B, T, C). The model runs only as far as it needs to."""
    x = model.tok(X) + model.pos(torch.arange(X.shape[1], device=X.device))
    for block in model.blocks[:layer + 1]:
        x = block(x)
    return x

def stream(model, layer, tokens, B, T, device, seed=0):
    """An endless stream of (B * (T - 1), C) batches of residual vectors, shuffled.

    Position 0 is left out: every window's first token sees only itself, and its
    vector is much larger than the rest, so it would pull the SAE towards it."""
    rng = np.random.default_rng(seed)
    while True:
        X = chunks(tokens, B, T, rng).to(device)
        x = residuals(model, layer, X)[:, 1:].reshape(-1, model.config["emb"])
        yield x[torch.randperm(len(x), device=device)]

# ---------- evaluation ----------
@torch.no_grad()
def variance_explained(sae, x):
    """1 - (squared error) / (variance of x): 1 is perfect, 0 is no better than the mean vector."""
    x_hat, _ = sae(x)
    return 1 - ((x - x_hat) ** 2).sum().item() / ((x - x.mean(0)) ** 2).sum().item()

@torch.no_grad()
def spliced_loss(model, layer, X, Y, replace):
    """The model's loss on X -> Y, with block `layer`'s output x replaced by replace(x).

    A forward hook that returns a value replaces the module's output, so everything
    after the block sees the replacement."""
    handle = model.blocks[layer].register_forward_hook(lambda module, inputs, output: replace(output))
    try:
        return gpt.lm_loss(model, X, Y).item()
    finally:
        handle.remove()

@torch.no_grad()
def evaluate(model, sae, layer, tokens, device, B=16, T=256, seed=1):
    """The numbers to judge an SAE by, on held-out text.

      variance  how much of the residual stream x̂ keeps
      clean     the model's own loss
      sae       the loss with x̂ in place of x after the block
      mean      the loss with the mean vector in place of x (all the layer's information gone)
      recovered (mean - sae) / (mean - clean): the share of the loss that the SAE keeps
      dead      the share of features that did not fire on any of these tokens"""
    rng = np.random.default_rng(seed)
    XY = chunks(tokens, B, T + 1, rng).to(device)
    X, Y = XY[:, :-1], XY[:, 1:]
    x = residuals(model, layer, X)
    flat = x[:, 1:].reshape(-1, x.shape[-1])
    mean = flat.mean(0)
    _, f = sae(flat)
    clean = gpt.lm_loss(model, X, Y).item()
    with_sae = spliced_loss(model, layer, X, Y, lambda out: torch.cat([out[:, :1], sae(out[:, 1:])[0]], 1))
    with_mean = spliced_loss(model, layer, X, Y, lambda out: torch.cat([out[:, :1], mean.expand_as(out[:, 1:])], 1))
    return dict(variance=variance_explained(sae, flat), clean=clean, sae=with_sae, mean=with_mean,
                recovered=(with_mean - with_sae) / (with_mean - clean),
                dead=(f.amax(0) == 0).float().mean().item())

def format_eval(e):
    return (f"variance explained {e['variance']:.3f} | loss {e['clean']:.3f} clean, {e['sae']:.3f} with the SAE, "
            f"{e['mean']:.3f} with the mean | {e['recovered']:.1%} recovered | {e['dead']:.1%} dead")

# ---------- training ----------
def train(model, layer, expansion, k, steps, lr, B, device, log_every=250):
    C = model.config["emb"]
    sae = SAE(C, C * expansion, k).to(device)
    acts = stream(model, layer, load_tokens("train"), B, model.block, device)
    first = next(acts)
    sae.set_scale(first)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    # The learning rate stays at lr, then falls to 0 over the last fifth of the steps.
    decay_from = int(steps * 0.8)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 1.0 if s < decay_from else (steps - s) / (steps - decay_from))
    batch = 4096
    buffer, fired = first, torch.zeros(sae.config["M"], dtype=torch.bool, device=device)
    start = time.time()
    for step in range(steps):
        if len(buffer) < batch:
            buffer = torch.cat([buffer, next(acts)])
        x, buffer = buffer[:batch], buffer[batch:]
        x_hat, f = sae(x)
        # The loss is the error in the SAE's own units (x * scale), so its size does not depend on the layer.
        loss = ((x - x_hat) * sae.scale).pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        schedule.step()
        sae.normalise_decoder()
        fired |= f.amax(0) > 0
        if (step + 1) % log_every == 0 or step == steps - 1:
            print(f"step {step + 1:>6}: loss {loss.item():.4f}, {(~fired).float().mean().item():.1%} of features "
                  f"not fired since the last log ({time.time() - start:.0f} s)", flush=True)
            fired.zero_()
    return sae

# ---------- looking at features ----------
@torch.no_grad()
def top_contexts(model, sae, layer, tokens, device, n_tokens=500_000, top=8, B=32, seed=2):
    """For every feature, the `top` tokens it fires on most over about n_tokens of text:
    (values (top, M), windows (top, M), positions (top, M)), plus the windows themselves (n, T)."""
    rng = np.random.default_rng(seed)
    T, M = model.block, sae.config["M"]
    best = torch.full((top, M), -1.0, device=device)
    where = torch.zeros((top, M), dtype=torch.long, device=device)
    windows = []
    for _ in range(max(1, n_tokens // (B * T))):
        X = chunks(tokens, B, T, rng).to(device)
        _, f = sae(residuals(model, layer, X))           # (B, T, M)
        f[:, 0] = -1.0                                   # (position 0: see stream())
        flat = f.reshape(-1, M)
        # Each token's place in all the windows so far: window number * T + position.
        ids = torch.arange(len(flat), device=device) + sum(len(w) for w in windows) * T
        values, rows = torch.cat([best, flat]).topk(top, dim=0)
        where = torch.cat([where, ids.unsqueeze(1).expand(-1, M)]).gather(0, rows)
        best = values
        windows.append(X.cpu())
    return best.cpu(), where.cpu(), torch.cat(windows)

def show(path, features, device, n_tokens, top=8, before=24):
    sae, info = load(path, device)
    model = gpt.load(device, Path(info["model"]))
    tok = WordBPE.load(TOKENISER)
    layer = info["layer"]
    values, where, windows = top_contexts(model, sae, layer, load_tokens("val"), device, n_tokens, top)
    T = model.block
    if not features:
        # By default, 20 features spread over how often they fire: the strongest first.
        alive = (values[0] > 0).nonzero().flatten()
        order = alive[values[0, alive].argsort(descending=True)]
        features = order[torch.linspace(0, len(order) - 1, min(20, len(order))).long()].tolist()
    print(f"{path}: layer {layer}, {sae.config['M']} features, k = {sae.config['k']}; "
          f"{(values[0] <= 0).float().mean().item():.1%} never fired on {n_tokens:,} tokens\n")
    for i in features:
        print(f"feature {i}:")
        for v, flat in zip(values[:, i].tolist(), where[:, i].tolist()):
            if v <= 0:
                break
            w, t = divmod(flat, T)
            ids = windows[w].tolist()
            left = tok.decode(ids[max(0, t - before):t]).replace("\n", " ")
            here = tok.decode(ids[t:t + 1]).replace("\n", "⏎")
            right = tok.decode(ids[t + 1:t + 4]).replace("\n", " ")
            print(f"  {v:6.2f}  …{left}[{here}]{right}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train an SAE on one layer, then evaluate it")
    gpt.add_device_option(t)
    t.add_argument("--checkpoint", type=Path, default=MODEL, help=f"the model (default {MODEL})")
    t.add_argument("--layer", type=int, default=3, help="after which block, from 0 (default 3)")
    t.add_argument("--expansion", type=int, default=16, help="features per residual dimension (default 16)")
    t.add_argument("--k", type=int, default=32, help="features on per token (default 32)")
    t.add_argument("--steps", type=int, default=5000, help="batches of 4096 vectors (default 5000)")
    t.add_argument("--lr", type=float, default=3e-4, help="default 3e-4")
    t.add_argument("--B", type=int, default=32, help="windows per model run (default 32)")
    s = sub.add_parser("show", help="print the texts that features fire on most")
    gpt.add_device_option(s)
    s.add_argument("sae", type=Path)
    s.add_argument("--features", default="", help="comma-separated feature numbers (default: 20 spread out)")
    s.add_argument("--tokens", type=int, default=500_000, help="how much validation text to search (default 500,000)")
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.cmd == "train":
        model = gpt.load(args.device, args.checkpoint)
        for p in model.parameters():
            p.requires_grad_(False)
        sae = train(model, args.layer, args.expansion, args.k, args.steps, args.lr, args.B, args.device)
        e = evaluate(model, sae, args.layer, load_tokens("val"), args.device)
        print(format_eval(e))
        path = OUT / f"{args.checkpoint.stem}_L{args.layer}_x{args.expansion}_k{args.k}.pt"
        save(sae, path, model=str(args.checkpoint), layer=args.layer, steps=args.steps, eval=e)
        print(f"saved {path}")
    else:
        features = [int(x) for x in args.features.split(",") if x]
        show(args.sae, features, args.device, args.tokens)
