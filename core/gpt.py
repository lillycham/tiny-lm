"""A small GPT: the transformer from transformer.py, made bigger and faster.

What changes from transformer.py:
  - All heads in one matrix multiply, not a Python loop over separate heads.
  - A longer context (64 characters) and a bigger model: 2.7 million parameters.
  - Dropout, against overfitting.
  - Training on the GPU (MPS), with a saved checkpoint, so main.py can load the
    trained model without training it again.

    python -m core.gpt                  # checks, then train on the GPU and save checkpoints/shakespeare/gpt.pt
    python -m core.gpt --block 128 --layers 4
    python -m core.gpt --help           # all the sizes you can change
"""
import argparse
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attention import attention
from core.data import V, decode, encode, train, val

# The default sizes. A checkpoint saves the sizes it was trained with.
CONFIG = dict(
    block=64,       # characters of context
    emb=192,        # size of each position's vector
    heads=6,        # heads per block, so each head has size 192 / 6 = 32
    layers=6,       # blocks
    dropout=0.2,    # fraction of values set to zero during training
)

CHECKPOINT = Path("checkpoints/shakespeare/gpt.pt")

# ---------- the model ----------
class CausalSelfAttention(nn.Module):
    """All the heads of MultiHeadAttention, computed together.

    transformer.py ran n_head small heads one after another. Here one Linear makes
    q, k and v for every head at once, and the heads become an extra batch axis:

        x (B, T, C) -> q, k, v (B, T, C) -> split into heads (B, n_head, T, hs)
          -> attention() on all heads at once -> join the heads (B, T, C) -> proj

    Your attention() already works on (B, n_head, T, hs): the @ and the mask only
    use the last two axes, so every axis in front is treated as a batch axis.
    """

    def __init__(self, C, n_head, dropout=0.0, smear=False):
        super().__init__()
        self.n_head = n_head
        # One layer for q, k and v of every head: C in, 3 * C out.
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C)
        self.drop = nn.Dropout(dropout)
        self.smear = smear
        if smear:
            self.a = nn.Parameter(torch.full((n_head,), 0.0))

    def split_heads(self, t):
        """(B, T, C) -> (B, n_head, T, hs), where hs = C // n_head."""
        B, T, C = t.shape

        return t.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

    def join_heads(self, t):
        """(B, n_head, T, hs) -> (B, T, C). The opposite of split_heads."""
        B, n_head, T, hs = t.shape

        return t.transpose(1,2).reshape(B, T, n_head * hs)
        
    def forward(self, x):
        """x (B, T, C) -> (B, T, C)."""
        C = x.shape[-1]

        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        k = self.smear_keys(k)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        return self.drop(self.proj(self.join_heads(out)))

    def smear_keys(self, k):
        # here: k is (B, n_head, T, hs), so smear it along T (dim=2)
        if self.smear:
            zeros = torch.zeros_like(k[:, :, :1])
            k_prev = torch.cat([zeros, k[:, :, :-1]], dim=2)

            # k'ₜ = σ(a)·kₜ + (1 − σ(a))·kₜ₋₁        one learned number a per head
            s = torch.sigmoid(self.a).view(-1, 1, 1)
            k = s * k  + (1 - s) * k_prev
        return k

class FeedForward(nn.Module):
    """As in transformer.py, with dropout at the end, and GELU in place of ReLU if gelu."""

    def __init__(self, C, dropout, gelu=False):
        super().__init__()
        act = nn.GELU(approximate="tanh") if gelu else nn.ReLU()
        self.net = nn.Sequential(nn.Linear(C, 4 * C), act, nn.Linear(4 * C, C), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """As in transformer.py, with the faster attention."""

    def __init__(self, C, n_head, dropout, gelu=False, smear=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = CausalSelfAttention(C, n_head, dropout, smear=smear)
        self.ln2 = nn.LayerNorm(C)
        self.ffwd = FeedForward(C, dropout, gelu)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    """Embeddings -> blocks -> LayerNorm -> logits.

    gelu and tied make the model like GPT-2, so it can become a Hugging Face GPT-2 model:
      - gelu: GELU in place of ReLU in every feed-forward layer.
      - tied: the output layer uses the token embedding matrix, and has no bias.
    smearing: smearing is not gpt-2 compatible, but can be enabled with `smear=True
    for improved capabilities.
    """

    def __init__(self, block, emb, heads, layers, dropout, vocab=V, gelu=False,
                  tied=False, smear=False, scaled_init=False):
        super().__init__()
        self.config = dict(block=block, emb=emb, heads=heads, layers=layers, dropout=dropout, vocab=vocab,
                           gelu=gelu, tied=tied, smear=smear, scaled_init=scaled_init)
        self.block = block
        self.tok = nn.Embedding(vocab, emb)
        self.pos = nn.Embedding(block, emb)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(emb, heads, dropout, gelu, smear) for _ in range(layers)])
        self.ln = nn.LayerNorm(emb)

        if tied:
            self.out = nn.Linear(emb, vocab, bias=False)
            nn.init.normal_(self.tok.weight, std=0.02)
            self.out.weight = self.tok.weight
        else:
            self.out = nn.Linear(emb, vocab)

        if scaled_init:
            for b in self.blocks:
                for layer in (b.attn.proj, b.ffwd.net[2]):
                    nn.init.normal_(layer.weight, std=0.02 / math.sqrt(2 * layers))
                    nn.init.zeros_(layer.bias)

    def forward(self, X, last_only=False):
        """X (B, T) IDs -> logits (B, T, V). With last_only, only the last position's,
        (B, 1, V): all that generation needs, and T times less work and memory for the
        output layer (with 16,384 tokens and T = 1,024, 67 MB of logits instead of 64 KB)."""
        x = self.tok(X) + self.pos(torch.arange(X.shape[1], device=X.device))
        x = self.blocks(self.drop(x))
        return self.out(self.ln(x[:, -1:] if last_only else x))

# ---------- data ----------
DEVICES = ["cpu", "mps", "cuda"]

def default_device():
    """The fastest device here: an NVIDIA GPU, then an Apple GPU, then the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def add_device_option(parser):
    """The same --device option for every script."""
    parser.add_argument("--device", default=default_device(), choices=DEVICES,
                        help=f"default {default_device()}, the fastest here")

def synchronize(device):
    """Wait until the GPU has finished. GPU work runs in the background, so a timer
       stopped before this measures only how long it took to queue the work."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

def batch(ids, B, block, device):
    """B random chunks of `block` characters, and the same chunks shifted by one."""
    starts = torch.randint(0, len(ids) - block, (B, 1))
    idx = starts + torch.arange(block)     # (B, block): the positions of every chunk
    if isinstance(ids, np.ndarray):
        # NumPy tokens, maybe read from disk only as needed: take the chunks, then make
        # them tensors. uint16 -> int64, because nn.Embedding needs int64 IDs.
        idx = idx.numpy()
        return (torch.from_numpy(ids[idx].astype(np.int64)).to(device),
                torch.from_numpy(ids[idx + 1].astype(np.int64)).to(device))
    return ids[idx].to(device), ids[idx + 1].to(device)

def lm_loss(model, X, Y):
    return F.cross_entropy(model(X).flatten(0, 1), Y.flatten())

@torch.no_grad()
def val_loss(model, ids, device, batches=40, B=64):
    """Mean loss on the same 40 random batches of ids every time."""
    # eval() turns dropout off: the model uses all its values when it is judged.
    model.eval()
    g = torch.random.get_rng_state()
    torch.manual_seed(1234)
    loss = sum(lm_loss(model, *batch(ids, B, model.block, device)).item() for _ in range(batches)) / batches
    torch.random.set_rng_state(g)
    model.train()
    return loss

# ---------- training ----------
def param_groups(model, weight_decay):
    """AdamW parameter groups: weight decay for the matrices, none for the rest.

    Weight decay pulls every weight a little toward 0 at each step, so that no
    weight grows big unless the loss needs it. That makes sense for the matrices:
    the Linear weights and the embeddings. It doesn't for the rest:
      - A LayerNorm weight is a scale that starts at 1. Pulling it to 0 fights the
        normalisation, and there is only one per feature, so it can't overfit.
      - A bias only shifts a value, and there are few of them.
    The matrices are exactly the parameters with 2 or more dimensions (p.dim() >= 2).
    The biases and LayerNorm weights have 1.
    """
    params = list(model.parameters())
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]

    return [{"params": decay, "weight_decay": weight_decay}, 
            {"params": no_decay, "weight_decay": 0.0}]

def run_info():
    """What made a checkpoint: the code version, the command and the date.

    dirty is True if tracked files had changes that weren't committed. Then diff holds
    those changes, so the exact code can still be rebuilt from the commit.
    """
    def git(*args):
        r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent)
        return r.stdout.strip() if r.returncode == 0 else None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status), "diff": git("diff", "HEAD") if status else "",
            "argv": sys.argv, "date": datetime.now().isoformat(timespec="seconds")}

def is_snapshot_step(n):
    """Steps 64, 128, 256, ...: close together early, when heads form, and far apart later."""
    return n >= 64 and n & (n - 1) == 0

def train_model(model, device, STEPS=5000, B=64, LR=1e-3, WARMUP=100, log_every=500,
                train_ids=None, val_ids=None, resume=None, resume_minutes=30,
                snapshots=None, compile=False, val_B=64, accum=1):
    """Train with AdamW. The learning rate warms up, then falls along a cosine curve.

    Trains on the character IDs from data.py, or on train_ids and val_ids if given.

    With resume, a path: save everything needed to carry on there every resume_minutes,
    and if the file already exists, carry on from it. Everything is the weights, the
    optimizer's state (Adam's m and v for every weight), the step, and the random
    number generator, so the batches after a restart are the ones there would have been.

    With snapshots, a path like checkpoints/stories/snapshots/stories_gpt: also save the model
    at steps 64, 128, 256, ... as stories_gpt_step64.pt and so on, to see how it changes.

    val_B is the batch size for the val loss. Lower it for long contexts and big
    vocabularies: the logits of one batch are B x block x vocab floats.

    With accum, each step is accum micro-batches of B sequences: their gradients add
    up, then one optimizer step. Like a batch of accum x B, in the memory of B.
    """
    step_model = torch.compile(model) if compile else model
    if train_ids is None:
        train_ids, val_ids = torch.tensor(train), torch.tensor(val)
    opt = torch.optim.AdamW(param_groups(model, 0.1), lr=LR)
    first = 0
    if resume and resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False)
        if (saved["STEPS"], saved["LR"], saved["config"], saved.get("accum", 1)) != (STEPS, LR, model.config, accum):
            raise ValueError(f"{resume} is from a run with other settings. Delete it to start again.")
        model.load_state_dict(saved["weights"])
        opt.load_state_dict(saved["optimizer"])
        torch.random.set_rng_state(saved["rng"].cpu())
        first = saved["step"]
        print(f"Carrying on from step {first} of {STEPS}, saved in {resume}", flush=True)

    def save_resume(step):
        """Write to a temporary file, then rename: a crash while saving can't break the old file."""
        tmp = resume.with_suffix(".tmp")
        torch.save({"weights": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "rng": torch.random.get_rng_state(), "STEPS": STEPS, "LR": LR, "config": model.config, "accum": accum,
                    "info": run_info()}, tmp)
        tmp.replace(resume)

    def lr_at(step):
        """Warm up for WARMUP steps, then follow a cosine curve down to LR / 10."""
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        progress = (step - WARMUP) / (STEPS - WARMUP)
        return LR / 10 + (LR - LR / 10) * 0.5 * (1 + math.cos(math.pi * progress))

    model.train()
    start = last_save = time.time()
    for step in range(first, STEPS):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        opt.zero_grad()
        total = 0
        for _ in range(accum):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                loss = lm_loss(step_model, *batch(train_ids, B, model.block, device)) / accum
                total += loss.detach()
            loss.backward()

        # Scale down very big gradients, so one unusual batch can't wreck the weights.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:5d}  train loss {float(total):.4f}"
                  f"  val loss {val_loss(model, val_ids, device, B=val_B):.4f}  ({time.time() - start:.0f}s)", flush=True)
        if snapshots and is_snapshot_step(step + 1):
            save(model, snapshots.with_name(f"{snapshots.name}_step{step + 1}.pt"))
        if resume and time.time() - last_save > resume_minutes * 60:
            save_resume(step + 1)
            last_save = time.time()

# ---------- generate text ----------
@torch.no_grad()
def generate(model, n, start="\n", temperature=1.0, encode=encode, decode=decode, stop=None):
    """Sample n tokens after `start`. Temperature below 1 makes safer choices.

    The tokens are characters, unless you give another tokeniser's encode and decode.
    If the model samples the token ID stop, end there, without it.
    """
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([[int(i) for i in encode(start)]], device=device)
    out = []
    for _ in range(n):
        logits = model(ids[:, -model.block:])[0, -1] / temperature
        char_next = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if char_next.item() == stop:
            break
        ids = torch.cat([ids, char_next[None]], dim=1)
        out.append(char_next.item())
    return decode(out)

def save(model, path=CHECKPOINT):
    """Save the weights and the sizes, so load() can rebuild the same model, and run_info()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "weights": model.state_dict(), "info": run_info()}, path)

def load_info(path):
    """The run_info() saved with a checkpoint, or None for checkpoints from before it."""
    return torch.load(path, map_location="cpu").get("info")

def load(device="cpu", path=CHECKPOINT):
    saved = torch.load(path, map_location=device)
    model = GPT(**saved["config"]).to(device)
    model.load_state_dict(saved["weights"])
    model.eval()
    return model

# ---------- checks ----------
def check_param_groups():
    """Count the parameters in each weight-decay group of the story model."""
    torch.manual_seed(0)
    model = GPT(block=256, emb=192, heads=6, layers=6, dropout=0.0, vocab=4096, gelu=True, tied=True)
    decay, no_decay = param_groups(model, 0.1)
    count = lambda group: sum(p.numel() for p in group["params"])
    print(f"3. Weight decay on {len(decay['params'])} tensors, {count(decay):,} numbers;"
          f" none on {len(no_decay['params'])} tensors, {count(no_decay):,} numbers")
    print("   (expected 26 tensors with 3,489,792 numbers, and 44 with 11,904: 6 blocks of 4 Linear weights,"
          " plus the 2 embeddings; the biases and LayerNorms are the rest)")

def check_against_loop():
    """Copy the weights of transformer.py's MultiHeadAttention and compare the outputs."""
    from shakespeare.transformer import MultiHeadAttention

    torch.manual_seed(0)
    C, n_head = 64, 4
    slow = MultiHeadAttention(C, n_head)
    fast = CausalSelfAttention(C, n_head).eval()   # eval(): no dropout, so the outputs can match
    with torch.no_grad():
        # Rows 0..C of qkv are q for every head, in head order; then k; then v.
        fast.qkv.weight.copy_(torch.cat(
            [torch.cat([getattr(h, name).weight for h in slow.heads]) for name in ("query", "key", "value")]))
        fast.proj.load_state_dict(slow.proj.state_dict())
    x = torch.randn(2, 10, C)
    print(f"1. Largest difference from the heads in a loop: {(fast(x) - slow(x)).abs().max():.2e}")

def check_speed(config, device):
    torch.manual_seed(0)
    model = GPT(**config).to(device)
    X, Y = batch(torch.tensor(train), 64, model.block, device)
    for _ in range(3):      # warm-up steps: the first steps on a GPU are slow
        lm_loss(model, X, Y).backward()
    t = time.time()
    for _ in range(10):
        lm_loss(model, X, Y).backward()
    synchronize(device)
    return (time.time() - t) / 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_option(parser)
    parser.add_argument("--steps", type=int, default=5000, help="training steps (default 5000)")
    for name, value in CONFIG.items():
        parser.add_argument(f"--{name}", type=type(value), default=value, help=f"default {value}")
    args = parser.parse_args()
    config = {name: getattr(args, name) for name in CONFIG}
    if config["emb"] % config["heads"]:
        parser.error("--emb must be a multiple of --heads, so every head gets the same size")

    check_against_loop()
    check_param_groups()
    gpu = f", {args.device.upper()} {check_speed(config, args.device) * 1000:.0f} ms" if args.device != "cpu" else ""
    print(f"\n2. Time for one training step: CPU {check_speed(config, 'cpu') * 1000:.0f} ms{gpu}")

    torch.manual_seed(0)
    model = GPT(**config).to(args.device)
    print(f"\n{sum(p.numel() for p in model.parameters()):,} parameters on {args.device},"
          f" starting val loss {val_loss(model, torch.tensor(val), args.device):.4f}")
    train_model(model, args.device, STEPS=args.steps)

    save(model)
    print(f"\nSaved to {CHECKPOINT}\n")
    print("ROMEO:" + generate(model, 500, start="ROMEO:"))
