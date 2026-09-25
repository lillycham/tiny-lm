"""A small GPT: the transformer from transformer.py, made bigger and faster.

What changes from transformer.py:
  - All heads in one matrix multiply, not a Python loop over separate heads.
  - A longer context (64 characters) and a bigger model: 2.7 million parameters.
  - Dropout, against overfitting.
  - Training on the GPU (MPS), with a saved checkpoint, so main.py can load the
    trained model without training it again.

    python gpt.py                  # checks, then train on the GPU and save checkpoints/gpt.pt
    python gpt.py --block 128 --layers 4
    python gpt.py --help           # all the sizes you can change
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import attention
from data import V, decode, encode, train, val

# The default sizes. A checkpoint saves the sizes it was trained with.
CONFIG = dict(
    block=64,       # characters of context
    emb=192,        # size of each position's vector
    heads=6,        # heads per block, so each head has size 192 / 6 = 32
    layers=6,       # blocks
    dropout=0.2,    # fraction of values set to zero during training
)

CHECKPOINT = Path("checkpoints/gpt.pt")

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

    def __init__(self, C, n_head, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        # One layer for q, k and v of every head: C in, 3 * C out.
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C)
        self.drop = nn.Dropout(dropout)

    def split_heads(self, t):
        """(B, T, C) -> (B, n_head, T, hs), where hs = C // n_head."""
        B, T, C = t.shape
        # TODO(Lilly): two steps.
        #   1. t.view(B, T, n_head, hs) cuts each position's C numbers into n_head
        #      pieces of hs numbers: head 0 gets the first hs, head 1 the next hs, ...
        #   2. .transpose(1, 2) swaps the T and n_head axes, so each head has its own
        #      (T, hs) block, as attention() needs.
        ...

    def join_heads(self, t):
        """(B, n_head, T, hs) -> (B, T, C). The opposite of split_heads."""
        B, n_head, T, hs = t.shape
        # TODO(Lilly): undo split_heads, in the opposite order.
        #   1. .transpose(1, 2) back to (B, T, n_head, hs).
        #   2. .reshape(B, T, n_head * hs) puts each position's heads side by side,
        #      like torch.cat(head_outs, dim=-1) did.
        #   (reshape, not view: after a transpose the numbers aren't in order in memory,
        #   and view only works when they are. reshape makes a copy when it must.)
        ...

    def forward(self, x):
        """x (B, T, C) -> (B, T, C)."""
        C = x.shape[-1]
        # TODO(Lilly): four steps.
        #   1. self.qkv(x) gives (B, T, 3 * C). .split(C, dim=-1) cuts it into three
        #      (B, T, C) pieces, in the order q, k, v.
        #   2. split_heads on each of q, k and v.
        #   3. attention(q, k, v), which returns (out, weights).
        #   4. join_heads on out, then self.proj, then self.drop.
        ...

class FeedForward(nn.Module):
    """As in transformer.py, with dropout at the end."""

    def __init__(self, C, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C, 4 * C), nn.ReLU(), nn.Linear(4 * C, C), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """As in transformer.py, with the faster attention."""

    def __init__(self, C, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = CausalSelfAttention(C, n_head, dropout)
        self.ln2 = nn.LayerNorm(C)
        self.ffwd = FeedForward(C, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    """Embeddings -> blocks -> LayerNorm -> logits."""

    def __init__(self, block, emb, heads, layers, dropout):
        super().__init__()
        self.config = dict(block=block, emb=emb, heads=heads, layers=layers, dropout=dropout)
        self.block = block
        self.tok = nn.Embedding(V, emb)
        self.pos = nn.Embedding(block, emb)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(emb, heads, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(emb)
        self.out = nn.Linear(emb, V)

    def forward(self, X):
        """X (B, T) IDs -> logits (B, T, V)."""
        x = self.tok(X) + self.pos(torch.arange(X.shape[1], device=X.device))
        return self.out(self.ln(self.blocks(self.drop(x))))

# ---------- data ----------
def batch(ids, B, block, device):
    """B random chunks of `block` characters, and the same chunks shifted by one."""
    starts = torch.randint(0, len(ids) - block, (B, 1))
    idx = starts + torch.arange(block)     # (B, block): the positions of every chunk
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
def train_model(model, device, STEPS=5000, B=64, LR=1e-3, WARMUP=100, log_every=500):
    """Train with AdamW. The learning rate warms up, then falls along a cosine curve."""
    train_ids, val_ids = torch.tensor(train), torch.tensor(val)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    def lr_at(step):
        """Warm up for WARMUP steps, then follow a cosine curve down to LR / 10."""
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        progress = (step - WARMUP) / (STEPS - WARMUP)
        return LR / 10 + (LR - LR / 10) * 0.5 * (1 + math.cos(math.pi * progress))

    model.train()
    start = time.time()
    for step in range(STEPS):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        loss = lm_loss(model, *batch(train_ids, B, model.block, device))
        opt.zero_grad()
        loss.backward()
        # Scale down very big gradients, so one unusual batch can't wreck the weights.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:5d}  train loss {loss.item():.4f}"
                  f"  val loss {val_loss(model, val_ids, device):.4f}  ({time.time() - start:.0f}s)", flush=True)

# ---------- generate text ----------
@torch.no_grad()
def generate(model, n, start="\n", temperature=1.0):
    """Sample n characters after `start`. Temperature below 1 makes safer choices."""
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([encode(start).tolist()], device=device)
    out = []
    for _ in range(n):
        logits = model(ids[:, -model.block:])[0, -1] / temperature
        char_next = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, char_next[None]], dim=1)
        out.append(char_next.item())
    return decode(out)

def save(model):
    """Save the weights and the sizes, so load() can rebuild the same model."""
    CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save({"config": model.config, "weights": model.state_dict()}, CHECKPOINT)

def load(device="cpu"):
    saved = torch.load(CHECKPOINT, map_location=device)
    model = GPT(**saved["config"]).to(device)
    model.load_state_dict(saved["weights"])
    model.eval()
    return model

# ---------- checks ----------
def check_against_loop():
    """Copy the weights of transformer.py's MultiHeadAttention and compare the outputs."""
    from transformer import MultiHeadAttention

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
    if device == "mps":
        torch.mps.synchronize()   # wait until the GPU has finished
    return (time.time() - t) / 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"], help="default mps")
    parser.add_argument("--steps", type=int, default=5000, help="training steps (default 5000)")
    for name, value in CONFIG.items():
        parser.add_argument(f"--{name}", type=type(value), default=value, help=f"default {value}")
    args = parser.parse_args()
    config = {name: getattr(args, name) for name in CONFIG}
    if config["emb"] % config["heads"]:
        parser.error("--emb must be a multiple of --heads, so every head gets the same size")

    check_against_loop()
    print(f"\n2. Time for one training step: CPU {check_speed(config, 'cpu') * 1000:.0f} ms,"
          f" MPS {check_speed(config, 'mps') * 1000:.0f} ms")

    torch.manual_seed(0)
    model = GPT(**config).to(args.device)
    print(f"\n{sum(p.numel() for p in model.parameters()):,} parameters on {args.device},"
          f" starting val loss {val_loss(model, torch.tensor(val), args.device):.4f}")
    train_model(model, args.device, STEPS=args.steps)

    save(model)
    print(f"\nSaved to {CHECKPOINT}\n")
    print("ROMEO:" + generate(model, 500, start="ROMEO:"))
