"""Self-attention, the core of a transformer, on its own.

The MLP joins its 8 context characters into one vector, with separate weights for
every position. Attention works differently: every position t makes a weighted
average of the positions 0..t, and the weights come from the characters themselves.

For a batch of B sequences of T positions, each position a vector of size C:

    q = x @ Wq    (B, T, hs)   query: "what am I looking for?"
    k = x @ Wk    (B, T, hs)   key:   "what do I contain?"
    v = x @ Wv    (B, T, hs)   value: "what do I give if you pick me?"

    scores  = q @ k^T / sqrt(hs)    (B, T, T)   scores[b, t, s]: how well query t matches key s
    weights = causal softmax        (B, T, T)   row t: weights for s = 0..t, zero after t, sums to 1
    out     = weights @ v           (B, T, hs)  row t: weighted average of the values v[0..t]

    python attention.py
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import V, decode, encode, train, val

# ---------- attention ----------
def causal_weights(scores):
    """Turn scores of shape (..., T, T) into attention weights of the same shape.

    Row t may only use positions 0..t: a position must not see the characters after
    it, because those are what the model has to predict. Each row must sum to 1.
    """
    # Hide the future
    x = scores.masked_fill(
        torch.ones(scores.shape[-1], scores.shape[-1], device=scores.device).tril() == 0, -torch.inf)

    # Softmax along the last axis
    return F.softmax(x, dim=-1)

def attention(q, k, v):
    """Causal attention. q, k and v have shape (B, T, hs). Returns (out, weights)."""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    weights = causal_weights(scores)
    out = weights @ v

    return out, weights

class Head(nn.Module):
    """One attention head: project x to q, k and v, then attend."""

    def __init__(self, C, hs):
        super().__init__()
        # No bias: q, k and v are plain projections of x.
        self.query = nn.Linear(C, hs, bias=False)
        self.key = nn.Linear(C, hs, bias=False)
        self.value = nn.Linear(C, hs, bias=False)

    def forward(self, x):
        """x (B, T, C) -> (out (B, T, hs), weights (B, T, T))."""
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = attention(q, k, v)

        return attn

# ---------- a tiny model to train ----------
BLOCK, EMB = 32, 32

class AttentionLM(nn.Module):
    """Character and position embeddings -> one attention head -> logits.

    Attention has no idea of order: it treats positions 0..t as an unordered set.
    The position embedding (a learned vector for each of the BLOCK positions) is
    added to each character's embedding, so the model can tell "just before" from
    "far back".
    """

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, EMB)
        self.pos = nn.Embedding(BLOCK, EMB)
        self.head = Head(EMB, EMB)
        self.out = nn.Linear(EMB, V)

    def embed(self, X):
        """X (B, T) IDs -> character + position embeddings (B, T, EMB)."""
        return self.tok(X) + self.pos(torch.arange(X.shape[1], device=X.device))

    def forward(self, X):
        """X (B, T) IDs -> logits (B, T, V)."""
        h, _ = self.head(self.embed(X))
        return self.out(h)

    def weights(self, X):
        """X (B, T) IDs -> the head's attention weights (B, T, T)."""
        _, weights = self.head(self.embed(X))
        return weights

def batch(ids, B):
    """B random chunks of BLOCK characters, and the same chunks shifted by one.

    Unlike the MLP, every position is a training example: position t predicts
    Y[:, t] from X[:, 0..t]. One chunk of 32 characters gives 32 examples.
    """
    starts = torch.randint(0, len(ids) - BLOCK, (B,))
    X = torch.stack([ids[s:s + BLOCK] for s in starts])
    Y = torch.stack([ids[s + 1:s + BLOCK + 1] for s in starts])
    return X, Y

def lm_loss(model, X, Y):
    logits = model(X)
    # cross_entropy wants (N, V) logits, so join the B and T axes.
    return F.cross_entropy(logits.flatten(0, 1), Y.flatten())

@torch.no_grad()
def val_loss(model, ids, batches=50, B=256):
    g = torch.random.get_rng_state()
    torch.manual_seed(1234)   # the same val chunks every time, so the numbers compare
    loss = sum(lm_loss(model, *batch(ids, B)).item() for _ in range(batches)) / batches
    torch.random.set_rng_state(g)
    return loss

def train_model(model, train_ids, val_ids, STEPS=5000, B=64, LR=3e-3, log_every=1000):
    # AdamW: gradient descent that adapts the step size for each parameter.
    # Attention models train badly with plain SGD, so from here on we use this.
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    start = time.time()
    for step in range(1, STEPS + 1):
        loss = lm_loss(model, *batch(train_ids, B))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            print(f"step {step:5d}  train loss {loss.item():.4f}  val loss {val_loss(model, val_ids):.4f}"
                  f"  ({time.time() - start:.0f}s)")

# ---------- generate text ----------
@torch.no_grad()
def generate(model, n, start="\n"):
    """Sample n characters that follow the text `start`.

    No padding needed, unlike the MLP: attention works on any length up to BLOCK.
    Longer contexts are cut to their last BLOCK characters, because there are only
    BLOCK position embeddings.
    """
    ids = encode(start).tolist()
    out = []
    for _ in range(n):
        logits = model(torch.tensor([ids[-BLOCK:]]))
        # Only the last position's prediction matters: it predicts the next character.
        char_next = torch.multinomial(F.softmax(logits[0, -1], dim=-1), 1).item()
        ids.append(char_next)
        out.append(char_next)
    return decode(out)

# ---------- checks ----------
def show(weights, labels):
    """Print a (T, T) weight matrix with the characters along both edges."""
    labels = ["␣" if c == " " else repr(c)[1:-1] for c in labels]
    print("      " + "".join(f"{c:>6}" for c in labels))
    for c, row in zip(labels, weights.tolist()):
        print(f"{c:>6}" + "".join(f"{w:6.2f}" if w else "     ." for w in row))

def check_average():
    print("1. With all scores equal, each row is an even average of the positions so far:")
    weights = causal_weights(torch.zeros(4, 4))
    show(weights, "abcd")
    x = torch.tensor([[1.0], [2.0], [3.0], [6.0]])
    print(f"   weights @ {x.flatten().tolist()} = {(weights @ x).flatten().tolist()}  (running mean)")

def check_against_pytorch():
    q, k, v = torch.randn(3, 2, 10, 16).unbind(0)
    out, _ = attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"\n2. Largest difference from PyTorch's own attention: {(out - ref).abs().max():.2e}")

def check_causal():
    torch.manual_seed(0)
    head = Head(8, 4)
    x = torch.randn(1, 10, 8)
    before, _ = head(x)
    x[0, 6] = torch.randn(8)   # change the input at position 6
    after, _ = head(x)
    changed = [t for t in range(10) if not torch.equal(before[0, t], after[0, t])]
    print(f"\n3. Change position 6 of the input. Output positions that change: {changed}")
    print("   (expected 6 to 9: the past can't see the future)")

def check_scaling():
    hs = 64
    q, k = torch.randn(2, 1, 1000, hs).unbind(0)
    raw = q @ k.transpose(-2, -1)
    print(f"\n4. With hs = {hs}, the spread (std) of q @ k^T is {raw.std():.1f},"
          f" and after / sqrt(hs) it is {(raw / math.sqrt(hs)).std():.2f}.")
    print("   Big scores make softmax put nearly all the weight on one position, and the"
          " gradients vanish, like a saturated tanh.")

if __name__ == "__main__":
    check_average()
    check_against_pytorch()
    check_causal()
    check_scaling()

    print("\n5. Train a tiny model: embeddings -> one head -> logits.")
    torch.manual_seed(0)
    train_ids, val_ids = torch.tensor(train), torch.tensor(val)
    model = AttentionLM()
    print(f"   {sum(p.numel() for p in model.parameters()):,} parameters,"
          f" starting val loss {val_loss(model, val_ids):.4f}")
    train_model(model, train_ids, val_ids)

    text = "ROMEO:\nWhat light through yonder"
    weights = model.weights(torch.tensor(encode(text)).unsqueeze(0))
    print(f"\n   Where the trained head looks, for the last 8 characters of {text!r}:")
    show(weights[0, -8:, -8:], text[-8:])
    print("   (Rows that don't sum to 1 put the rest of their weight on earlier characters.)")
