"""A transformer block: the attention head from attention.py, grown into the real thing.

One head gave 2.34, because it could do only one kind of look-up and had no hidden
layer. A block fixes both:

    x = x + MultiHeadAttention(LayerNorm(x))    the positions share information
    x = x + FeedForward(LayerNorm(x))           each position thinks about what it got

and a transformer is a stack of these blocks. Every block keeps the shape (B, T, C),
so they stack like Lego.

    python transformer.py
"""
import torch
import torch.nn as nn

from attention import BLOCK, Head, batch, generate, train_model, val_loss
from data import V, train, val

EMB, N_HEAD = 64, 4

# ---------- the parts of a block ----------
class MultiHeadAttention(nn.Module):
    """Several heads side by side, each with its own q, k and v.

    Each head can learn a different pattern: one may look at the last character,
    another at the start of the word, another at the last newline. With C = 64 and
    4 heads, each head has hs = 16, so joined together they give back C = 64.
    """

    def __init__(self, C, n_head):
        super().__init__()
        # nn.ModuleList is a list that PyTorch can see into, so it tracks the heads' parameters.
        self.heads = nn.ModuleList([Head(C, C // n_head) for _ in range(n_head)])
        # A layer to mix the heads' results together.
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        """x (B, T, C) -> (B, T, C)."""
        head_outs = [out for out, _ in (h(x) for h in self.heads)]

        joined = torch.cat(head_outs, dim=-1)

        return self.proj(joined)

class FeedForward(nn.Module):
    """The hidden layer: the same small MLP on every position, separately.

    Attention only makes weighted averages, which are linear. This is where each
    position does non-linear work on what attention collected.
    """

    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C, 4 * C), nn.ReLU(), nn.Linear(4 * C, C))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Multi-head attention, then feed-forward, each with a LayerNorm and a residual connection.

    LayerNorm: rescales each position's vector to mean 0 and spread 1 (then a learned
    scale and shift), so the numbers stay a sensible size however deep the stack.

    Residual connection: x = x + f(x), not x = f(x). Each part only adds a change to
    x. In the backward pass the + passes the gradient straight through, so even the
    first block gets a strong gradient in a deep stack.
    """

    def __init__(self, C, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = MultiHeadAttention(C, n_head)
        self.ln2 = nn.LayerNorm(C)
        self.ffwd = FeedForward(C)

    def forward(self, x):
        """x (B, T, C) -> (B, T, C)."""
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

# ---------- the model ----------
class TransformerLM(nn.Module):
    """Embeddings -> n_layer blocks -> LayerNorm -> logits."""

    def __init__(self, n_layer):
        super().__init__()
        self.tok = nn.Embedding(V, EMB)
        self.pos = nn.Embedding(BLOCK, EMB)
        self.blocks = nn.Sequential(*[Block(EMB, N_HEAD) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(EMB)
        self.out = nn.Linear(EMB, V)

    def forward(self, X):
        """X (B, T) IDs -> logits (B, T, V)."""
        x = self.tok(X) + self.pos(torch.arange(X.shape[1], device=X.device))
        return self.out(self.ln(self.blocks(x)))

# ---------- checks ----------
def check_shapes():
    x = torch.randn(2, 10, EMB)
    mha = MultiHeadAttention(EMB, N_HEAD)(x)
    ff = FeedForward(EMB)(x)
    block = Block(EMB, N_HEAD)(x)
    print(f"1. Input {tuple(x.shape)} -> multi-head {tuple(mha.shape)}, feed-forward {tuple(ff.shape)},"
          f" block {tuple(block.shape)}")
    print("   (all four should be the same)")

def check_causal():
    torch.manual_seed(0)
    model = TransformerLM(n_layer=4)
    X = torch.randint(0, V, (1, 10))
    before = model(X)
    X[0, 6] = (X[0, 6] + 1) % V   # change the character at position 6
    after = model(X)
    changed = [t for t in range(10) if not torch.equal(before[0, t], after[0, t])]
    print(f"\n2. Through 4 blocks: change character 6, and the outputs that change are {changed}")
    print("   (expected 6 to 9)")

if __name__ == "__main__":
    check_shapes()
    check_causal()

    train_ids, val_ids = torch.tensor(train), torch.tensor(val)
    for n_layer in (1, 4):
        torch.manual_seed(0)
        model = TransformerLM(n_layer)
        print(f"\n{n_layer} block{'s' * (n_layer > 1)}, {sum(p.numel() for p in model.parameters()):,} parameters,"
              f" starting val loss {val_loss(model, val_ids):.4f}")
        train_model(model, train_ids, val_ids, LR=1e-3)

    print("\nROMEO:" + generate(model, 300, start="ROMEO:"))
