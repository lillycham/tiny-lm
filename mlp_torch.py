"""The MLP language model again, in PyTorch.

Same model as mlp.py: 8 characters of context -> embeddings -> tanh layer -> logits.
The difference is that autograd writes grads() for us: we only write the forward
pass, and loss.backward() computes every gradient.

    python mlp_torch.py            # autograd check against mlp.py, then training
    python mlp_torch.py --device mps
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import V, decode, stoi, train, val

BLOCK, EMB, HIDDEN = 8, 16, 300

# ---------- data ----------
def windows(ids):
    """Same (X, Y) windows as mlp.py, as tensors.

    unfold(0, BLOCK, 1) is PyTorch's sliding_window_view: windows of size BLOCK, step 1.
    """
    ids = torch.tensor(ids)
    return ids[:-1].unfold(0, BLOCK, 1), ids[BLOCK:]

# ---------- model ----------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # create the three layers as attributes, so PyTorch tracks their parameters:
        self.C  = nn.Embedding(V, EMB)
        self.l1 = nn.Linear(BLOCK * EMB, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, V)

        # Start with tiny output weights, as in mlp.py, so the starting loss is about ln(65).
        # (nn.Linear already scales its weights by 1/sqrt(in_size), like our W1.)
        with torch.no_grad():
            self.l2.weight.mul_(0.01)
            self.l2.bias.zero_()

    def forward(self, X):
        """Logits of shape (B, V) for a batch X of shape (B, BLOCK)."""
        E = self.C(X).flatten(1)

        H = torch.tanh(self.l1(E))

        logits = self.l2(H)

        return logits

@torch.no_grad()   # no gradients needed, so PyTorch can skip the autograd bookkeeping
def full_loss(model, X, Y, chunk=50_000):
    total = 0.0
    for i in range(0, len(X), chunk):
        total += F.cross_entropy(model(X[i:i + chunk]), Y[i:i + chunk], reduction="sum").item()
    return total / len(X)

# ---------- training ----------
def train_model(model, X_train, Y_train, X_val, Y_val, LR=0.2, STEPS=60_000, BATCH=128, log_every=10_000):
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    # Divide the learning rate by 10 for the last quarter, as in mlp.py.
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(0.75 * STEPS)], gamma=0.1)
    start = time.time()

    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(X_train), (BATCH,), device=X_train.device)
        xb, yb = X_train[idx], Y_train[idx]

        loss = F.cross_entropy(model(xb), yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        sched.step()

        if log_every and step % log_every == 0:
            print(f"step {step:6d}  train loss {full_loss(model, X_train[:200_000], Y_train[:200_000]):.4f}"
                  f"  val loss {full_loss(model, X_val, Y_val):.4f}  ({time.time() - start:.0f}s)")

# ---------- generate text ----------
@torch.no_grad()
def generate(model, n, start="\n"):
    """Same sliding-window sampling as mlp.py."""
    device = next(model.parameters()).device
    context = ([stoi["\n"]] * BLOCK + [stoi[c] for c in start])[-BLOCK:]
    out = []
    for _ in range(n):
        logits = model(torch.tensor([context], device=device))
        char_next = torch.multinomial(F.softmax(logits, dim=1)[0], 1).item()
        out.append(char_next)
        context = context[1:] + [char_next]
    return decode(out)

# ---------- autograd vs. your grads() ----------
def autograd_check():
    """Copy mlp.py's parameters into the PyTorch model and compare the gradients."""
    import mlp

    model = MLP().double()   # float64, like NumPy, so the numbers can match closely
    with torch.no_grad():
        model.C.weight.copy_(torch.from_numpy(mlp.params["C"]))
        # nn.Linear stores its weight as (out, in), the transpose of our W1 (in, out).
        model.l1.weight.copy_(torch.from_numpy(mlp.params["W1"].T))
        model.l1.bias.copy_(torch.from_numpy(mlp.params["b1"]))
        model.l2.weight.copy_(torch.from_numpy(mlp.params["W2"].T))
        model.l2.bias.copy_(torch.from_numpy(mlp.params["b2"]))

    X, Y = mlp.X_train[:32], mlp.Y_train[:32]
    F.cross_entropy(model(torch.from_numpy(X.copy())), torch.from_numpy(Y)).backward()
    yours = mlp.grads(mlp.params, X, Y)

    pairs = {
        "C": (yours["C"], model.C.weight.grad),
        "W1": (yours["W1"], model.l1.weight.grad.T),
        "b1": (yours["b1"], model.l1.bias.grad),
        "W2": (yours["W2"], model.l2.weight.grad.T),
        "b2": (yours["b2"], model.l2.bias.grad),
    }
    for name, (g_np, g_torch) in pairs.items():
        print(f"{name:>2}  largest difference {np.abs(g_np - g_torch.numpy()).max():.2e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                        help="cpu is faster for a model this small (default cpu)")
    args = parser.parse_args()

    print("autograd vs. your grads() from mlp.py:")
    autograd_check()

    torch.manual_seed(0)
    X_train, Y_train = (t.to(args.device) for t in windows(train))
    X_val, Y_val = (t.to(args.device) for t in windows(val))
    model = MLP().to(args.device)
    print(f"\nstarting val loss {full_loss(model, X_val, Y_val):.4f}")
    train_model(model, X_train, Y_train, X_val, Y_val)

    print("\nROMEO:" + generate(model, 300, start="ROMEO:"))
