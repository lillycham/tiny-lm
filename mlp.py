"""MLP language model (Bengio et al., 2003) on Tiny Shakespeare.

Predicts the next character from the previous BLOCK characters:

    BLOCK char IDs -> embeddings (lookup in C) -> concatenate -> tanh hidden layer -> logits -> softmax

Shapes, for a batch of B examples:
    X  (B, BLOCK)            character IDs
    E  (B, BLOCK * EMB)      embeddings, concatenated per example
    H  (B, HIDDEN)           hidden layer
    P  (B, V)                next-character probabilities
"""
import numpy as np

from bigram_sgd import softmax
from data import V, decode, stoi, train, val

BLOCK = 8      # characters of context
EMB = 16       # size of each character's embedding vector
HIDDEN = 300   # hidden layer neurons

rng = np.random.default_rng(0)

# ---------- data: context windows ----------
def windows(ids):
    """Split a 1D array of character IDs into (X, Y) training examples.

    X[i] is BLOCK consecutive IDs and Y[i] is the ID that comes right after them.
    For ids = [a, b, c, d, e] and BLOCK = 3:
        X = [[a, b, c],      Y = [d,
             [b, c, d]]           e]
    """
    X = np.lib.stride_tricks.sliding_window_view(ids[:-1], BLOCK)
    Y = ids[BLOCK:]
    return X, Y

X_train, Y_train = windows(train)
X_val, Y_val = windows(val)

# ---------- parameters ----------
# The scaling keeps the starting activations a sensible size: tanh isn't saturated,
# and the output logits start near zero, so the starting loss is close to ln(65).
params = {
    "C": rng.normal(0, 1, (V, EMB)),
    "W1": rng.normal(0, 1, (BLOCK * EMB, HIDDEN)) / np.sqrt(BLOCK * EMB),
    "b1": np.zeros(HIDDEN),
    "W2": rng.normal(0, 1, (HIDDEN, V)) * 0.01,
    "b2": np.zeros(V),
}

# ---------- forward pass ----------
def forward(params, X):
    """Return (E, H, P) for a batch X of shape (B, BLOCK). E and H are needed for backprop later."""
    C, W1, b1, W2, b2 = (params[k] for k in ("C", "W1", "b1", "W2", "b2"))
    # TODO(Lilly): E = the embeddings of every ID in X, concatenated per example.
    #   C[X] looks up a row of C for every ID. Check its shape, then reshape it to (B, BLOCK * EMB).
    #   Tip: .reshape(len(X), -1) lets NumPy work out the second size.
    E = C[X].reshape(len(X), -1)

    # TODO(Lilly): H = the tanh hidden layer. Same shape of formula as the circle classifier.
    H = np.tanh(E @ W1 + b1)

    # TODO(Lilly): P = softmax of the output logits.
    P = softmax(H @ W2 + b2)
    return E, H, P

def loss(params, X, Y):
    """Mean negative log-likelihood of the examples (X, Y)."""
    # TODO(Lilly): the same loss as bigram_sgd.py, with P from forward().
    _,_,P = forward(params, X)
    correct = P[(np.arange(len(Y)), Y)]
    neg_log = -np.log(correct)

    return neg_log.mean()


def full_loss(params, X, Y, chunk=20_000):
    """loss() over a big dataset, in chunks so the arrays stay a sensible size."""
    total = 0.0
    for i in range(0, len(X), chunk):
        total += loss(params, X[i:i + chunk], Y[i:i + chunk]) * len(X[i:i + chunk])
    return total / len(X)

# ---------- backward pass ----------
def grads(params, X, Y):
    """Gradients of loss(params, X, Y) for every parameter, as a dict with the same keys as params."""
    C, W1, b1, W2, b2 = (params[k] for k in ("C", "W1", "b1", "W2", "b2"))
    B = len(X)
    E, H, P = forward(params, X)

    # TODO(Lilly): d_logits, shape (B, V). Same as bigram_sgd.py.

    # TODO(Lilly): output layer. The layer was logits = H @ W2 + b2, so:
    #   gW2 = H.T @ d_logits           (300, B) @ (B, 65) -> (300, 65), same shape as W2
    #   gb2 = d_logits summed over the batch axis, shape (65,)
    #   dH  = d_logits @ W2.T          the gradient flowing back into H, shape (B, 300)

    # TODO(Lilly): through tanh. H = tanh(z), and tanh'(z) = 1 - H**2.
    #   dz = dH times that, element by element. Shape (B, 300).

    # TODO(Lilly): hidden layer. It was z = E @ W1 + b1, so this is the same pattern as the
    #   output layer: gW1 from E and dz, gb1 by summing dz, and dE flowing back into E.
    #   Check that each gradient has the same shape as its parameter.

    # TODO(Lilly): embeddings. E was C[X] reshaped to (B, BLOCK * EMB).
    #   1. Undo the reshape: dE back to (B, BLOCK, EMB), the shape C[X] had.
    #   2. Scatter-add into gC = np.zeros_like(C) with np.add.at, indexed by X.
    #      Each of the B * BLOCK looked-up rows sends its gradient back to the row of C it came from.

    return {"C": gC, "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

def grad_check(params, X, Y, checks=3, eps=1e-5):
    """Compare grads() with a numerical estimate on a few entries of each parameter."""
    g = grads(params, X, Y)
    check_rng = np.random.default_rng(1)
    for name, p in params.items():
        for _ in range(checks):
            if name == "C":
                # Only rows of characters in this batch get a gradient, so check one of those.
                i = (int(check_rng.choice(X.ravel())), int(check_rng.integers(EMB)))
            else:
                i = tuple(int(check_rng.integers(0, s)) for s in p.shape)
            old = p[i]
            p[i] = old + eps
            up = loss(params, X, Y)
            p[i] = old - eps
            down = loss(params, X, Y)
            p[i] = old
            numeric = (up - down) / (2 * eps)
            print(f"{name:>2}{str(i):>12}  yours {g[name][i]: .8f}  numeric {numeric: .8f}")

# ---------- training ----------
def train_model(params, rng, LR=0.2, STEPS=60_000, BATCH=128, log_every=5_000):
    """Train params in place with mini-batch gradient descent."""
    for step in range(1, STEPS + 1):
        idx = rng.integers(0, len(X_train), BATCH)
        g = grads(params, X_train[idx], Y_train[idx])

        # Learning-rate decay: smaller steps for the last quarter, so the weights settle
        # instead of bouncing around the minimum.
        lr = LR if step <= 0.75 * STEPS else LR / 10

        # TODO(Lilly): the gradient descent step, for every parameter in params.
        #   Loop over the names, and update each array in place with -=.

        if log_every and step % log_every == 0:
            # Training loss on a 200k sample, so the check stays quick.
            train_loss = full_loss(params, X_train[:200_000], Y_train[:200_000])
            print(f"step {step:6d}  train loss {train_loss:.4f}  val loss {full_loss(params, X_val, Y_val):.4f}")

# ---------- generate text ----------
def generate(params, n, rng, start="\n"):
    """Sample n characters, continuing from the text `start`.

    The model always needs BLOCK characters of context, so a short start is padded
    on the left with newlines, and a long one is cut to its last BLOCK characters.
    """
    context = ([stoi["\n"]] * BLOCK + [stoi[c] for c in start])[-BLOCK:]
    out = []
    for _ in range(n):
        _, _, P = forward(params, np.array([context]))
        char_next = rng.choice(V, p=P[0])
        out.append(char_next)
        # Slide the window along: drop the oldest character, add the new one.
        context = context[1:] + [char_next]
    return decode(out)

if __name__ == "__main__":
    print(f"X_train {X_train.shape}, Y_train {Y_train.shape}")
    n_params = sum(p.size for p in params.values())
    print(f"{n_params:,} parameters")
    # Before any training, this should be close to ln(65) = 4.17.
    print(f"starting val loss {full_loss(params, X_val, Y_val):.4f}")

    print("\ngradient check, on 32 examples:")
    grad_check(params, X_train[:32], Y_train[:32])

    print("\ntraining:")
    train_model(params, rng)
