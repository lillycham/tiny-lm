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
from data import V, train, val

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
    # TODO(Lilly): build X with shape (len(ids) - BLOCK, BLOCK) and Y with shape (len(ids) - BLOCK,).
    #   A list comprehension over i with slices works, but it's slow on a million characters.
    #   Faster: np.lib.stride_tricks.sliding_window_view(ids, BLOCK) gives every window
    #   of length BLOCK. Check which windows you need to drop so X and Y line up.
    ...

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

    # TODO(Lilly): H = the tanh hidden layer. Same shape of formula as the circle classifier.

    # TODO(Lilly): P = softmax of the output logits.
    ...

def loss(params, X, Y):
    """Mean negative log-likelihood of the examples (X, Y)."""
    # TODO(Lilly): the same loss as bigram_sgd.py, with P from forward().
    ...

def full_loss(params, X, Y, chunk=20_000):
    """loss() over a big dataset, in chunks so the arrays stay a sensible size."""
    total = 0.0
    for i in range(0, len(X), chunk):
        total += loss(params, X[i:i + chunk], Y[i:i + chunk]) * len(X[i:i + chunk])
    return total / len(X)

if __name__ == "__main__":
    print(f"X_train {X_train.shape}, Y_train {Y_train.shape}")
    n_params = sum(p.size for p in params.values())
    print(f"{n_params:,} parameters")
    # Before any training, this should be close to ln(65) = 4.17.
    print(f"starting val loss {full_loss(params, X_val, Y_val):.4f}")
