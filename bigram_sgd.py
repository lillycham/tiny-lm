"""Bigram model trained with gradient descent instead of counting.

The model is a (V, V) table of logits W. For current character a, row W[a] goes
through softmax to give the probabilities for the next character. Training
should bring the validation loss close to the count-based model's 2.48.
"""
import numpy as np

from data import V, train, val

x_train, y_train = train[:-1], train[1:]
x_val, y_val = val[:-1], val[1:]

def softmax(logits):
    """
    turn each row of `logits` (shape (B, V)) into probabilities.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)

def loss(W, x, y):
    """
    mean negative log-likelihood of the pairs (x, y) under W.
    """
    P = softmax(W[x])
    correct = P[(np.arange(len(y)), y)]
    neg_log = -np.log(correct)

    return neg_log.mean()

# ---------- training ----------
def train_probs(rng, LR=50, STEPS=5000, BATCH=1024, log_every=500):
    """Train W with gradient descent and return its (V, V) probability table."""
    # Small random logits, so every row starts close to uniform (loss ~ ln(65) = 4.17).
    W = rng.normal(0, 0.01, size=(V, V))

    for step in range(STEPS + 1):
        if log_every and step % log_every == 0:
            print(f"step {step:5d}  val loss {loss(W, x_val, y_val):.4f}")

        # A random mini-batch of pairs.
        idx = rng.integers(0, len(x_train), BATCH)
        xb, yb = x_train[idx], y_train[idx]

        P = softmax(W[xb])                       # (BATCH, V)

        # gradient of the mean loss with respect to the logits, `d_logits`.
        d_logits = P.copy()
        d_logits[np.arange(BATCH), yb] -= 1
        d_logits /= BATCH

        # gradient for W, `gW`.
        gW = np.zeros_like(W)
        np.add.at(gW, xb, d_logits)

        # the gradient descent step.
        W -= LR * gW

    return softmax(W)

if __name__ == "__main__":
    from bigram import count_probs

    learned_probs = train_probs(np.random.default_rng(0))

    # ---------- compare with the count table ----------
    # If training worked, the learned probabilities should be close to the counted ones.
    counts = np.zeros((V, V))
    np.add.at(counts, (x_train, y_train), 1)
    # Rare characters (like '$') turn up in few batches, so their rows barely train.
    # Compare only the characters that appear at least 1,000 times.
    common = counts.sum(axis=1) >= 1000
    diff = np.abs(learned_probs - count_probs())[common].max()
    print(f"\nlargest difference from the count table, common characters: {diff:.3f}")
