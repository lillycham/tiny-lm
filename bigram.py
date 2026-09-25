"""Character-level bigram language model on Tiny Shakespeare.

A bigram model predicts the next character from the current character only.
"""
import numpy as np

from data import V, decode, stoi, train, val

# Each training example is a pair: (current character, next character).
x_train, y_train = train[:-1], train[1:]
x_val, y_val = val[:-1], val[1:]

# ---------- step 1: count-based bigram ----------
def count_probs(S=1):
    """(V, V) next-character probabilities from pair counts, with smoothing value S."""
    # Initialise the counts - a (V, V) array. Each counts[a, b] is how many times char
    # b follows a in the training pairs.
    counts = np.zeros((V, V), dtype=int)
    np.add.at(counts, (x_train, y_train), 1)

    # Initialising the probability array - row A is the probability distribution for
    # each character after a.
    return (counts + S) / (counts + S).sum(axis=1, keepdims=True)

def val_loss(probs):
    """Avg. neg. log-likelihood of a (V, V) probability table on the validation pairs."""
    neg_log = -np.log(probs[x_val, y_val])
    return neg_log.mean()

# ---------- generate text ----------
def generate(probs, n, rng, start="\n"):
    """Sample n characters from a (V, V) probability table, starting after `start`."""
    current = stoi[start]
    out = []
    for _ in range(n):
        char_next = rng.choice(V, p=probs[current])
        out.append(char_next)
        current = char_next
    return decode(out)

if __name__ == "__main__":
    print(f"vocab size {V}, train pairs {len(x_train):,}, val pairs {len(x_val):,}")
    probs = count_probs()
    print(f"negative log-likelihood: {val_loss(probs)}")
    print(generate(probs, 500, np.random.default_rng(0)))
