"""Character-level bigram language model on Tiny Shakespeare.

A bigram model predicts the next character from the current character only.
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------- data ----------
text = open("data/tinyshakespeare.txt").read()

# Vocabulary: every distinct character, with a lookup table each way.
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

def encode(s):
    return np.array([stoi[c] for c in s])

def decode(ids):
    return "".join(itos[int(i)] for i in ids)

data = encode(text)

# First 90% to train on, last 10% to check the model on text it hasn't seen.
split = int(0.9 * len(data))
train, val = data[:split], data[split:]

# Each training example is a pair: (current character, next character).
x_train, y_train = train[:-1], train[1:]
x_val, y_val = val[:-1], val[1:]

print(f"vocab size {V}, train pairs {len(x_train):,}, val pairs {len(x_val):,}")
print("example pairs:", [(itos[a], itos[b]) for a, b in zip(x_train[:8], y_train[:8])])

# ---------- step 1: count-based bigram ----------
# TODO(Lilly): build a (V, V) array `counts` where counts[a, b] is how many times
#   character b follows character a in the training pairs.
#   Hint: np.add.at(array, (rows, cols), 1) adds 1 at every (row, col) pair,
#   including repeated pairs. (Plain `counts[x, y] += 1` counts each pair only once.)

# TODO(Lilly): turn `counts` into a (V, V) array `probs`, where each row sums to 1.
#   Row a is then the model's probability distribution for the character after a.
#   Question to think about: what happens to a pair that never appears in training?

# TODO(Lilly): compute the average negative log-likelihood on the validation pairs:
#   the mean of -log(probs[x, y]) over all (x, y) in (x_val, y_val).
#   That's the same cross-entropy loss as the circle classifier, with V classes instead of 2.
