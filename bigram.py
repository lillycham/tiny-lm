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

# Initialise the counts - a (V, V) array. Each counts[a, b] is how many times char
# b follows a in the training pairs.
counts = np.zeros((V, V), dtype=int)

# Smoothing value
S = 1

np.add.at(counts, (x_train, y_train), 1)


# Initialising the probability array - row A is the probability distribution for
# each character after a.
probs = (counts + S) / (counts + S).sum(axis=1, keepdims=True)

# Compute the avg. neg. log-likelihood on validation pairs
neg_log = -np.log(probs[x_val, y_val])

val_loss = neg_log.mean()
print(f"negative log-likelihood: {val_loss}")
