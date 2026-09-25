"""Tiny Shakespeare as character IDs, shared by the models."""
import numpy as np

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
