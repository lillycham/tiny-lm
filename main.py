"""Generate Shakespeare-ish text from one of the models.

    python main.py counts
    python main.py sgd -n 1000 --seed 7
    python main.py mlp --start "ROMEO:"
"""
import argparse

import numpy as np

import bigram
import bigram_sgd
import mlp
from data import stoi

# Each model builder returns (validation loss, sample function).
# sample(n, rng, start) generates n characters that follow the text `start`.

def build_counts(rng, args):
    probs = bigram.count_probs(S=args.smoothing)
    # A bigram only sees one character of context, so it continues from the last one.
    return bigram.val_loss(probs), lambda n, rng, start: bigram.generate(probs, n, rng, start[-1])

def build_sgd(rng, args):
    probs = bigram_sgd.train_probs(rng, STEPS=args.steps or 5000, log_every=0)
    return bigram.val_loss(probs), lambda n, rng, start: bigram.generate(probs, n, rng, start[-1])

def build_mlp(rng, args):
    mlp.train_model(mlp.params, rng, STEPS=args.steps or 60_000, log_every=10_000)
    loss = mlp.full_loss(mlp.params, mlp.X_val, mlp.Y_val)
    return loss, lambda n, rng, start: mlp.generate(mlp.params, n, rng, start)

MODELS = {"counts": build_counts, "sgd": build_sgd, "mlp": build_mlp}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=MODELS,
                        help="counts: count table; sgd: bigram trained with gradient descent; mlp: 8-character context")
    parser.add_argument("-n", type=int, default=500, help="number of characters to generate (default 500)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for training and sampling (default 0)")
    parser.add_argument("--start", default="\n",
                        help="text to continue from (default newline); the bigrams use its last character")
    parser.add_argument("--smoothing", type=float, default=1, help="counts: pseudo-count added to every pair (default 1)")
    parser.add_argument("--steps", type=int, help="sgd, mlp: training steps (default 5000 for sgd, 60000 for mlp)")
    args = parser.parse_args()
    unknown = sorted(set(args.start) - set(stoi))
    if not args.start or unknown:
        parser.error(f"--start must be non-empty text made of characters from the training text; unknown: {unknown}")

    rng = np.random.default_rng(args.seed)
    loss, sample = MODELS[args.model](rng, args)
    print(f"[{args.model}] val loss {loss:.4f}\n")
    print(args.start + sample(args.n, rng, args.start))

if __name__ == "__main__":
    main()
