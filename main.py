"""Generate Shakespeare-ish text from one of the models.

    python main.py counts
    python main.py sgd -n 1000 --seed 7
"""
import argparse

import numpy as np

from bigram import count_probs, generate, val_loss
from bigram_sgd import train_probs
from data import stoi

MODELS = {
    "counts": lambda rng, args: count_probs(S=args.smoothing),
    "sgd": lambda rng, args: train_probs(rng, STEPS=args.steps, log_every=0),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=MODELS, help="counts: count table; sgd: trained with gradient descent")
    parser.add_argument("-n", type=int, default=500, help="number of characters to generate (default 500)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for training and sampling (default 0)")
    parser.add_argument("--start", default="\n", help="character to start after (default newline)")
    parser.add_argument("--smoothing", type=float, default=1, help="counts: pseudo-count added to every pair (default 1)")
    parser.add_argument("--steps", type=int, default=5000, help="sgd: training steps (default 5000)")
    args = parser.parse_args()
    if args.start not in stoi:
        parser.error(f"--start must be one character from the training text, not {args.start!r}")

    rng = np.random.default_rng(args.seed)
    probs = MODELS[args.model](rng, args)
    print(f"[{args.model}] val loss {val_loss(probs):.4f}\n")
    print(generate(probs, args.n, rng, start=args.start))

if __name__ == "__main__":
    main()
