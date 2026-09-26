"""Generate Shakespeare-ish text from one of the models.

    python main.py counts
    python main.py sgd -n 1000 --seed 7
    python main.py mlp --start "ROMEO:"
    python main.py torch --device mps
    python main.py attention --start "ROMEO:"
    python main.py gpt --temperature 0.8     # after python gpt.py has saved a checkpoint
    python main.py bpe -n 250                # after python bpe_gpt.py has saved a checkpoint
    python main.py stories --temperature 0.8 # after python stories_gpt.py has saved a checkpoint
    python main.py tokens --start "Once upon a time, Tom's cat sat."
                                             # how the two BPE tokenisers split the text
"""
import argparse
import sys

import numpy as np

import bigram
import bigram_sgd
import mlp
from data import stoi, train, val

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

def build_torch(rng, args):
    # Import here, because torch takes a moment to load and the other models don't need it.
    import torch
    import mlp_torch

    # PyTorch has its own random generator, so seed it from --seed.
    torch.manual_seed(args.seed)
    device = args.device or "cpu"
    X_train, Y_train = (t.to(device) for t in mlp_torch.windows(train))
    X_val, Y_val = (t.to(device) for t in mlp_torch.windows(val))
    model = mlp_torch.MLP().to(device)
    mlp_torch.train_model(model, X_train, Y_train, X_val, Y_val, STEPS=args.steps or 60_000)
    loss = mlp_torch.full_loss(model, X_val, Y_val)
    return loss, lambda n, rng, start: mlp_torch.generate(model, n, start)

def build_attention(rng, args):
    import torch
    import attention

    torch.manual_seed(args.seed)
    train_ids, val_ids = torch.tensor(train), torch.tensor(val)
    model = attention.AttentionLM()
    attention.train_model(model, train_ids, val_ids, STEPS=args.steps or 5000)
    # An estimate from 50 random batches of validation text, as in attention.py.
    return attention.val_loss(model, val_ids), lambda n, rng, start: attention.generate(model, n, start)

def build_gpt(rng, args):
    import torch
    import gpt

    # Training takes minutes, so gpt.py trains once and saves the model; this only loads it.
    if not gpt.CHECKPOINT.exists():
        sys.exit(f"No trained model at {gpt.CHECKPOINT}. Run python gpt.py first.")
    torch.manual_seed(args.seed)
    device = args.device or "mps"
    model = gpt.load(device)
    loss = gpt.val_loss(model, torch.tensor(val), device)
    return loss, lambda n, rng, start: gpt.generate(model, n, start, args.temperature)

def build_bpe(rng, args):
    import torch
    import bpe_gpt
    import gpt

    if not bpe_gpt.CHECKPOINT.exists():
        sys.exit(f"No trained model at {bpe_gpt.CHECKPOINT}. Run python bpe_gpt.py first.")
    torch.manual_seed(args.seed)
    device = args.device or "mps"
    tok = bpe_gpt.BPE.load(bpe_gpt.TOKENISER)
    model = gpt.load(device, bpe_gpt.CHECKPOINT)
    loss = gpt.val_loss(model, torch.tensor(tok.encode(bpe_gpt.text[bpe_gpt.split:])), device)
    return loss, lambda n, rng, start: gpt.generate(model, n, start, args.temperature, tok.encode, tok.decode)

def build_stories(rng, args):
    import torch
    import gpt
    import stories_gpt

    if not stories_gpt.CHECKPOINT.exists():
        sys.exit(f"No trained model at {stories_gpt.CHECKPOINT}. Run python stories_gpt.py first.")
    torch.manual_seed(args.seed)
    device = args.device or "mps"
    tok = stories_gpt.WordBPE.load(stories_gpt.TOKENISER)
    model = gpt.load(device, stories_gpt.CHECKPOINT)
    loss = gpt.val_loss(model, stories_gpt.load_tokens("val"), device)
    return loss, lambda n, rng, start: stories_gpt.story(model, tok, start, args.temperature, n)[len(start):]

MODELS = {"counts": build_counts, "sgd": build_sgd, "mlp": build_mlp, "torch": build_torch,
          "attention": build_attention, "gpt": build_gpt, "bpe": build_bpe, "stories": build_stories}

def show_tokens(text):
    """Print how each saved BPE tokeniser splits text, with | between the tokens."""
    import bpe_gpt
    import word_bpe

    tokenisers = [("Shakespeare BPE (bpe.py)", bpe_gpt.TOKENISER, bpe_gpt.BPE),
                  ("TinyStories BPE (word_bpe.py)", word_bpe.TOKENISER, word_bpe.WordBPE)]
    for name, path, cls in tokenisers:
        if not path.exists():
            print(f"{name}: no tokeniser at {path} yet\n")
            continue
        tok = cls.load(path)
        ids = tok.encode(text)
        print(f"{name}: {len(ids)} tokens for {len(text)} characters\n{tok.show(ids)}\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=[*MODELS, "tokens"],
                        help="counts: count table; sgd: bigram trained with gradient descent; "
                             "mlp: 8-character context; torch: the same MLP in PyTorch; "
                             "attention: one attention head over 32 characters; "
                             "gpt: a small transformer, loaded from the checkpoint gpt.py saves; "
                             "bpe: the same transformer on BPE tokens, from the checkpoint bpe_gpt.py saves "
                             "(its val loss is per token, not per character); "
                             "stories: a GPT-2 shaped transformer on TinyStories, from the checkpoint "
                             "stories_gpt.py saves (per token; it stops when a story ends); "
                             "tokens: no model, only show how the BPE tokenisers split --start")
    parser.add_argument("-n", type=int, default=500,
                        help="number of characters to generate, or tokens for bpe and stories (default 500)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for training and sampling (default 0)")
    parser.add_argument("--start",
                        help="text to continue from (default newline, or 'Once upon a time' for stories);"
                             " the bigrams use its last character")
    parser.add_argument("--smoothing", type=float, default=1, help="counts: pseudo-count added to every pair (default 1)")
    parser.add_argument("--steps", type=int, help="training steps (default 5000 for sgd and attention, 60000 for mlp and torch)")
    parser.add_argument("--device", choices=["cpu", "mps"],
                        help="torch: device to train on (default cpu, faster for this model); "
                             "gpt, bpe and stories: device to run on (default mps)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="gpt, bpe and stories: below 1 gives safer, more repetitive text; above 1 more random (default 1)")
    args = parser.parse_args()
    if args.start is None:
        args.start = "Once upon a time" if args.model == "stories" else "\n"
    if args.model == "tokens":
        # Any text works here, because both tokenisers start from bytes.
        show_tokens(args.start)
        return
    # The stories model reads bytes, so any text works. The others know only Shakespeare's characters.
    unknown = [] if args.model == "stories" else sorted(set(args.start) - set(stoi))
    if not args.start or unknown:
        parser.error(f"--start must be non-empty text made of characters from the training text; unknown: {unknown}")

    rng = np.random.default_rng(args.seed)
    loss, sample = MODELS[args.model](rng, args)
    print(f"[{args.model}] val loss {loss:.4f}\n")
    print(args.start + sample(args.n, rng, args.start))

if __name__ == "__main__":
    main()
