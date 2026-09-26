"""The GPT from gpt.py, trained on BPE tokens rather than characters.

The model is the same size, with the same 64 positions of context. Only the
vocabulary changes: 512 BPE tokens, not 65 characters. Each token is about 2
characters, so the 64 positions now see about 125 characters of text.

The losses aren't directly comparable. The character model's loss is per
character, and this model's loss is per token, which has about 2 characters to
guess. To compare them, both become bits per character.

    python bpe_gpt.py          # about 10 minutes on the GPU
    python main.py bpe         # then generate from the saved model
"""
import argparse
import math
import time
from pathlib import Path

import torch

import gpt
from bpe import BPE, VOCAB_SIZE
from data import split, text, val

TOKENISER = Path("checkpoints/bpe.json")
CHECKPOINT = Path("checkpoints/gpt_bpe.pt")

def bits_per_char(loss, chars_per_token):
    """A loss in nats per token -> bits per character."""
    # TODO(Lilly): two steps.
    #   1. Nats per token -> nats per character. A token holds chars_per_token
    #      characters, and its loss is the cost of guessing all of them. Share that
    #      cost out equally between the characters.
    #   2. Nats -> bits. A nat uses log base e and a bit uses log base 2, and
    #      log2(x) = ln(x) / ln(2). So divide by math.log(2).
    raise NotImplementedError

def get_tokeniser(train_text):
    """Load the saved tokeniser, or train one and save it."""
    if TOKENISER.exists():
        return BPE.load(TOKENISER)
    print("Training the tokeniser...", flush=True)
    bpe = BPE.train(train_text, VOCAB_SIZE)
    bpe.save(TOKENISER)
    return bpe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"], help="default mps")
    parser.add_argument("--steps", type=int, default=5000, help="training steps (default 5000)")
    args = parser.parse_args()

    train_text, val_text = text[:split], text[split:]
    bpe = get_tokeniser(train_text)
    t = time.time()
    # TODO(Lilly): encode train_text and val_text with bpe, and make each one a tensor
    #   with torch.tensor(...). These replace the character IDs from data.py.
    train_ids = val_ids = None
    chars_per_token = len(val_text) / len(val_ids)
    print(f"Encoded in {time.time() - t:.0f}s: {len(train_ids):,} training tokens, {len(val_ids):,} validation tokens,"
          f" {chars_per_token:.2f} characters per token", flush=True)

    print(f"\n1. bits_per_char(1.5, 1) = {bits_per_char(1.5, 1):.3f} and bits_per_char(3.0, 2) = {bits_per_char(3.0, 2):.3f}")
    print("   (expected 2.164 for both: 3 nats for 2 characters is 1.5 nats for each)")

    char_loss = None
    if gpt.CHECKPOINT.exists():
        char_loss = gpt.val_loss(gpt.load(args.device), torch.tensor(val), args.device)
        print(f"\n2. Character GPT: {char_loss:.4f} nats per character = {bits_per_char(char_loss, 1):.3f} bits per character")

    torch.manual_seed(0)
    model = gpt.GPT(**gpt.CONFIG, vocab=VOCAB_SIZE).to(args.device)
    print(f"\n{sum(p.numel() for p in model.parameters()):,} parameters on {args.device},"
          f" starting val loss {gpt.val_loss(model, val_ids, args.device):.4f} per token", flush=True)
    gpt.train_model(model, args.device, STEPS=args.steps, train_ids=train_ids, val_ids=val_ids)
    gpt.save(model, CHECKPOINT)

    loss = gpt.val_loss(model, val_ids, args.device)
    print(f"\n3. BPE GPT: {loss:.4f} nats per token = {bits_per_char(loss, chars_per_token):.3f} bits per character")
    if char_loss is not None:
        print(f"   Character GPT: {bits_per_char(char_loss, 1):.3f} bits per character")
    print(f"\nSaved to {CHECKPOINT}\n")
    print("ROMEO:" + gpt.generate(model, 250, start="ROMEO:", encode=bpe.encode, decode=bpe.decode))
