"""A GPT trained on TinyStories: short, simple stories, in 4,096 word-level BPE tokens.

The model is GPT-2 shaped (GELU, tied embeddings), so it can become a Hugging Face
GPT-2 model later. With 256 tokens of context it sees a whole story at a time.

Shakespeare was 511k tokens, so the models there saw each token about 40 times and
overfit. TinyStories has 553M tokens, and this run sees only 61M of them, each one
once. So there's no overfitting, and no need for dropout.

    python word_bpe.py            # first, to make the tokens
    python stories_gpt.py         # about 30 minutes on the GPU
    python main.py stories        # then write stories with the saved model
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

import gpt
from word_bpe import EOT_ID, TOKENISER, TOKENS, VOCAB_SIZE, WordBPE

CONFIG = dict(block=256, emb=192, heads=6, layers=6, dropout=0.0, vocab=VOCAB_SIZE, gelu=True, tied=True)
CHECKPOINT = Path("checkpoints/stories_gpt.pt")
B = 32

def load_tokens(split):
    """The token IDs that word_bpe.py saved for split ("train" or "val")."""
    # TODO(Lilly): np.load the file str(TOKENS).format(split=split), with mmap_mode="r".
    #   That maps the file into memory without reading it: NumPy reads each part from
    #   disk only when you use it. So the 1.1 GB file costs almost no memory.
    raise NotImplementedError

def story(model, tok, start="Once upon a time", temperature=1.0, n=400):
    """Write one story that starts with start, until the model ends it."""
    return start + gpt.generate(model, n, start, temperature, tok.encode, tok.decode, stop=EOT_ID)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"], help="default mps")
    parser.add_argument("--steps", type=int, default=7500, help="training steps (default 7500)")
    args = parser.parse_args()

    train_ids, val_ids = load_tokens("train"), load_tokens("val")
    tok = WordBPE.load(TOKENISER)
    print(f"{len(train_ids):,} training tokens, {len(val_ids):,} validation tokens")

    torch.manual_seed(0)
    model = gpt.GPT(**CONFIG).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = args.steps * B * CONFIG["block"]
    print(f"\n1. {n_params:,} parameters. The embeddings are shared, so they count once:"
          f" {CONFIG['vocab']} x {CONFIG['emb']} = {CONFIG['vocab'] * CONFIG['emb']:,}")
    print(f"   Training sees {n_tokens / 1e6:.0f}M tokens, {n_tokens / n_params:.0f} per parameter")
    start_loss = gpt.val_loss(model, val_ids, args.device)
    print(f"\n2. Starting val loss {start_loss:.4f}, expected about ln({VOCAB_SIZE}) = {math.log(VOCAB_SIZE):.4f}\n",
          flush=True)

    t = time.time()
    gpt.train_model(model, args.device, STEPS=args.steps, B=B, train_ids=train_ids, val_ids=val_ids)
    gpt.save(model, CHECKPOINT)
    print(f"\nTrained in {(time.time() - t) / 60:.0f} minutes. Saved to {CHECKPOINT}\n")
    torch.manual_seed(0)
    for _ in range(2):
        print(story(model, tok) + "\n")
