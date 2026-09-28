"""Anneal the web model: a short last stage of training on SimpleStories, mixed with web text.

Modern pretraining ends like this: the broad data first, then a small amount of clean data
while the learning rate falls to near zero. Skills learned last are not overwritten.
Half the mix is the web text the model already trained on, so it doesn't forget it.

SimpleStories (Finke et al. 2025): ~2.1M short stories by GPT-4o-mini, with more varied
topics, styles and names than TinyStories. Two of its 7 train files (~175M tokens) and
its test file, encoded with the web tokeniser:

    python -m web.anneal encode                  # data/simplestories_{train,val}.bin
    python -m web.anneal --compile               # web_gpt_768w12l.pt -> web_gpt_768w12l_anneal.pt

--data picks other token files, data/NAME_{train,val}.bin: e.g. --data wiki, from
web/wikipedia.py. --share is its fraction of the mix.

The learning rate starts where the main run ended (6e-5, its LR / 10). AdamW's state
isn't in the checkpoint, so it starts again: a short warm-up, then a cosine down.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from core import gpt
from web import web_data
from web import web_gpt
from web.web_data import TOKENISER
from core.word_bpe import WordBPE

STORY_FILES = [Path(f"data/simplestories_train_{i:02d}.parquet") for i in range(2)]
STORY_TEST = Path("data/simplestories_test.parquet")
STORY_TOKENS = Path("data/simplestories_{split}.bin")
PROMPTS = ["Once upon a time, a girl named Priya", "The old lighthouse keeper", "Tom and Mia went to the park. Mia gave the ball to"]

def story_jobs(split):
    """(path, row group) for every row group of the SimpleStories files for split."""
    files = STORY_FILES if split == "train" else [STORY_TEST]
    return [(p, g) for p in files for g in range(pq.ParquetFile(p).num_row_groups)]

def load_story_tokens(split, data="simplestories"):
    """The anneal data's tokens: data/simplestories_{split}.bin by default."""
    return np.memmap(f"data/{data}_{split}.bin", dtype=np.uint16, mode="r")

def mix(stories, web, fraction=0.5, seed=0):
    """All of stories, then a random stretch of web, sized so stories are `fraction` of
    the whole. batch() takes chunks at random places, so a fraction of the chunks are stories."""
    n_web = round(len(stories) * (1 - fraction) / fraction)
    start = int(np.random.default_rng(seed).integers(0, len(web) - n_web + 1))
    return np.concatenate([np.asarray(stories), np.asarray(web[start:start + n_web])])

def losses(model, device, val_B, data="simplestories"):
    """(anneal data val loss, web val loss): what the anneal should lower, and what it shouldn't raise."""
    return (gpt.val_loss(model, load_story_tokens("val", data), device, B=val_B),
            gpt.val_loss(model, web_data.load_tokens("val"), device, B=val_B))

if __name__ == "__main__":
    if sys.argv[1:2] == ["encode"]:
        for split in ("val", "train"):
            web_data.encode(story_jobs(split), Path(str(STORY_TOKENS).format(split=split)), column="story")
        sys.exit()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    parser.add_argument("--base", type=Path, default=web_gpt.checkpoint_path(web_gpt.CONFIG),
                        help="the model to anneal (default %(default)s)")
    parser.add_argument("--tag", default="anneal", help="checkpoint label (default %(default)s)")
    parser.add_argument("--tokens", type=float, default=1.5e8, help="training tokens (default 1.5e8)")
    parser.add_argument("--data", default="simplestories",
                        help="anneal data: data/NAME_{train,val}.bin (default %(default)s; wiki: web/wikipedia.py)")
    parser.add_argument("--share", "--stories", type=float, default=0.5,
                        help="its fraction of the mix, the rest web text (default 0.5)")
    parser.add_argument("--batch", type=int, default=32, help="sequences per micro-batch (default 32)")
    parser.add_argument("--accum", type=int, default=4, help="micro-batches per step (default 4)")
    parser.add_argument("--val-batch", type=int, default=16, help="sequences per val loss batch (default 16)")
    parser.add_argument("--lr", type=float, default=6e-5, help="peak learning rate (default 6e-5)")
    parser.add_argument("--warmup", type=int, default=100, help="warm-up steps (default 100)")
    parser.add_argument("--log-every", type=int, default=100, help="steps between val losses (default 100)")
    parser.add_argument("--compile", action="store_true", help="train through torch.compile: for CUDA")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = gpt.load(args.device, args.base)
    out = web_gpt.checkpoint_path(model.config, args.tag)
    train_ids = mix(load_story_tokens("train", args.data), web_data.load_tokens("train"), args.share, args.seed)
    steps = web_gpt.steps_for(args.tokens, args.batch, model.block, args.accum)
    print(f"{args.base}: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Mix: {len(train_ids):,} tokens, {args.share:.0%} {args.data}."
          f" {steps:,} steps: {steps * args.accum * args.batch * model.block / 1e6:.0f}M tokens")
    story, web = losses(model, args.device, args.val_batch, args.data)
    print(f"\nBefore: {args.data} val loss {story:.4f}, web val loss {web:.4f}\n", flush=True)

    t = time.time()
    gpt.train_model(model, args.device, STEPS=steps, B=args.batch, LR=args.lr, WARMUP=args.warmup,
                    log_every=args.log_every, train_ids=train_ids, val_ids=load_story_tokens("val", args.data),
                    resume=out.with_suffix(".resume.pt"), compile=args.compile, val_B=args.val_batch,
                    accum=args.accum)
    gpt.save(model, out)
    out.with_suffix(".resume.pt").unlink(missing_ok=True)
    story_after, web_after = losses(model, args.device, args.val_batch, args.data)
    print(f"\nTrained in {(time.time() - t) / 60:.0f} minutes. Saved to {out}")
    print(f"After:  {args.data} val loss {story_after:.4f} ({story_after - story:+.4f}),"
          f" web val loss {web_after:.4f} ({web_after - web:+.4f})\n")
    tok = WordBPE.load(TOKENISER)
    torch.manual_seed(0)
    for prompt in PROMPTS + web_gpt.PROMPTS:
        print(web_gpt.continue_text(model, tok, prompt) + "\n")
