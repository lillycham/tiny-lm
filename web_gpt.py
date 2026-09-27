"""Part 3: a GPT-2 small sized model on FineWeb-Edu, general English from the web.

stories_gpt.py's model, scaled up: 12 layers, width 768, 12 heads, 1,024 tokens of
context. The same blocks as GPT-2 small; with our 16,384-token vocabulary the
embedding is 12.6M instead of 38.6M, so ~98M parameters instead of 124M. The 85M in
the blocks is the same.

Chinchilla: about 20 training tokens per parameter, ~2B. The default, 2.7B tokens, is
~4 of FineWeb-Edu's big files after web_data.py has encoded them.

    python web_data.py encode train data/fineweb_edu_00{0,1,2,3}.parquet --max-tokens 2.7e9
    python web_gpt.py --compile                            # on a rented GPU, ~5-6 h on a 5090
    python web_gpt.py --emb 256 --heads 4 --layers 4 --block 256 --batch 8 --accum 1 --tokens 5e6   # a quick check
"""
import argparse
import math
import time
from pathlib import Path

import torch

import gpt
import web_data
from web_data import EOT_ID, TOKENISER, VOCAB_SIZE
from word_bpe import WordBPE

CONFIG = dict(block=1024, emb=768, heads=12, layers=12, dropout=0.0, vocab=VOCAB_SIZE, gelu=True, tied=True,
              scaled_init=True)
CHECKPOINTS = Path("checkpoints")
PROMPTS = ["The water cycle is", "In 1905, Albert Einstein", "To make bread, you need"]

def checkpoint_path(config, tag=None):
    """checkpoints/web_gpt_768w12l.pt, then _tag if given."""
    return CHECKPOINTS / (f"web_gpt_{config['emb']}w{config['layers']}l" + (f"_{tag}" if tag else "") + ".pt")

def steps_for(tokens, B, block, accum=1):
    """Training steps to see about `tokens` tokens, accum x B sequences of block tokens a step."""
    return max(1, round(tokens / (accum * B * block)))

def count(model):
    """(all parameters, the ones outside the embeddings). Chinchilla counts the second."""
    total = sum(p.numel() for p in model.parameters())
    return total, total - model.tok.weight.numel() - model.pos.weight.numel()

def continue_text(model, tok, start, temperature=0.8, n=150):
    return start + gpt.generate(model, n, start, temperature, tok.encode, tok.decode, stop=EOT_ID)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    for name in ("block", "emb", "heads", "layers"):
        parser.add_argument(f"--{name}", type=int, default=CONFIG[name], help=f"default {CONFIG[name]}")
    parser.add_argument("--tokens", type=float, default=2.7e9, help="training tokens (default 2.7e9)")
    parser.add_argument("--batch", type=int, default=32, help="sequences per micro-batch (default 32)")
    parser.add_argument("--accum", type=int, default=4,
                        help="micro-batches per step (default 4: 4 x 32 x 1,024 = 131k tokens a step; 16 is GPT-2's 0.5M)")
    parser.add_argument("--val-batch", type=int, default=16,
                        help="sequences per val loss batch (default 16; 64 would need ~4 GB for its logits)")
    parser.add_argument("--lr", type=float, default=6e-4, help="peak learning rate (default 6e-4, as GPT-2 small)")
    parser.add_argument("--warmup", type=int, default=500, help="warm-up steps (default 500)")
    parser.add_argument("--log-every", type=int, default=250, help="steps between val losses (default 250)")
    parser.add_argument("--tag", help="a label for the checkpoint name")
    parser.add_argument("--snapshots", action="store_true",
                        help="also save the model at steps 64, 128, 256, ... in checkpoints/snapshots/")
    parser.add_argument("--compile", action="store_true", help="train through torch.compile: for CUDA")
    parser.add_argument("--scaled-init", action=argparse.BooleanOptionalAction, default=CONFIG["scaled_init"],
                        help="GPT-2's smaller init for the layers that write to the residual stream (default on)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    args = parser.parse_args()
    if args.emb % args.heads:
        parser.error("--emb must be a multiple of --heads, so every head gets the same size")

    config = dict(CONFIG, block=args.block, emb=args.emb, heads=args.heads, layers=args.layers,
                  scaled_init=args.scaled_init)
    out = checkpoint_path(config, args.tag)
    resume = out.with_suffix(".resume.pt")
    steps = steps_for(args.tokens, args.batch, args.block, args.accum)

    train_ids, val_ids = web_data.load_tokens("train"), web_data.load_tokens("val")
    tok = WordBPE.load(TOKENISER)
    print(f"{len(train_ids):,} training tokens, {len(val_ids):,} validation tokens")

    torch.manual_seed(args.seed)
    model = gpt.GPT(**config).to(args.device)
    total, blocks = count(model)
    seen = steps * args.accum * args.batch * args.block
    print(f"\n1. {total:,} parameters, {blocks:,} outside the embeddings")
    print(f"   {steps:,} steps of {args.accum} x {args.batch} x {args.block} tokens: {seen / 1e9:.2f}B tokens,"
          f" {seen / blocks:.0f} per parameter, {seen / len(train_ids):.2f} passes over the data")
    if seen > len(train_ids):
        print("   (more than one pass: encode more files for new data)")
    start_loss = gpt.val_loss(model, val_ids, args.device, B=args.val_batch)
    print(f"\n2. Starting val loss {start_loss:.4f}, expected about ln({VOCAB_SIZE}) = {math.log(VOCAB_SIZE):.4f}\n",
          flush=True)

    t = time.time()
    gpt.train_model(model, args.device, STEPS=steps, B=args.batch, LR=args.lr, WARMUP=args.warmup,
                    log_every=args.log_every, train_ids=train_ids, val_ids=val_ids, resume=resume,
                    snapshots=CHECKPOINTS / "snapshots" / out.stem if args.snapshots else None,
                    compile=args.compile, val_B=args.val_batch, accum=args.accum)
    gpt.save(model, out)
    resume.unlink(missing_ok=True)
    print(f"\nTrained in {(time.time() - t) / 60:.0f} minutes. Saved to {out}\n")
    torch.manual_seed(0)
    for prompt in PROMPTS:
        print(continue_text(model, tok, prompt) + "\n")
