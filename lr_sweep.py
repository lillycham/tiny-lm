"""Find a good learning rate with short runs, before a long one.

Each run starts from the same random weights and trains for --steps steps, with
the same warm-up and cosine decay as a long run, only shorter. Then it compares
the validation losses. Short runs tend to like slightly higher learning rates
than long ones, so pick the best, or one step below it if two are close.

    python lr_sweep.py                        # width 384, 500 steps, 4 learning rates
    python lr_sweep.py --emb 192 --lrs 1e-3 3e-3
"""
import argparse
import time

import torch

import gpt
import stories_gpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"], help="default mps")
    parser.add_argument("--steps", type=int, default=500, help="steps per run (default 500)")
    parser.add_argument("--emb", type=int, default=384, help="width (default 384)")
    parser.add_argument("--layers", type=int, default=stories_gpt.CONFIG["layers"])
    parser.add_argument("--heads", type=int, default=stories_gpt.CONFIG["heads"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[5e-4, 1e-3, 2e-3, 3e-3],
                        help="learning rates to try (default 5e-4 1e-3 2e-3 3e-3)")
    args = parser.parse_args()

    config = dict(stories_gpt.CONFIG, emb=args.emb, layers=args.layers, heads=args.heads)
    train_ids, val_ids = stories_gpt.load_tokens("train"), stories_gpt.load_tokens("val")
    tokens_per_step = stories_gpt.B * config["block"]
    results = []
    for lr in args.lrs:
        torch.manual_seed(0)
        model = gpt.GPT(**config).to(args.device)
        if not results:
            print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {args.steps} steps per run\n")
        print(f"LR {lr:g}", flush=True)
        t = time.time()
        gpt.train_model(model, args.device, STEPS=args.steps, B=stories_gpt.B, LR=lr,
                        log_every=args.steps // 2, train_ids=train_ids, val_ids=val_ids)
        seconds = time.time() - t
        results.append((lr, gpt.val_loss(model, val_ids, args.device), seconds))

    print("\n    LR   val loss   ms/step   tokens/s")
    for lr, loss, seconds in results:
        print(f"{lr:6g}   {loss:8.4f}   {seconds / args.steps * 1000:7.0f}   {tokens_per_step * args.steps / seconds:8,.0f}")
    best = min(results, key=lambda r: r[1])
    print(f"\nBest: LR {best[0]:g}. (The times include the two validation passes of each run.)")
