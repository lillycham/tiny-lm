"""SFT on Dolly 15k: a general chat model from the web model.

instruct_sft.py taught one kind of request. Dolly is ~15,000 questions and answers
written by people at Databricks: open questions, questions about a given paragraph,
classification, brainstorming, summaries and creative writing (CC BY-SA 3.0).

    User: Why can camels survive for long without water?
    Assistant: Camels use the fat in their humps ...<|endoftext|>

With a paragraph to answer from, it comes after the question and a blank line.
The base is the web model, annealed with 25% stories by default: that anneal kept
all of its web loss, so it knows as much as the web model and writes better stories.

    curl -L -o data/dolly-15k.jsonl \\
        https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
    python chat_sft.py                # a few minutes on the GPU

A 98M model will answer in the right form, but it will invent most facts.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import gpt
import sft
from web_data import TOKENISER
from word_bpe import WordBPE

DOLLY = Path("data/dolly-15k.jsonl")
BASE = Path("checkpoints/web_gpt_768w12l_anneal25.pt")
BLOCK = 512             # ~93% of the conversations fit
QUESTIONS = ["What is the capital of France?", "Why is the sky blue?", "Give me three ideas for a rainy day.",
             "Write a short poem about the sea.", "What is photosynthesis?", "Who was Albert Einstein?"]

def prompt(instruction, context=""):
    if context.strip():
        return f"User: {instruction.strip()}\n\n{context.strip()}\nAssistant: "
    return f"User: {instruction.strip()}\nAssistant: "

def load_rows(path=DOLLY):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def split(rows, n_val=500, seed=0):
    """(train, val): the same shuffle every time, so val is always held out."""
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    return rows[n_val:], rows[:n_val]

def dataset(tok, rows, block=BLOCK):
    """Every conversation that fits in block, as two (N, block) int16 tensors."""
    X, Y = [], []
    for r in rows:
        ex = sft.make_example(tok, prompt(r["instruction"], r["context"]), r["response"].strip(), block)
        if ex:
            X.append(np.array(ex[0], dtype=np.int16))
            Y.append(np.array(ex[1], dtype=np.int16))
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))

def answer(model, tok, question, context="", temperature=0.7, n=200):
    return gpt.generate(model, n, prompt(question, context), temperature, tok.encode, tok.decode, stop=tok.eot_id)

def show_answers(model, tok, questions):
    torch.manual_seed(0)
    for q in questions:
        print(f"   User: {q}\n   Assistant: {answer(model, tok, q).strip()}\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    parser.add_argument("--base", type=Path, default=BASE, help="the model to fine-tune (default %(default)s)")
    parser.add_argument("--steps", type=int, default=1000, help="fine-tuning steps (default 1000, ~2 passes)")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate (default 5e-5)")
    parser.add_argument("--tag", help="a label for the checkpoint name")
    args = parser.parse_args()
    out = args.base.with_name(f"{args.base.stem}_chat" + (f"_{args.tag}" if args.tag else "") + ".pt")
    tok = WordBPE.load(TOKENISER)

    t = time.time()
    train, held_out = split(load_rows())
    X, Y = dataset(tok, train)
    val = dataset(tok, held_out)
    print(f"1. {len(X):,} of {len(train):,} training conversations fit in {BLOCK} tokens;"
          f" {len(val[0])} validation ({time.time() - t:.0f}s)")
    print("   " + prompt(train[0]["instruction"], train[0]["context"]).replace("\n", "\n   ")[:400])

    model = gpt.load(args.device, args.base)
    print(f"\n2. Before SFT: val answer loss {sft.answer_loss(model, *val, args.device):.4f}\n")
    show_answers(model, tok, QUESTIONS[:3])

    sft.fine_tune(model, X, Y, args.device, STEPS=args.steps, LR=args.lr, val=val)
    gpt.save(model, out)

    print(f"\n3. After SFT: val answer loss {sft.answer_loss(model, *val, args.device):.4f}\n")
    show_answers(model, tok, QUESTIONS)
    print("   Held-out questions, with Dolly's answer:\n")
    torch.manual_seed(0)
    for r in [r for r in held_out if not r["context"]][:5]:
        print(f"   User: {r['instruction']}\n   Assistant: {answer(model, tok, r['instruction']).strip()}"
              f"\n   (Dolly: {r['response'].strip()[:300]})\n", flush=True)
    print(f"Saved to {out}")
