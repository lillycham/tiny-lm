"""Supervised fine-tuning (SFT): teach the story GPT to answer a request.

stories_gpt.py taught the model to continue stories. SFT keeps training the same
model, with the same loss, on conversations:

    User: Tell me a story about Lily.
    Assistant: Once upon a time, there was a girl named Lily...<|endoftext|>

The conversations come from TinyStories itself: every story with "named Lily"
in it becomes a request for a story about Lily. So the model already knows the
language, and only has to learn the format, and to use the name it was given.

The one new idea is loss masking. The model must learn to write the answer, not
to guess the user's request. So the targets for the user's tokens become -100,
and F.cross_entropy skips every target of -100. The same goes for padding.

    python sft.py          # about 5 minutes on the GPU, with the checks
"""
import argparse
import re
import time
from pathlib import Path

import torch

import gpt
import stories_gpt
from word_bpe import EOT_ID, TOKENISER, TRAIN_FILE, WordBPE, stories

CHECKPOINT = Path("checkpoints/stories_sft.pt")
BLOCK = stories_gpt.CONFIG["block"]
IGNORE = -100                     # F.cross_entropy skips targets with this value
NAMED = re.compile(r"\bnamed ([A-Z][a-z]+)")

def prompt(name):
    return f"User: Tell me a story about {name}.\nAssistant: "

def conversations(max_bytes):
    """(name, prompt, answer) for every story with "named <Name>" in it.

    The prompt ends with the space after "Assistant:", so the answer starts with
    "Once", as every story did in pretraining. (" Once" with its space is rare in
    TinyStories, so the tokeniser splits it into " " and "Once".)
    """
    for story in stories(TRAIN_FILE, max_bytes=max_bytes):
        m = NAMED.search(story)
        if m:
            yield m.group(1), prompt(m.group(1)), story

def make_example(tok, prompt, answer):
    """One conversation -> (x, y), two lists of BLOCK token IDs. None if too long.

    x is the input and y the targets, shifted by one as in every model so far.
    """
    ids = tok.encode(prompt) + tok.encode(answer) + [EOT_ID]
    if len(ids) > BLOCK + 1:
        return None
    # TODO(Lilly): four steps.
    #   1. x = ids[:-1] and y = ids[1:], as in gpt.batch.
    #   2. Mask the prompt. y[t] is the token after x[t]. The first answer token
    #      is ids[P], where P = the number of prompt tokens, and it is y[P - 1].
    #      So y[0] .. y[P - 2] are prompt tokens: set them to IGNORE.
    #      (Slice assignment works on lists: y[:k] = [IGNORE] * k.)
    #   3. Pad both to BLOCK: x with EOT_ID, y with IGNORE. Padding is never
    #      learned from, so what x holds there doesn't matter.
    #   4. return x, y
    x, y = ids[:-1], ids[1:]
    P = len(tok.encode(prompt))

    y[:P - 1] = [IGNORE] * (P - 1)

    pad = BLOCK - len(x)

    return x + [EOT_ID] * pad, y + [IGNORE] * pad

def dataset(tok, max_bytes):
    """Every conversation that fits, as two (N, BLOCK) tensors, and the names in it."""
    X, Y, names = [], [], set()
    for name, p, a in conversations(max_bytes):
        ex = make_example(tok, p, a)
        if ex:
            X.append(ex[0])
            Y.append(ex[1])
            names.add(name)
    return torch.tensor(X), torch.tensor(Y), names

@torch.no_grad()
def answer_loss(model, X, Y, device, B=64):
    """Mean loss on the answer tokens only: the masked targets don't count."""
    model.eval()
    total = n = 0
    for i in range(0, len(X), B):
        x, y = X[i:i + B].to(device), Y[i:i + B].to(device)
        k = (y != IGNORE).sum().item()
        total += gpt.lm_loss(model, x, y).item() * k
        n += k
    model.train()
    return total / n

def name_test(model, tok, names, samples=3, temperature=0.8):
    """For each name, how many of the model's stories use that name."""
    return {name: sum(name in gpt.generate(model, 250, prompt(name), temperature, tok.encode, tok.decode,
                                           stop=EOT_ID) for _ in range(samples))
            for name in names}

def show_test(title, result, samples, seen):
    known = [n for n in result if n in seen]
    new = [n for n in result if n not in seen]
    print(f"{title}: {sum(result[n] for n in known)}/{len(known) * samples} for names in the SFT data,"
          f" {sum(result[n] for n in new)}/{len(new) * samples} for names it never saw")

def fine_tune(model, X, Y, device, STEPS, B=32, LR=3e-4, WARMUP=50, val=None):
    """AdamW, a short warm-up, then a constant learning rate: SFT is short."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    model.train()
    start = time.time()
    for step in range(STEPS):
        for group in opt.param_groups:
            group["lr"] = LR * min(1, (step + 1) / WARMUP)
        i = torch.randint(0, len(X), (B,))
        loss = gpt.lm_loss(model, X[i].to(device), Y[i].to(device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"step {step + 1:4d}  train loss {loss.item():.4f}"
                  f"  val answer loss {answer_loss(model, *val, device):.4f}  ({time.time() - start:.0f}s)", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"], help="default mps")
    parser.add_argument("--steps", type=int, default=600, help="fine-tuning steps (default 600)")
    args = parser.parse_args()
    tok = WordBPE.load(TOKENISER)

    x, y = make_example(tok, prompt("Lily"), "Once upon a time.")
    P = len(tok.encode(prompt("Lily")))
    print("1. One example: the target at each position, with * if it is learned")
    print("   " + "  ".join(f"{tok.decode([x[i + 1] if y[i] == IGNORE else y[i]])!r}{'' if y[i] == IGNORE else '*'}"
                            for i in range(P + 5)))
    print(f"   {sum(t != IGNORE for t in y)} of {BLOCK} targets are learned (expected 6: 5 answer tokens and"
          f" <|endoftext|>), and the input is {len(x)} long")

    t = time.time()
    X, Y, seen = dataset(tok, max_bytes=20_000_000)
    n_val = 500
    val = X[:n_val], Y[:n_val]
    X, Y = X[n_val:], Y[n_val:]
    print(f"\n2. {len(X):,} training and {n_val} validation conversations, {len(seen)} names ({time.time() - t:.0f}s)")

    model = gpt.load(args.device, stories_gpt.CHECKPOINT)
    names = ["Lily", "Tim", "Sue", "Max", "Mia", "Ben", "Jack", "Daisy", "Priya", "Oscar", "Hana", "Leo", "Nina"]
    samples = 3
    torch.manual_seed(0)
    before = name_test(model, tok, names, samples)
    print(f"\n3. Before SFT: val answer loss {answer_loss(model, *val, args.device):.4f}")
    show_test("   Stories that use the name", before, samples, seen)
    torch.manual_seed(0)
    print("   " + prompt("Priya").replace("\n", "\n   ")
          + gpt.generate(model, 120, prompt("Priya"), 0.8, tok.encode, tok.decode, stop=EOT_ID) + "\n")

    fine_tune(model, X, Y, args.device, STEPS=args.steps, val=val)
    gpt.save(model, CHECKPOINT)

    torch.manual_seed(0)
    after = name_test(model, tok, names, samples)
    print(f"\n4. After SFT: val answer loss {answer_loss(model, *val, args.device):.4f}")
    show_test("   Stories that use the name", after, samples, seen)
    print("   " + "  ".join(f"{n} {before[n]}->{after[n]}" for n in names))
    torch.manual_seed(0)
    print("\n   " + prompt("Priya").replace("\n", "\n   ")
          + gpt.generate(model, 250, prompt("Priya"), 0.8, tok.encode, tok.decode, stop=EOT_ID))
    print(f"\nSaved to {CHECKPOINT}")
