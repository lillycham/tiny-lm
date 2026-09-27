"""SFT on TinyStoriesInstruct: stories that follow a request.

sft.py asked for a story about a name. Here the request is harder: three words the
story must use, and sometimes features it must have.

    User: Tell me a story that uses the words help, mud and bossy, with dialogue and a twist.
    Assistant: Lily and Sam were playing in the park...<|endoftext|>

TinyStoriesInstruct comes from the TinyStories authors. Each story has labelled
fields before it (Words:, Features:, Summary:, Random sentence:), in any order. We
keep the stories that have Words: and come after their fields, and turn Words: and
Features: into the request. The model starts from the base story model, not from
sft.py's model, so the before and after are a clean comparison.

    python -m stories.instruct_sft     # about 8 minutes on the GPU, with the checks
    python -m stories.instruct_sft --train-file data/TinyStories-Instruct-train-300MB.txt --steps 6000 --tag 300mb
    python -m tools.main instruct --start "dragon, soup, happy"

The web model (web_gpt_*, Part 3) works too, with its own tokeniser:
    python -m stories.instruct_sft --base checkpoints/web/web_gpt_768w12l_anneal.pt \
        --train-file data/TinyStories-Instruct-train-300MB.txt --steps 2000 --lr 1e-4 --tag 300mb
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch

from core import gpt
from stories import sft
from stories import stories_gpt
from web import web_data
from core.word_bpe import EOT, TOKENISER, WordBPE

TRAIN_FILE = Path("data/TinyStories-Instruct-train-30MB.txt")   # the first 30 MB of the 2.7 GB file
VAL_FILE = Path("data/TinyStories-Instruct-valid.txt")
CHECKPOINT = Path("checkpoints/stories/stories_instruct.pt")

def output_path(base, tag=None):
    """Where to save: stories_instruct.pt with base's sizes added, then _tag if given,
    in base's folder (checkpoints/web/ for a web base)."""
    out = stories_gpt.fine_tuned_path(CHECKPOINT, base)
    out = base.parent / out.name
    return out.with_name(f"{out.stem}_{tag}.pt") if tag else out

def tokeniser_for(base):
    """The web models have their own tokeniser; every other base uses the story one."""
    return web_data.TOKENISER if "web_gpt" in base.name else TOKENISER

# How each feature reads in a request: "..., with dialogue and a twist."
FEATURES = {
    "Dialogue": "dialogue",
    "Twist": "a twist",
    "MoralValue": "a moral",
    "BadEnding": "a bad ending",
    "Foreshadowing": "foreshadowing",
    "Conflict": "a conflict",
}
FIELD = re.compile(r"^(Features|Words|Summary|Random sentence|Story):[ \t]*", re.M)

def examples(path):
    """Each example in the file as a dict of its fields, in their order, as strings.

    Read line by line, so even the full 2.7 GB file needs little memory. Text after
    the last <|endoftext|> is dropped: the file was cut off in the middle of it.
    """
    chunk = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            chunk += line
            while EOT in chunk:
                done, chunk = chunk.split(EOT, 1)
                # re.split with a group gives ["", name, value, name, value, ...].
                parts = FIELD.split(done.strip())
                yield dict(zip(parts[1::2], (v.strip() for v in parts[2::2])))

def request(words, features):
    """(["help", "mud", "bossy"], ["Dialogue", "Twist"]) ->
    "Tell me a story that uses the words help, mud and bossy, with dialogue and a twist."

    With no features: "Tell me a story that uses the words help, mud and bossy."
    """
    def and_list(items):
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]

    text = "Tell me a story that uses the words " + and_list(words)

    if features:
        text += ", with " + and_list([FEATURES[f] for f in features])

    return text + "."

def prompt(req):
    return f"User: {req}\nAssistant: "

def conversations(path):
    """(words, features, prompt, story) for every example with words, and the story last."""
    for ex in examples(path):
        if "Words" not in ex or "Story" not in ex or list(ex)[-1] != "Story":
            continue
        words = [w.strip() for w in ex["Words"].split(",")]
        features = [f.strip() for f in ex.get("Features", "").split(",") if f.strip() in FEATURES]
        yield words, features, prompt(request(words, features)), ex["Story"]

def dataset(tok, path, limit=None):
    """Every conversation that fits in the context, as two (N, BLOCK) int16 tensors.

    int16 holds every token ID (under 4,096) and the -100 mask. A Python list of 256
    ints costs about 9 KB, so the full 2.7 GB file as lists would need about 17 GB;
    as int16 rows it needs about 1 GB. fine_tune and answer_loss make each batch int64.
    """
    X, Y = [], []
    for words, features, p, story in conversations(path):
        ex = sft.make_example(tok, p, story)
        if ex:
            X.append(np.array(ex[0], dtype=np.int16))
            Y.append(np.array(ex[1], dtype=np.int16))
            if limit and len(X) == limit:
                break
    if not X:
        raise ValueError(f"No conversations in {path} fit in the context")
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))

def words_used(story, words):
    """How many of words appear in story. "help" counts in "helped", and case doesn't matter."""
    # TODO(Lilly): count the words w for which re.search(pattern, story, re.IGNORECASE)
    #   finds something. The pattern: r"\b" + re.escape(w). \b is a word boundary, so
    #   "help" matches "help" and "helped" but not "whelp". re.escape keeps any
    #   punctuation in w from acting as regex syntax.
    #   sum() over True and False counts the Trues, as in name_test in sft.py.
    return sum(bool(re.search(r"\b" + re.escape(w), story, re.IGNORECASE)) for w in words)


def follow_test(model, tok, tests, temperature=0.8):
    """Word and dialogue scores on the held-out requests in tests."""
    used = total = 0
    quotes = {True: [0, 0], False: [0, 0]}     # asked for dialogue? -> [stories with ", stories]
    for words, features, p, _ in tests:
        story = gpt.generate(model, 300, p, temperature, tok.encode, tok.decode, stop=tok.eot_id)
        used += words_used(story, words)
        total += len(words)
        q = quotes["Dialogue" in features]
        q[0] += '"' in story
        q[1] += 1
    return used, total, quotes

def show_test(title, result):
    used, total, quotes = result
    print(f"{title}: {used}/{total} required words used ({used / total:.0%});"
          f" quotes in {quotes[True][0]}/{quotes[True][1]} stories asked for dialogue,"
          f" {quotes[False][0]}/{quotes[False][1]} not asked")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    parser.add_argument("--steps", type=int, default=600, help="fine-tuning steps (default 600)")
    parser.add_argument("--base", type=Path, default=stories_gpt.CHECKPOINT,
                        help=f"the model to fine-tune (default {stories_gpt.CHECKPOINT})")
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE,
                        help=f"the training examples (default {TRAIN_FILE})")
    parser.add_argument("--tag", help="a label for the checkpoint name, e.g. 300mb, so it doesn't replace the default run")
    parser.add_argument("--tests", type=int, default=40, help="held-out requests to test (default 40)")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate (default 3e-4; lower for big bases)")
    args = parser.parse_args()
    out = output_path(args.base, args.tag)
    tok = WordBPE.load(tokeniser_for(args.base))

    print(f"1. {request(['help', 'mud', 'bossy'], ['Dialogue', 'Twist'])}")
    print(f"   {request(['dragon'], [])}")
    print('   (expected "Tell me a story that uses the words help, mud and bossy, with dialogue and a twist."'
          ' and "Tell me a story that uses the words dragon.")')
    print(f"   words_used: {words_used('Lily helped. The MUD was fun. A whelp.', ['help', 'mud', 'bossy'])}"
          " (expected 2: helped and MUD count, bossy and whelp don't)")

    t = time.time()
    X, Y = dataset(tok, args.train_file)
    held_out = list(conversations(VAL_FILE))
    val = dataset(tok, VAL_FILE, limit=500)
    tests = held_out[-args.tests:]      # the end of the file, away from the 500 for the val loss
    print(f"\n2. {len(X):,} training conversations, 500 validation, {len(tests)} test requests"
          f" ({time.time() - t:.0f}s)")
    print("   " + tests[0][2].replace("\n", "\n   "))

    model = gpt.load(args.device, args.base)
    torch.manual_seed(0)
    print(f"\n3. Before SFT: val answer loss {sft.answer_loss(model, *val, args.device):.4f}")
    show_test("   Test", follow_test(model, tok, tests))

    sft.fine_tune(model, X, Y, args.device, STEPS=args.steps, LR=args.lr, val=val)
    gpt.save(model, out)

    torch.manual_seed(0)
    print(f"\n4. After SFT: val answer loss {sft.answer_loss(model, *val, args.device):.4f}")
    show_test("   Test", follow_test(model, tok, tests))
    torch.manual_seed(0)
    p = prompt(request(["dragon", "soup", "happy"], ["Dialogue"]))
    print("\n   " + p.replace("\n", "\n   ")
          + gpt.generate(model, 300, p, 0.8, tok.encode, tok.decode, stop=tok.eot_id))
    print(f"\nSaved to {out}")
