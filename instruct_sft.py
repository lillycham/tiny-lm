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

    python instruct_sft.py     # about 8 minutes on the GPU, with the checks
    python main.py instruct --start "dragon, soup, happy"
"""
import argparse
import re
import time
from pathlib import Path

import torch

import gpt
import sft
import stories_gpt
from word_bpe import EOT, EOT_ID, TOKENISER, WordBPE

TRAIN_FILE = Path("data/TinyStories-Instruct-train-30MB.txt")   # the first 30 MB of the 2.7 GB file
VAL_FILE = Path("data/TinyStories-Instruct-valid.txt")
CHECKPOINT = Path("checkpoints/stories_instruct.pt")

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
    """Each example in the file as a dict of its fields, in their order, as strings."""
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = text.split(EOT)
    if not text.rstrip().endswith(EOT):
        chunks = chunks[:-1]        # the file was cut off in the middle of the last one
    for chunk in chunks:
        # re.split with a group gives ["", name, value, name, value, ...].
        parts = FIELD.split(chunk.strip())
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
    """Every conversation that fits in the context, as two (N, BLOCK) tensors."""
    X, Y = [], []
    for words, features, p, story in conversations(path):
        ex = sft.make_example(tok, p, story)
        if ex:
            X.append(ex[0])
            Y.append(ex[1])
            if limit and len(X) == limit:
                break
    return torch.tensor(X), torch.tensor(Y)

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
        story = gpt.generate(model, 300, p, temperature, tok.encode, tok.decode, stop=EOT_ID)
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
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"], help="default mps")
    parser.add_argument("--steps", type=int, default=600, help="fine-tuning steps (default 600)")
    parser.add_argument("--tests", type=int, default=40, help="held-out requests to test (default 40)")
    args = parser.parse_args()
    tok = WordBPE.load(TOKENISER)

    print(f"1. {request(['help', 'mud', 'bossy'], ['Dialogue', 'Twist'])}")
    print(f"   {request(['dragon'], [])}")
    print('   (expected "Tell me a story that uses the words help, mud and bossy, with dialogue and a twist."'
          ' and "Tell me a story that uses the words dragon.")')
    print(f"   words_used: {words_used('Lily helped. The MUD was fun. A whelp.', ['help', 'mud', 'bossy'])}"
          " (expected 2: helped and MUD count, bossy and whelp don't)")

    t = time.time()
    X, Y = dataset(tok, TRAIN_FILE)
    held_out = list(conversations(VAL_FILE))
    val = dataset(tok, VAL_FILE, limit=500)
    tests = held_out[-args.tests:]      # the end of the file, away from the 500 for the val loss
    print(f"\n2. {len(X):,} training conversations, 500 validation, {len(tests)} test requests"
          f" ({time.time() - t:.0f}s)")
    print("   " + tests[0][2].replace("\n", "\n   "))

    model = gpt.load(args.device, stories_gpt.CHECKPOINT)
    torch.manual_seed(0)
    print(f"\n3. Before SFT: val answer loss {sft.answer_loss(model, *val, args.device):.4f}")
    show_test("   Test", follow_test(model, tok, tests))

    sft.fine_tune(model, X, Y, args.device, STEPS=args.steps, val=val)
    gpt.save(model, CHECKPOINT)

    torch.manual_seed(0)
    print(f"\n4. After SFT: val answer loss {sft.answer_loss(model, *val, args.device):.4f}")
    show_test("   Test", follow_test(model, tok, tests))
    torch.manual_seed(0)
    p = prompt(request(["dragon", "soup", "happy"], ["Dialogue"]))
    print("\n   " + p.replace("\n", "\n   ")
          + gpt.generate(model, 300, p, 0.8, tok.encode, tok.decode, stop=EOT_ID))
    print(f"\nSaved to {CHECKPOINT}")
