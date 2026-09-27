"""SFT with system prompts: rules to follow, and who is who.

chat_sft.py's model was never trained with a system prompt. It only copies from one:
told "You are Zork.", it says "I am Zork." to "Who are you?", but it does that nearly
as often after "The user is Zork." And it ignores rules like "Answer in capitals."

Here some Dolly answers get a System: line with a rule, and the answer is rewritten to
follow it. Two rules are held out: close cousins of trained ones, to test whether the
model learns "follow the System line" or only the six rules it saw.

    System: Answer in capital letters.
    User: Why can camels survive for long without water?
    Assistant: CAMELS USE THE FAT IN THEIR HUMPS ...<|endoftext|>

Identity examples teach whose name is whose: "You are X." or "The user is X.", then
"Who are you?" or "Who am I?", answered from the right one, or "I don't know" when the
system line names only the other. The test names never appear in training.

    python system_sft.py            # ~10-15 min on the Mac, with the tests before and after
"""
import argparse
import random
import re
import statistics as st
import time
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

import chat_sft
import gpt
import sft
from web_data import TOKENISER
from word_bpe import WordBPE

BASE = Path("checkpoints/web_gpt_768w12l_anneal25.pt")
BEFORE = Path("checkpoints/web_gpt_768w12l_anneal25_chat_s250.pt")   # the chat model, to compare with

# ---------- rules ----------
def sentences(text):
    """Split text into sentences: after . ! or ? and a space, and at every line break
    (Dolly's lists have one item per line). Rough, but enough here."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]

# Each rule rewrites an answer to follow it (apply), and tests whether an answer
# follows it (check). apply(answer) must always pass check. Answers are Dolly's, so
# they can be one word, several paragraphs, or a list already.

def apply_one_sentence(answer):
    # TODO(Lilly): the first sentence of answer. sentences() gives them as a list.
    raise NotImplementedError

def check_one_sentence(answer):
    # TODO(Lilly): True if answer is exactly one sentence.
    raise NotImplementedError

def apply_capitals(answer):
    # TODO(Lilly): all in capital letters. Strings have a method for it.
    raise NotImplementedError

def check_capitals(answer):
    # TODO(Lilly): True if no letter is lowercase. Careful: "123" has no lowercase
    #   letters either. Is that a pass? (It's fine either way; just decide.)
    raise NotImplementedError

def apply_lowercase(answer):          # held out
    # TODO(Lilly): all in lowercase letters.
    raise NotImplementedError

def check_lowercase(answer):
    # TODO(Lilly): True if no letter is uppercase.
    raise NotImplementedError

def apply_start_with(word):
    """A rule's apply that starts the answer with word: "Well," or "Sure!"."""
    def apply(answer):
        # TODO(Lilly): word, a space, then the answer, stripped. Make the answer's first letter
        #   lowercase after "Well," ("Well, camels use..."), but keep it after "Sure!"
        #   ("Sure! Camels use..."). Hint: word.endswith("!").
        raise NotImplementedError
    return apply

def check_start_with(word):
    def check(answer):
        # TODO(Lilly): True if answer starts with word.
        raise NotImplementedError
    return check

def apply_question(answer):
    # TODO(Lilly): the answer, then " Does that help?" at the end. strip() it first,
    #   so the question doesn't come after a trailing newline.
    raise NotImplementedError

def check_question(answer):
    # TODO(Lilly): True if the answer ends with "?" (ignoring spaces at the end).
    raise NotImplementedError

def apply_list(answer):
    # TODO(Lilly): each sentence on its own line, after "- ". "\n".join(...) joins lines.
    raise NotImplementedError

def check_list(answer):
    # TODO(Lilly): True if every line starts with "- ". answer.splitlines() gives the
    #   lines. Skip empty lines, and an empty answer isn't a list.
    raise NotImplementedError

def apply_quotes(answer):
    # TODO(Lilly): the answer in double quotes: "...". strip() it first.
    raise NotImplementedError

def check_quotes(answer):
    # TODO(Lilly): True if it starts and ends with a double quote, and is longer than
    #   one character (a lone '"' does both).
    raise NotImplementedError

class Rule(NamedTuple):
    system: str
    apply: Callable[[str], str]
    check: Callable[[str], bool]

RULES = {
    "one sentence": Rule("Answer in one sentence.", apply_one_sentence, check_one_sentence),
    "capitals": Rule("Answer in capital letters.", apply_capitals, check_capitals),
    "well": Rule('Start your answer with "Well,".', apply_start_with("Well,"), check_start_with("Well,")),
    "question": Rule("End your answer with a question.", apply_question, check_question),
    "list": Rule("Answer as a bulleted list.", apply_list, check_list),
    "quotes": Rule("Put your answer in quotation marks.", apply_quotes, check_quotes),
    # Held out: never in training. Cousins of "capitals" and "well".
    "lowercase": Rule("Answer in lowercase letters.", apply_lowercase, check_lowercase),
    "sure": Rule('Start your answer with "Sure!".', apply_start_with("Sure!"), check_start_with("Sure!")),
}
TRAIN_RULES = ["one sentence", "capitals", "well", "question", "list", "quotes"]
TEST_RULES = ["lowercase", "sure"]

# ---------- identity ----------
TRAIN_NAMES = ["Ava", "Bruno", "Chen", "Dara", "Emeka", "Fiona", "Goran", "Hana", "Ivo", "Juno", "Kira",
               "Leon", "Mira", "Nils", "Olu", "Pia", "Quinn", "Rafa", "Sana", "Theo", "Uma", "Vik",
               "Wren", "Xia", "Yusuf", "Zara", "Nova-3", "Tiko-9", "Byte-12", "Orla-5"]
TEST_NAMES = ["LilyM-98M", "Zork", "Priya", "Ada-7", "Kofi", "Ingrid", "Mateo", "Pixel-4"]
YOU = ["You are {}.", "Your name is {}."]                        # names the assistant
USER = ["The user is {}.", "The user's name is {}."]            # names the user
WHO_YOU = ["Who are you?", "What is your name?"]
WHO_ME = ["Who am I?", "What is my name?"]

def identity_example(rng, names):
    """(system, question, answer): the name is the assistant's or the user's, and the
    question asks about one of them. Asked about the one not named: "I don't know"."""
    name = rng.choice(names)
    names_you = rng.random() < 0.5
    system = rng.choice(YOU if names_you else USER).format(name)
    asks_you = rng.random() < 0.5
    question = rng.choice(WHO_YOU if asks_you else WHO_ME)
    if asks_you and names_you:
        answer = rng.choice(["I am {}.", "My name is {}."]).format(name)
    elif not asks_you and not names_you:
        answer = rng.choice(["You are {}.", "Your name is {}."]).format(name)
    elif asks_you:
        answer = "I don't have a name."
    else:
        answer = "I don't know your name."
    return system, question, answer

# ---------- the data ----------
def examples(rows, seed=0, rule_share=0.5, identity_share=0.15, rules=TRAIN_RULES):
    """(system, question, context, answer) for every row: rule_share of them with a
    rule applied, the rest plain Dolly. Then identity examples, identity_share of the
    total. Shuffled."""
    rng = random.Random(seed)
    out = []
    for r in rows:
        answer = r["response"].strip()
        if rng.random() < rule_share:
            rule = RULES[rng.choice(rules)]
            out.append((rule.system, r["instruction"], r["context"], rule.apply(answer)))
        else:
            out.append(("", r["instruction"], r["context"], answer))
    n_identity = round(len(out) * identity_share / (1 - identity_share))
    out += [(s, q, "", a) for s, q, a in (identity_example(rng, TRAIN_NAMES) for _ in range(n_identity))]
    rng.shuffle(out)
    return out

def dataset(tok, exs, block=chat_sft.BLOCK):
    X, Y = [], []
    for system, q, c, a in exs:
        ex = sft.make_example(tok, chat_sft.conversation([], q, c, system), a, block)
        if ex:
            X.append(np.array(ex[0], dtype=np.int16))
            Y.append(np.array(ex[1], dtype=np.int16))
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))

# ---------- tests ----------
@torch.no_grad()
def answer(model, tok, question, system="", n=100, temperature=0.7):
    out = gpt.generate(model, n, chat_sft.conversation([], question, "", system), temperature,
                       tok.encode, tok.decode, stop=tok.eot_id)
    if next(model.parameters()).device.type == "mps":
        torch.mps.empty_cache()
    return out.strip()

def rule_test(model, tok, questions):
    """{rule: share of answers that follow it}, and "(none)": for each rule, the share of
    answers without a system prompt that follow it anyway."""
    torch.manual_seed(0)
    plain = [answer(model, tok, q) for q in questions]
    out = {}
    for name, rule in RULES.items():
        torch.manual_seed(0)
        got = [answer(model, tok, q, rule.system) for q in questions]
        out[name] = (st.mean(map(rule.check, got)), st.mean(map(rule.check, plain)), got[0])
    return out

def show_rule_test(title, result):
    print(f"{title}\n   {'rule':14s} {'with':>5s} {'without':>8s}   first answer")
    for name, (with_, without, first) in result.items():
        held = " (held out)" if name in TEST_RULES else ""
        print(f"   {name:14s} {with_:5.0%} {without:8.0%}   {first[:70]!r}{held}")

@torch.no_grad()
def logprob(model, tok, prompt, answer):
    p, a = tok.encode(prompt), tok.encode(answer)
    ids = torch.tensor([p + a], device=next(model.parameters()).device)
    logp = F.log_softmax(model(ids[:, :-1]).float(), dim=-1)[0]
    return sum(logp[len(p) - 1 + j, a[j]].item() for j in range(len(a)))

def binding_test(model, tok, names=TEST_NAMES):
    """log P of the name as the answer, mean over names, for each system line and question.
    A model that binds names to roles scores "You are X" high for "Who are you?", and
    "The user is X" high for "Who am I?". A copier scores both high for both."""
    rows = []
    for question, prefix in [("Who are you?", "I am"), ("Who am I?", "You are")]:
        for system in ["You are {}.", "The user is {}.", ""]:
            s = [logprob(model, tok, chat_sft.conversation([], question, "", system.format(n)) + prefix, " " + n)
                 for n in names]
            rows.append((question, system.format("X") or "(none)", st.mean(s)))
    return rows

def show_binding_test(title, rows):
    print(title)
    for question, system, score in rows:
        print(f"   {question:13s} System: {system:16s} {score:7.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    parser.add_argument("--base", type=Path, default=BASE, help="the model to fine-tune (default %(default)s)")
    parser.add_argument("--steps", type=int, default=400, help="fine-tuning steps (default 400)")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate (default 5e-5)")
    parser.add_argument("--accum", type=int, default=4,
                        help="micro-batches per step of 32 (default 4: 8 examples each, for the Mac's memory)")
    parser.add_argument("--tests", type=int, default=20, help="held-out questions for the rule test (default 20)")
    parser.add_argument("--tag", help="a label for the checkpoint name")
    args = parser.parse_args()
    out = args.base.with_name(f"{args.base.stem}_system" + (f"_{args.tag}" if args.tag else "") + ".pt")
    tok = WordBPE.load(TOKENISER)

    print("1. A Dolly answer under each rule:")
    sample = "Camels use the fat in their humps for energy. They can also drink a lot of water at once."
    for name, rule in RULES.items():
        got = rule.apply(sample)
        print(f"   {name:14s} {got!r}  follows it: {rule.check(got)}, plain follows it: {rule.check(sample)}")

    t = time.time()
    train, held_out = chat_sft.split(chat_sft.load_rows())
    X, Y = dataset(tok, examples(train))
    val = dataset(tok, examples(held_out, seed=1))
    questions = [r["instruction"] for r in held_out if not r["context"] and len(r["instruction"]) < 120][:args.tests]
    print(f"\n2. {len(X):,} training examples, {len(val[0])} validation ({time.time() - t:.0f}s)")

    before = gpt.load(args.device, BEFORE)
    show_rule_test(f"\n3. Before: {BEFORE.name} (the chat model)", rule_test(before, tok, questions))
    show_binding_test("\n   Binding, log P of the name:", binding_test(before, tok))
    del before
    if args.device == "mps":
        torch.mps.empty_cache()

    model = gpt.load(args.device, args.base)
    sft.fine_tune(model, X, Y, args.device, STEPS=args.steps, LR=args.lr, val=val, accum=args.accum)
    gpt.save(model, out)
    show_rule_test(f"\n4. After: {out.name}", rule_test(model, tok, questions))
    show_binding_test("\n   Binding, log P of the name:", binding_test(model, tok))
    print(f"\nSaved to {out}")
