"""SFT with system prompts: rules to follow, and who is who.

chat_sft.py's model was never trained with a system prompt. It only copies from one:
told "You are Zork.", it says "I am Zork." to "Who are you?", but it does that nearly
as often after "The user is Zork." And it ignores rules like "Answer in capitals."

Here some Dolly answers get a System: line with a rule, and the answer is rewritten to
follow it. Each rule has several wordings; the last is never trained, to test whether
the model follows the meaning or the words. Some rules are never trained at all.

    System: Answer in capital letters.
    User: Why can camels survive for long without water?
    Assistant: CAMELS USE THE FAT IN THEIR HUMPS ...<|endoftext|>

Identity examples teach whose name is whose: "You are X." or "The user is X.", then
"Who are you?" or "Who am I?", answered from the right one, or "I don't know" when the
system line names only the other. The test names never appear in training.

Version 2 (--tag v2). Version 1 learned that a name in the System line means "talk
about the name": every name it saw came with an identity question. Now names also come
before ordinary questions (answered as usual), and with rules. Capitals is held out:
the tokeniser splits capital letters into rare pieces ("ENERGY" is E|N|ER|G|Y), so
capitals tests spelling more than following the System line.

    python -m web.system_sft --tag v2                  # ~45 min training on the Mac, then the tests
    python -m web.system_sft --tag v2 --before checkpoints/web/web_gpt_768w12l_anneal25_system.pt
"""
import argparse
import itertools
from functools import reduce
import random
import re
import statistics as st
import time
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from web import chat_sft
from core import gpt
from stories import sft
from web.web_data import TOKENISER
from core.word_bpe import WordBPE

BASE = Path("checkpoints/web/web_gpt_768w12l_anneal25.pt")

# ---------- rules ----------
def sentences(text: str):
    """Split text into sentences: after . ! or ? and a space, and at every line break
    (Dolly's lists have one item per line). Rough, but enough here."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]

# Each rule rewrites an answer to follow it (apply), and tests whether an answer
# follows it (check). apply(answer) must always pass check. Answers are Dolly's, so
# they can be one word, several paragraphs, or a list already.

def apply_one_sentence(answer: str):    
    return sentences(answer)[0]

def check_one_sentence(answer: str):
    return len(sentences(answer)) == 1

def apply_capitals(answer: str):
    return answer.upper()

def check_capitals(answer : str):
    return answer.upper() == answer

def apply_lowercase(answer: str):
    return answer.lower()

def check_lowercase(answer: str):
    return answer.lower() == answer 

def apply_start_with(word):
    """A rule's apply that starts the answer with word: "Well," or "Sure!"."""
    def apply(answer):
        answer = answer.strip()
        if not word.endswith("!"):
            answer = answer[:1].lower() + answer[1:]
        return f"{word} {answer}"
    return apply

def check_start_with(word):
    def check(answer):
        return answer.startswith(word)
    return check

def apply_question(answer):
    return answer.strip() + " Does that help?"

def check_question(answer: str):
    return answer.rstrip()[-1:] == "?"

def apply_list(answer: str):
    return "\n".join("- " + s for s in sentences(answer))

def check_list(answer: str):
    lines = [s for s in answer.splitlines() if s.strip()]
    return len(lines) > 0 and all(s.startswith("- ") for s in lines)

def apply_quotes(answer: str):
    return '"' + answer.strip() + '"'

def check_quotes(answer: str):
    return len(answer) > 1 and answer.startswith('"') and answer.endswith('"') 

def apply_exclaim(answer: str):
    return " ".join(re.sub(r"[.!?]*$", "", s) + "!" for s in sentences(answer))

def check_exclaim(answer: str):
    ss = sentences(answer)
    return len(ss) > 0 and all(s.endswith("!") for s in ss)

def apply_numbered(answer: str):
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences(answer), 1))

def check_numbered(answer: str):
    lines = [s for s in answer.splitlines() if s.strip()]
    return len(lines) > 0 and all(s.startswith(f"{i}. ") for i, s in enumerate(lines, 1))

def apply_end_with(words):
    """A rule's apply that ends the answer with words: "Hope this helps!" or "Thanks for asking."."""
    def apply(answer):
        return f"{answer.strip()} {words}"
    return apply

def check_end_with(words):
    def check(answer):
        return answer.rstrip().endswith(words)
    return check

class Rule(NamedTuple):
    systems: tuple      # ways to say the rule. Training uses all but the last.
    apply: Callable[[str], str]
    check: Callable[[str], bool]

RULES = {
    "one sentence": Rule(("Answer in one sentence.", "Keep your answer to a single sentence.",
                          "Reply with just one sentence."), apply_one_sentence, check_one_sentence),
    "well": Rule(('Start your answer with "Well,".', 'Begin your reply with the word "Well,".',
                  'Your first word must be "Well,".'), apply_start_with("Well,"), check_start_with("Well,")),
    "in short": Rule(('Start your answer with "In short,".', 'Begin your reply with "In short,".',
                      'Your first words must be "In short,".'),
                     apply_start_with("In short,"), check_start_with("In short,")),
    "question": Rule(("End your answer with a question.", "Finish your reply by asking a question.",
                      "Close your answer with a question."), apply_question, check_question),
    "list": Rule(("Answer as a bulleted list.", "Use bullet points.",
                  "Format your answer as a list of bullet points."), apply_list, check_list),
    "numbered": Rule(("Answer as a numbered list.", "Number each point of your answer.",
                      "Format your answer as a numbered list."), apply_numbered, check_numbered),
    "quotes": Rule(("Put your answer in quotation marks.", "Wrap your reply in double quotes.",
                    "Your answer must be inside quotation marks."), apply_quotes, check_quotes),
    "exclaim": Rule(("End every sentence with an exclamation mark.", "Finish each sentence with an exclamation mark.",
                     "Every sentence must end with an exclamation mark."), apply_exclaim, check_exclaim),
    "hope": Rule(('End your answer with "Hope this helps!".', 'Sign off with "Hope this helps!".',
                  'Your last words must be "Hope this helps!".'),
                 apply_end_with("Hope this helps!"), check_end_with("Hope this helps!")),
    # Held out: never in training. Cousins of trained rules, and capitals (hard: see above).
    "sure": Rule(('Start your answer with "Sure!".', 'Your first word must be "Sure!".'),
                 apply_start_with("Sure!"), check_start_with("Sure!")),
    "thanks": Rule(('End your answer with "Thanks for asking.".', 'Your last words must be "Thanks for asking.".'),
                   apply_end_with("Thanks for asking."), check_end_with("Thanks for asking.")),
    "lowercase": Rule(("Answer in lowercase letters.", "Please reply using only lowercase letters."),
                      apply_lowercase, check_lowercase),
    "capitals": Rule(("Answer in capital letters.", "Please reply using only capital letters."),
                     apply_capitals, check_capitals),
}
TRAIN_RULES = ["one sentence", "well", "in short", "question", "list", "numbered", "quotes", "exclaim", "hope"]
TEST_RULES = ["sure", "thanks", "lowercase", "capitals"]

# ---------- identity ----------
TRAIN_NAMES = ["Ava", "Bruno", "Chen", "Dara", "Emeka", "Fiona", "Goran", "Hana", "Ivo", "Juno", "Kira",
               "Leon", "Mira", "Nils", "Olu", "Pia", "Quinn", "Rafa", "Sana", "Theo", "Uma", "Vik",
               "Wren", "Xia", "Yusuf", "Zara", "Nova-3", "Tiko-9", "Byte-12", "Orla-5"]
TEST_NAMES = ["LilyM-98M", "Zork", "Priya", "Ada-7", "Kofi", "Ingrid", "Mateo", "Pixel-4"]
YOU = ["You are {}.", "Your name is {}."]                        # names the assistant
USER = ["The user is {}.", "The user's name is {}."]            # names the user
WHO_YOU = ["Who are you?", "What is your name?"]
WHO_ME = ["Who am I?", "What is my name?"]

def identity_line(rng, names):
    """A System line that names the assistant or the user."""
    return rng.choice(YOU + USER).format(rng.choice(names))

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
def examples(rows, seed=0, rule_share=0.5, identity_share=0.15, rules=TRAIN_RULES,
             name_share=0.3, name_with_rule=0.2, rule_on_identity=0.3):
    """(system, question, context, answer) for every row: rule_share of them with a
    rule applied, the rest plain Dolly. Then identity examples, identity_share of the
    total. Shuffled.

    So that a name in the System line is only background unless asked about:
    name_share of the plain rows get an identity line (the answer doesn't change),
    name_with_rule of the rule rows get one before the rule, and rule_on_identity of
    the identity examples get a rule too. Rules use every wording but the last.
    """
    rng = random.Random(seed)
    out = []
    def pick_rule():
        rule = RULES[rng.choice(rules)]
        return rule, rng.choice(rule.systems[:-1])
    for r in rows:
        answer = r["response"].strip()
        if rng.random() < rule_share:
            rule, system = pick_rule()
            if rng.random() < name_with_rule:
                system = f"{identity_line(rng, TRAIN_NAMES)} {system}"
            out.append((system, r["instruction"], r["context"], rule.apply(answer)))
        else:
            system = identity_line(rng, TRAIN_NAMES) if rng.random() < name_share else ""
            out.append((system, r["instruction"], r["context"], answer))
    n_identity = round(len(out) * identity_share / (1 - identity_share))
    for _ in range(n_identity):
        system, question, answer = identity_example(rng, TRAIN_NAMES)
        if rng.random() < rule_on_identity:
            rule, rule_system = pick_rule()
            system, answer = f"{system} {rule_system}", rule.apply(answer)
        out.append((system, question, "", answer))
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
    """{rule: (share following it with its first wording, with its last wording, without a
    system line, the first answer)}. For a trained rule, the first wording was trained
    and the last never was; a held-out rule was never trained in any wording."""
    torch.manual_seed(0)
    plain = [answer(model, tok, q) for q in questions]
    out = {}
    for name, rule in RULES.items():
        shares, first = [], None
        for system in (rule.systems[0], rule.systems[-1]):
            torch.manual_seed(0)
            got = [answer(model, tok, q, system) for q in questions]
            shares.append(st.mean(map(rule.check, got)))
            first = first if first is not None else got[0]
        out[name] = (*shares, st.mean(map(rule.check, plain)), first)
    return out

def show_rule_test(title, result):
    print(f"{title}\n   {'rule':13s} {'trained':>8s} {'new':>8s} {'none':>5s}   first answer")
    print(f"   {'':13s} {'wording':>8s} {'wording':>8s}")
    for name, (seen, new, without, first) in result.items():
        held = " (held out)" if name in TEST_RULES else ""
        print(f"   {name:13s} {seen:8.0%} {new:8.0%} {without:5.0%}   {first[:60]!r}{held}")

def leak_test(model, tok, questions, names=TEST_NAMES):
    """Share of answers to ordinary questions that bring up the name (or start with
    "I am"), under "You are X." and "The user is X.", and with no system line."""
    def leaks(a, name):
        return name.lower() in a.lower() or a.startswith("I am")
    out = {}
    for system in ["You are {}.", "The user is {}.", ""]:
        torch.manual_seed(0)
        got = [(answer(model, tok, q, system.format(n)), n) for q, n in zip(questions, itertools.cycle(names))]
        out[system.format("X") or "(none)"] = (st.mean(leaks(a, n) for a, n in got), got[0][0])
    return out

def show_leak_test(title, result):
    print(title)
    for system, (share, first) in result.items():
        print(f"   System: {system:16s} {share:5.0%}   {first[:60]!r}")

@torch.no_grad()
def logprob(model, tok, prompt, answer):
    p, a = tok.encode(prompt), tok.encode(answer)
    ids = torch.tensor([p + a], device=next(model.parameters()).device)
    logp = F.log_softmax(model(ids[:, :-1]).float(), dim=-1)[0]
    return sum(logp[len(p) - 1 + j, a[j]].item() for j in range(len(a)))

BINDING_CASES = [   # (system line, question, the answer with the name, the answer without)
    ("You are {}.", "Who are you?", "I am {}.", "I don't have a name."),
    ("The user is {}.", "Who are you?", "I am {}.", "I don't have a name."),
    ("You are {}.", "Who am I?", "You are {}.", "I don't know your name."),
    ("The user is {}.", "Who am I?", "You are {}.", "I don't know your name."),
]

def binding_test(model, tok, names=TEST_NAMES):
    """For each system line and question: log P(the answer that gives the name) minus
    log P(the "don't know" answer), as (system, question, name is right?, mean, std error)
    over names. Above 0: the model gives the name.

    A copier gives the name in all four cases. A model that binds names to roles gives
    it only when the name belongs to the one asked about. Whole answers, because the
    choice is made at their first token: once "I am" is given, the name is the only
    sensible next word, whatever the model believes.
    """
    rows = []
    for system, question, named, unknown in BINDING_CASES:
        d = []
        for n in names:
            p = chat_sft.conversation([], question, "", system.format(n))
            d.append(logprob(model, tok, p, named.format(n)) - logprob(model, tok, p, unknown))
        right = system.startswith("You") == (question in WHO_YOU)
        rows.append((system.format("X"), question, right, st.mean(d), st.stdev(d) / len(d) ** 0.5))
    return rows

def show_binding_test(title, rows):
    print(title)
    for system, question, right, mean, se in rows:
        print(f"   System: {system:16s} {question:13s} {mean:+7.2f} +- {se:.2f}   (the name is {'right' if right else 'wrong'})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    parser.add_argument("--base", type=Path, default=BASE, help="the model to fine-tune (default %(default)s)")
    parser.add_argument("--steps", type=int, default=300, help="fine-tuning steps (default 300)")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate (default 5e-5)")
    parser.add_argument("--accum", type=int, default=4,
                        help="micro-batches per step of 32 (default 4: 8 examples each, for the Mac's memory)")
    parser.add_argument("--tests", type=int, default=20, help="held-out questions for the tests (default 20)")
    parser.add_argument("--before", type=Path, help="also test this model first, to compare (slow: ~15 min on the Mac)")
    parser.add_argument("--tag", help="a label for the checkpoint name, e.g. v2")
    args = parser.parse_args()
    out = args.base.with_name(f"{args.base.stem}_system" + (f"_{args.tag}" if args.tag else "") + ".pt")
    tok = WordBPE.load(TOKENISER)

    print("1. A Dolly answer under each rule:")
    sample = "Camels use the fat in their humps for energy. They can also drink a lot of water at once."
    for name, rule in RULES.items():
        got = rule.apply(sample)
        print(f"   {name:13s} {got!r}  follows it: {rule.check(got)}, plain follows it: {rule.check(sample)}")

    t = time.time()
    train, held_out = chat_sft.split(chat_sft.load_rows())
    X, Y = dataset(tok, examples(train))
    val = dataset(tok, examples(held_out, seed=1))
    questions = [r["instruction"] for r in held_out if not r["context"] and len(r["instruction"]) < 120][:args.tests]
    print(f"\n2. {len(X):,} training examples, {len(val[0])} validation ({time.time() - t:.0f}s)")

    def run_tests(title, m):
        show_rule_test(title, rule_test(m, tok, questions))
        show_leak_test("\n   Names leaking into ordinary answers:", leak_test(m, tok, questions))
        show_binding_test("\n   Binding, log P(answer with the name) - log P(\"don't know\"):", binding_test(m, tok))

    if args.before:
        before = gpt.load(args.device, args.before)
        run_tests(f"\n3. Before: {args.before.name}", before)
        del before
        if args.device == "mps":
            torch.mps.empty_cache()

    model = gpt.load(args.device, args.base)
    sft.fine_tune(model, X, Y, args.device, STEPS=args.steps, LR=args.lr, val=val, accum=args.accum)
    gpt.save(model, out)
    run_tests(f"\n4. After: {out.name}", model)
    print(f"\nSaved to {out}")
