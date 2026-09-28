"""system_sft.py: the rules, the identity examples and the mix.

The rule tests need Lilly's apply and check functions; until then they fail with
NotImplementedError.
"""
import random

import pytest

from stories import sft
from web import system_sft as ss
from core.word_bpe import WordBPE

ANSWERS = ["Camels use the fat in their humps for energy. They can also drink a lot of water at once.",
           "Tope",
           "1. Go to a museum\n2. Bake bread\n3. Read a book",
           "The capital is Paris! It is on the Seine.\n\nIt has about 2 million people.",
           "  Water boils at 100 °C at sea level.  "]

def test_sentences():
    assert ss.sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]
    assert ss.sentences("1. Go out\n2. Bake\n\nEnd") == ["1.", "Go out", "2.", "Bake", "End"]
    assert ss.sentences("  ") == []

@pytest.mark.parametrize("name", list(ss.RULES))
@pytest.mark.parametrize("answer", ANSWERS)
def test_every_rewritten_answer_follows_its_rule(name, answer):
    rule = ss.RULES[name]
    assert rule.check(rule.apply(answer)), rule.apply(answer)

@pytest.mark.parametrize("name", list(ss.RULES))
def test_a_plain_answer_does_not_follow_the_rule(name):
    """The first answer breaks every rule, so a check that always says True fails here."""
    assert not ss.RULES[name].check(ANSWERS[0])

def test_rule_examples():
    r = ss.RULES
    assert r["one sentence"].apply(ANSWERS[0]) == "Camels use the fat in their humps for energy."
    assert r["well"].apply("Camels use fat.") == "Well, camels use fat."
    assert r["sure"].apply("Camels use fat.") == "Sure! Camels use fat."
    assert r["question"].apply("Tope\n") == "Tope Does that help?"
    assert r["list"].apply("One. Two.") == "- One.\n- Two."
    assert r["quotes"].apply(" Tope ") == '"Tope"'
    assert not r["list"].check("")
    assert not r["quotes"].check('"')

def test_held_out_rules_are_never_trained():
    assert not set(ss.TRAIN_RULES) & set(ss.TEST_RULES)
    assert set(ss.TRAIN_RULES) | set(ss.TEST_RULES) == set(ss.RULES)
    assert not set(ss.TRAIN_NAMES) & set(ss.TEST_NAMES)

def test_identity_answers_follow_the_role():
    rng = random.Random(0)
    seen = set()
    for _ in range(400):
        system, question, answer = ss.identity_example(rng, ["Zork"])
        names_you, asks_you = system.startswith(("You", "Your")), question in ss.WHO_YOU
        assert ("Zork" in answer) == (names_you == asks_you)
        seen.add((names_you, asks_you))
    assert len(seen) == 4                       # all four cases come up

def test_examples_mix(monkeypatch):
    monkeypatch.setitem(ss.RULES, "shout", ss.Rule(("Shout.", "Yell."), lambda a: a + "!!", lambda a: a.endswith("!!")))
    rows = [{"instruction": f"Q{i}?", "context": "", "response": f"A{i}."} for i in range(400)]
    exs = ss.examples(rows, rules=["shout"])
    who = set(ss.WHO_YOU + ss.WHO_ME)
    identity = [e for e in exs if e[1] in who]
    dolly = [e for e in exs if e[1] not in who]
    ruled = [e for e in dolly if e[3].endswith("!!")]
    plain = [e for e in dolly if not e[3].endswith("!!")]
    assert len(dolly) == 400 and abs(len(identity) / len(exs) - 0.15) < 0.01
    assert 0.4 < len(ruled) / 400 < 0.6
    assert all(e[0].endswith("Shout.") for e in ruled)                  # the trained wording only
    assert not any("Yell." in e[0] for e in exs)                        # the last wording is held out
    named_plain = [e for e in plain if e[0]]
    assert 0.2 < len(named_plain) / len(plain) < 0.4                    # names before ordinary questions
    assert all(e[3] == e[1].replace("Q", "A").replace("?", ".") for e in plain)   # answered as usual
    assert any(e[0] != "Shout." for e in ruled)                         # names with rules too
    assert any(e[0].endswith("Shout.") for e in identity)               # and rules on identity answers
    assert ss.examples(rows, rules=["shout"]) == exs                    # the same every time

def test_every_rule_has_a_held_out_wording():
    for name, rule in ss.RULES.items():
        assert len(rule.systems) >= (3 if name in ss.TRAIN_RULES else 2), name

def test_new_rule_examples():
    r = ss.RULES
    assert r["exclaim"].apply("One. Two? Three") == "One! Two! Three!"
    assert r["numbered"].apply("One. Two.") == "1. One.\n2. Two."
    assert not r["numbered"].check("2. One.\n1. Two.")
    assert r["hope"].apply("Tope \n") == "Tope Hope this helps!"
    assert r["in short"].apply("Camels use fat.") == "In short, camels use fat."

def test_dataset_learns_only_the_answer():
    tok = WordBPE.train(["System: Shout. User: Q? Assistant: A!!"] * 5, 270)
    X, Y = ss.dataset(tok, [("Shout.", "Q?", "", "A!!")], block=64)
    assert [t for t in Y[0].tolist() if t != sft.IGNORE] == tok.encode("A!!") + [tok.eot_id]

def test_binding_test_compares_whole_answers():
    import torch
    from core import gpt
    torch.manual_seed(0)
    tok = WordBPE.train(["System: You are Zork. User: Who are you? Assistant: I am Zork. I don't know."] * 3, 280)
    model = gpt.GPT(block=64, emb=32, heads=2, layers=1, dropout=0.0, vocab=tok.eot_id + 1, gelu=True, tied=True).eval()
    rows = ss.binding_test(model, tok, names=["Zork", "Ada"])
    assert [r[2] for r in rows] == [True, False, False, True]          # which cases name the right one
    assert all(isinstance(r[3], float) and r[4] >= 0 for r in rows)
