"""system_sft.py: the rules, the identity examples and the mix.

The rule tests need Lilly's apply and check functions; until then they fail with
NotImplementedError.
"""
import random

import pytest

import sft
import system_sft as ss
from word_bpe import WordBPE

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
    monkeypatch.setitem(ss.RULES, "shout", ss.Rule("Shout.", lambda a: a + "!!", lambda a: a.endswith("!!")))
    rows = [{"instruction": f"Q{i}?", "context": "", "response": f"A{i}."} for i in range(200)]
    exs = ss.examples(rows, rules=["shout"])
    ruled = [e for e in exs if e[0] == "Shout."]
    identity = [e for e in exs if e[0] and e[0] != "Shout."]
    assert len(exs) == 200 + len(identity)
    assert 0.35 < len(ruled) / 200 < 0.65 and all(a.endswith("!!") for *_, a in ruled)
    assert abs(len(identity) / len(exs) - 0.15) < 0.01
    assert ss.examples(rows, rules=["shout"]) == exs                  # the same every time

def test_dataset_learns_only_the_answer():
    tok = WordBPE.train(["System: Shout. User: Q? Assistant: A!!"] * 5, 270)
    X, Y = ss.dataset(tok, [("Shout.", "Q?", "", "A!!")], block=64)
    assert [t for t in Y[0].tolist() if t != sft.IGNORE] == tok.encode("A!!") + [tok.eot_id]
