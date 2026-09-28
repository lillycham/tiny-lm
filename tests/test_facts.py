"""tools/facts.py: the fact tests, on a tiny model with its own tokeniser."""
import pytest
import torch

from core import gpt
from core.word_bpe import WordBPE
from tools import facts
from tools import knowledge as kn

CAPITALS = [("Oslo", "Norway"), ("Lima", "Peru"), ("Rome", "Italy")]

@pytest.fixture(scope="module")
def tiny():
    text = ("The city of Oslo is the capital of Norway. Lima is the capital of Peru. "
            "The capital of Italy is Rome. Ada was born in Peru and lived in Norway.")
    tok = WordBPE.train([text] * 3, 300)
    torch.manual_seed(0)
    model = gpt.GPT(block=32, emb=16, heads=2, layers=1, dropout=0.0, vocab=tok.eot_id + 1, gelu=True, tied=True).eval()
    return model, tok

def test_the_lists_are_consistent():
    countries = [c for _, c in facts.CAPITALS]
    assert len(set(countries)) == len(countries)
    assert not set(facts.MADE_UP) & {c for c, _ in facts.CAPITALS}
    assert all(born != linked for _, born, linked in facts.PAIRS)
    assert all("{}" in t for t in facts.FRAMES.values())

def test_spaced_adds_one_space_and_keeps_the_rest():
    for (k, a), (_, b) in zip(kn.LEVELS.items(), facts.spaced().items()):
        assert b[0] == " " + a[0] and b[1] == " " + a[1] and b[2:] == a[2:]

def test_pieces_shows_the_split(tiny):
    _, tok = tiny
    assert facts.pieces(tok, "Oslo").replace("|", "") == "Oslo"

def test_rank_puts_the_best_option_first(tiny):
    model, tok = tiny
    options = ["Norway", "Peru", "Italy"]
    s = facts.scores(model, tok, "Oslo is the capital of", options)
    best = max(s, key=s.get)
    r, top, lead = facts.rank(model, tok, "Oslo is the capital of", best, options)
    assert r == 1 and top == best and lead >= 0
    worst = min(s, key=s.get)
    assert facts.rank(model, tok, "Oslo is the capital of", worst, options)[0] == 3

def test_the_tests_return_one_result_per_item(tiny):
    model, tok = tiny
    ranks = facts.framings(model, tok, CAPITALS)
    assert set(ranks) == set(facts.FRAMES)
    assert all(set(r) == {"Oslo", "Lima", "Rome"} and all(1 <= v <= 3 for v in r.values()) for r in ranks.values())
    rev = facts.reverse(model, tok, CAPITALS)
    assert set(rev) == {"Norway", "Peru", "Italy"} and all(t in {"Oslo", "Lima", "Rome"} for _, t in rev.values())
    made = facts.made_up(model, tok, ["Varnholm"], CAPITALS)
    assert made["Varnholm"][0] in {"Norway", "Peru", "Italy"} and made["Varnholm"][1] >= 0
    diffs = facts.born_vs_linked(model, tok, [("Ada", "Peru", "Norway")], ["was born in", "lived in"])
    assert set(diffs) == {("Ada", "was born in"), ("Ada", "lived in")}
