"""tools/knowledge.py: the probe's scoring, on a tiny model with its own tokeniser."""
import torch

from core import gpt
from core.word_bpe import WordBPE
from tools import knowledge as kn

def test_people_have_one_right_option_per_level():
    for level, (template, neutral, options, field) in kn.LEVELS.items():
        assert "{}" in template and "{}" not in neutral
        assert all(p[field] in options for p in kn.PEOPLE if p[field])
        assert len(set(options)) == len(options)

def test_probe_scores_and_calibrates():
    people = [("Ada", "scientist", "England"), ("Bo", "painter", "France")]
    levels = {"category": ("{} was a famous", "This person was a famous", ["scientist", "painter"], 1)}
    tok = WordBPE.train(["Ada was a famous scientist. Bo was a famous painter. This person was"] * 3, 290)
    torch.manual_seed(0)
    model = gpt.GPT(block=32, emb=16, heads=2, layers=1, dropout=0.0, vocab=tok.eot_id + 1, gelu=True, tied=True).eval()
    r = kn.probe(model, tok, people, levels)["category"]
    assert 0 <= r["raw"] <= 1 and 0 <= r["calibrated"] <= 1 and r["chance"] == 0.5
    assert set(r["answers"]) == {"Ada", "Bo"}

def test_logprob_is_a_sum_over_the_answer_tokens():
    tok = WordBPE.train(["the cat sat on the mat"] * 5, 262)
    torch.manual_seed(0)
    model = gpt.GPT(block=32, emb=16, heads=2, layers=1, dropout=0.0, vocab=tok.eot_id + 1, gelu=True, tied=True).eval()
    whole = kn.logprob(model, tok, "the", " cat sat")
    assert whole < kn.logprob(model, tok, "the", " cat") < 0
