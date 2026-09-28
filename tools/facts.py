"""Fact tests: how a prompt's form changes what a web model seems to know.

Follow-ups to the knowledge probe (tools/knowledge.py), with the same scoring
(log-probability after <|endoftext|>). Five tests:
  A. The probe with the name first ("Albert") and after a space (" Albert"). A word
     at the very start of a text has no space before it, so the tokeniser often splits
     it into rare pieces: "Albert" is Al|bert, " Albert" is one token.
  B. City -> country, three framings: "Oslo is the capital of", " Oslo is the capital
     of", "The city of Oslo is the capital of".
  C. The other direction: "The capital of Norway is" -> Oslo?
  D. Made-up cities: how far the top country leads when there is no fact to find.
  E. Born in, lived in, died in: does the verb pick the right country, or does the
     country a person is linked with win (Marie Curie -> France, not Poland)?

    python -m tools.facts checkpoints/web/web_gpt_768w12l_wikifull3.pt
"""
import argparse
from pathlib import Path

import torch

from core import gpt
from core.word_bpe import WordBPE
from tools.knowledge import LEVELS, logprob, probe
from web.web_data import TOKENISER

# From well known to less known
CAPITALS = [("London", "the United Kingdom"), ("Paris", "France"), ("Berlin", "Germany"), ("Rome", "Italy"),
            ("Madrid", "Spain"), ("Tokyo", "Japan"), ("Moscow", "Russia"), ("Beijing", "China"),
            ("Cairo", "Egypt"), ("Athens", "Greece"), ("Ottawa", "Canada"), ("Canberra", "Australia"),
            ("Warsaw", "Poland"), ("Lisbon", "Portugal"), ("Oslo", "Norway"), ("Nairobi", "Kenya"),
            ("Lima", "Peru"), ("Hanoi", "Vietnam"), ("Wellington", "New Zealand"), ("Ankara", "Turkey")]
FRAMES = {"start": "{} is the capital of", "space": " {} is the capital of",
          "city of": "The city of {} is the capital of"}
REVERSE = "The capital of {} is"
MADE_UP = ["Varnholm", "Teskaro", "Pellinbury", "Ostravik"]
# (name, country of birth, a different country the person is linked with)
PAIRS = [("Marie Curie", "Poland", "France"), ("Frédéric Chopin", "Poland", "France"),
         ("Joseph Stalin", "Georgia", "Russia"), ("Adolf Hitler", "Austria", "Germany"),
         ("Albert Einstein", "Germany", "the United States")]
VERBS = ["was born in", "lived in", "died in"]

def pieces(tok, text):
    """How the tokeniser splits text, e.g. "Al|bert"."""
    return "|".join(tok.decode([i]) for i in tok.encode(text))

def scores(model, tok, prompt, options):
    return {o: logprob(model, tok, prompt, " " + o) for o in options}

def rank(model, tok, prompt, right, options):
    """(rank of the right option, 1 = best; the top option; its lead over the second)."""
    s = scores(model, tok, prompt, options)
    order = sorted(s, key=s.get, reverse=True)
    return order.index(right) + 1, order[0], s[order[0]] - s[order[1]]

def spaced(levels=LEVELS):
    """The probe's levels with a space before each prompt, so names are whole tokens."""
    return {k: (" " + t, " " + n, o, f) for k, (t, n, o, f) in levels.items()}

def framings(model, tok, capitals=CAPITALS, frames=FRAMES):
    """{frame: {city: rank of its country}}."""
    countries = [c for _, c in capitals]
    return {f: {city: rank(model, tok, t.format(city), country, countries)[0] for city, country in capitals}
            for f, t in frames.items()}

def reverse(model, tok, capitals=CAPITALS):
    """{country: (rank of its capital, the top city)}."""
    cities = [c for c, _ in capitals]
    return {country: rank(model, tok, REVERSE.format(country), city, cities)[:2] for city, country in capitals}

def made_up(model, tok, cities=MADE_UP + ["Paris"], capitals=CAPITALS):
    """{city: (top country, its lead over the second)}. Paris is the real-fact comparison."""
    countries = [c for _, c in capitals]
    out = {}
    for city in cities:
        s = scores(model, tok, FRAMES["city of"].format(city), countries)
        top, second = sorted(s.values(), reverse=True)[:2]
        out[city] = (max(s, key=s.get), top - second)
    return out

def born_vs_linked(model, tok, pairs=PAIRS, verbs=VERBS):
    """{(name, verb): log P(birth country) - log P(linked country)}; above 0 picks the birth country."""
    out = {}
    for name, born, linked in pairs:
        for verb in verbs:
            s = scores(model, tok, f" {name} {verb}", [born, linked])
            out[name, verb] = s[born] - s[linked]
    return out

def show(model, tok):
    print("\n   A. Knowledge probe, calibrated share right")
    for label, levels in (("name first   ", LEVELS), ("leading space", spaced())):
        r = probe(model, tok, levels=levels)
        print(f"      {label}  " + "   ".join(f"{k} {v['calibrated']:4.0%}" for k, v in r.items()))

    ranks = framings(model, tok)
    print(f"\n   B. City -> country: rank of the right country of {len(CAPITALS)} (1 = best)")
    print(f"      {'city':11s} {'tokens at start':16s}" + "".join(f"{f:>9s}" for f in FRAMES))
    for city, _ in CAPITALS:
        print(f"      {city:11s} {pieces(tok, city):16s}" + "".join(f"{ranks[f][city]:9d}" for f in FRAMES))
    print(f"      {'right':28s}" + "".join(f"{sum(r == 1 for r in ranks[f].values()):>6d}/{len(CAPITALS)}"
                                           for f in FRAMES))

    rev = reverse(model, tok)
    misses = [f"{country} -> {top} (rank {r})" for country, (r, top) in rev.items() if r > 1]
    print(f"\n   C. Country -> city: {sum(r == 1 for r, _ in rev.values())}/{len(CAPITALS)} right"
          + (f"; missed {', '.join(misses)}" if misses else ""))

    print("\n   D. Made-up cities: the top country and its lead over the second (Paris is real)")
    for city, (top, lead) in made_up(model, tok).items():
        print(f"      {city:11s} {top:20s} lead {lead:.2f}")

    print("\n   E. log P(birth country) - log P(linked country); above 0 picks the birth country")
    print(f"      {'':48s}" + "".join(f"{v:>14s}" for v in VERBS))
    diffs = born_vs_linked(model, tok)
    for name, born, linked in PAIRS:
        print(f"      {name + f' ({born} vs {linked})':48s}" + "".join(f"{diffs[name, v]:+14.2f}" for v in VERBS))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="web-tokeniser models to test")
    args = parser.parse_args()
    tok = WordBPE.load(TOKENISER)
    for path in args.checkpoints:
        print(f"\n{path.name}")
        model = gpt.load(args.device, path)
        show(model, tok)
        del model
        if args.device == "mps":
            torch.mps.empty_cache()
