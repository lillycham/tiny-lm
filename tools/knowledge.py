"""A knowledge probe: what does a web model know about famous people?

Three levels, each a choice between fixed options, scored by log-probability:
  1. Category:   "Albert Einstein was a famous" -> " scientist"? (of 8 categories)
  2. Birthplace: "Albert Einstein was born in"  -> " Germany"?   (of 16 countries)
  3. Birth year: "Albert Einstein was born in the year" -> " 1879"? (of 25 years)

Associations (Einstein, physics) appear together thousands of times in the training
text. A birth country turned out to be one too: wrong answers are the country a person
is linked with (Marie Curie -> France, where she worked). Some are traps for that:
Stalin was born in Georgia, not Russia, and Hitler in Austria, not Germany. A birth
year is a real specific fact: nothing else about the person hints at it. Only people
with one certain birth year have one (not Newton: 1642 or 1643, by the calendar).

Some words are likely anywhere ("England" more than "Georgia"). So each option is also
scored against the same sentence with no name ("This person was born in"), and the
calibrated score is the difference (pointwise mutual information): how much more
likely the name makes the option.

    python -m tools.knowledge checkpoints/web/web_gpt_768w12l.pt checkpoints/web/web_gpt_768w12l_anneal25.pt
"""
import argparse
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F

from core import gpt
from core.word_bpe import WordBPE
from web.web_data import TOKENISER

# (name, category, country of birth as it is today, birth year if certain)
PEOPLE = [
    ("Albert Einstein", "scientist", "Germany", "1879"), ("Isaac Newton", "scientist", "England", None),
    ("Charles Darwin", "scientist", "England", "1809"), ("Marie Curie", "scientist", "Poland", "1867"),
    ("Galileo Galilei", "scientist", "Italy", "1564"), ("Louis Pasteur", "scientist", "France", "1822"),
    ("Michael Faraday", "scientist", "England", "1791"),
    ("Leonardo da Vinci", "painter", "Italy", "1452"), ("Pablo Picasso", "painter", "Spain", "1881"),
    ("Vincent van Gogh", "painter", "the Netherlands", "1853"), ("Claude Monet", "painter", "France", "1840"),
    ("Rembrandt", "painter", "the Netherlands", "1606"), ("Frida Kahlo", "painter", "Mexico", "1907"),
    ("Ludwig van Beethoven", "composer", "Germany", "1770"),
    ("Wolfgang Amadeus Mozart", "composer", "Austria", "1756"),
    ("Johann Sebastian Bach", "composer", "Germany", "1685"), ("Frédéric Chopin", "composer", "Poland", "1810"),
    ("Pyotr Tchaikovsky", "composer", "Russia", "1840"),
    ("William Shakespeare", "writer", "England", "1564"), ("Charles Dickens", "writer", "England", "1812"),
    ("Jane Austen", "writer", "England", "1775"), ("Leo Tolstoy", "writer", "Russia", "1828"),
    ("Mark Twain", "writer", "the United States", "1835"),
    ("Plato", "philosopher", "Greece", None), ("Aristotle", "philosopher", "Greece", None),
    ("Friedrich Nietzsche", "philosopher", "Germany", "1844"), ("Confucius", "philosopher", "China", None),
    ("Christopher Columbus", "explorer", "Italy", None), ("Ferdinand Magellan", "explorer", "Portugal", None),
    ("Marco Polo", "explorer", "Italy", None), ("Vasco da Gama", "explorer", "Portugal", None),
    ("Abraham Lincoln", "politician", "the United States", "1809"),
    ("Winston Churchill", "politician", "England", "1874"),
    ("Joseph Stalin", "politician", "Georgia", None), ("Adolf Hitler", "politician", "Austria", "1889"),
    ("Nelson Mandela", "politician", "South Africa", "1918"),
    ("Napoleon Bonaparte", "general", "France", "1769"), ("Julius Caesar", "general", "Italy", None),
]
LEVELS = {   # name: (prompt with {} for the name, the same with no name, the options, which field is right)
    "category": ("{} was a famous", "This person was a famous", sorted({p[1] for p in PEOPLE}), 1),
    "birthplace": ("{} was born in", "This person was born in", sorted({p[2] for p in PEOPLE}), 2),
    "birth year": ("{} was born in the year", "This person was born in the year",
                   sorted({p[3] for p in PEOPLE if p[3]}), 3),
}

@torch.no_grad()
def logprob(model, tok, prompt, answer):
    """log P(answer | prompt), summed over the answer's tokens, after <|endoftext|>."""
    p, a = tok.encode(prompt), tok.encode(answer)
    ids = torch.tensor([[tok.eot_id] + p + a], device=next(model.parameters()).device)
    logp = F.log_softmax(model(ids[:, :-1]).float(), dim=-1)[0]
    return sum(logp[len(p) + j, a[j]].item() for j in range(len(a)))

def probe(model, tok, people=PEOPLE, levels=LEVELS):
    """{level: {"raw": share right, "calibrated": share right, "margin": mean calibrated
    score of the right option minus the best wrong one, "answers": {name: calibrated pick}}}."""
    out = {}
    for level, (template, neutral, options, field) in levels.items():
        prior = {o: logprob(model, tok, neutral, " " + o) for o in options}
        raw = calibrated = 0
        margins, answers = [], {}
        known = [p for p in people if p[field]]           # only people with a certain answer
        for person in known:
            name, right = person[0], person[field]
            scores = {o: logprob(model, tok, template.format(name), " " + o) for o in options}
            pmi = {o: scores[o] - prior[o] for o in options}
            raw += max(scores, key=scores.get) == right
            pick = max(pmi, key=pmi.get)
            calibrated += pick == right
            margins.append(pmi[right] - max(v for o, v in pmi.items() if o != right))
            answers[name] = pick
        out[level] = {"raw": raw / len(known), "calibrated": calibrated / len(known),
                      "margin": st.mean(margins), "chance": 1 / len(options), "answers": answers}
    return out

def show(path, result):
    print(f"\n{path}")
    for level, r in result.items():
        print(f"   {level:10s} ({len(r['answers'])} people) raw {r['raw']:4.0%}   calibrated {r['calibrated']:4.0%}"
              f"   margin {r['margin']:+6.2f}   (chance {r['chance']:.0%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="web-tokeniser models to probe")
    parser.add_argument("--answers", action="store_true", help="also show each person's calibrated answers")
    args = parser.parse_args()
    tok = WordBPE.load(TOKENISER)
    for path in args.checkpoints:
        model = gpt.load(args.device, path)
        result = probe(model, tok)
        show(path.name, result)
        if args.answers:
            for name, *truth in PEOPLE:
                got = [result[level]["answers"].get(name) for level in LEVELS]
                marks = ["-" if t is None else "ok" if g == t else f"said {g}" for g, t in zip(got, truth)]
                print(f"      {name:24s} {truth[0]:11s} {marks[0]:16s} {truth[1]:17s} {marks[1]:20s}"
                      f" {truth[2] or '':5s} {marks[2]}")
        del model
        if args.device == "mps":
            torch.mps.empty_cache()
