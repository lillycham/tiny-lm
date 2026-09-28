"""A knowledge probe: what does a web model know about famous people?

Two levels, each a choice between fixed options, scored by log-probability:
  1. Category:  "Albert Einstein was a famous" -> " scientist"? (of 8 categories)
  2. Birthplace: "Albert Einstein was born in"  -> " Germany"?   (of 16 countries)

Associations (Einstein, physics) appear together thousands of times in the training
text; a birthplace appears far less often. So level 1 should come first and easier.
Some birthplaces are traps for a model that goes by association: Stalin was born in
Georgia, not Russia, and Hitler in Austria, not Germany.

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

# (name, category, country of birth, as it is today)
PEOPLE = [
    ("Albert Einstein", "scientist", "Germany"), ("Isaac Newton", "scientist", "England"),
    ("Charles Darwin", "scientist", "England"), ("Marie Curie", "scientist", "Poland"),
    ("Galileo Galilei", "scientist", "Italy"), ("Louis Pasteur", "scientist", "France"),
    ("Michael Faraday", "scientist", "England"),
    ("Leonardo da Vinci", "painter", "Italy"), ("Pablo Picasso", "painter", "Spain"),
    ("Vincent van Gogh", "painter", "the Netherlands"), ("Claude Monet", "painter", "France"),
    ("Rembrandt", "painter", "the Netherlands"), ("Frida Kahlo", "painter", "Mexico"),
    ("Ludwig van Beethoven", "composer", "Germany"), ("Wolfgang Amadeus Mozart", "composer", "Austria"),
    ("Johann Sebastian Bach", "composer", "Germany"), ("Frédéric Chopin", "composer", "Poland"),
    ("Pyotr Tchaikovsky", "composer", "Russia"),
    ("William Shakespeare", "writer", "England"), ("Charles Dickens", "writer", "England"),
    ("Jane Austen", "writer", "England"), ("Leo Tolstoy", "writer", "Russia"),
    ("Mark Twain", "writer", "the United States"),
    ("Plato", "philosopher", "Greece"), ("Aristotle", "philosopher", "Greece"),
    ("Friedrich Nietzsche", "philosopher", "Germany"), ("Confucius", "philosopher", "China"),
    ("Christopher Columbus", "explorer", "Italy"), ("Ferdinand Magellan", "explorer", "Portugal"),
    ("Marco Polo", "explorer", "Italy"), ("Vasco da Gama", "explorer", "Portugal"),
    ("Abraham Lincoln", "politician", "the United States"), ("Winston Churchill", "politician", "England"),
    ("Joseph Stalin", "politician", "Georgia"), ("Adolf Hitler", "politician", "Austria"),
    ("Nelson Mandela", "politician", "South Africa"),
    ("Napoleon Bonaparte", "general", "France"), ("Julius Caesar", "general", "Italy"),
]
LEVELS = {   # name: (prompt with {} for the name, the same with no name, the options, which field is right)
    "category": ("{} was a famous", "This person was a famous", sorted({p[1] for p in PEOPLE}), 1),
    "birthplace": ("{} was born in", "This person was born in", sorted({p[2] for p in PEOPLE}), 2),
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
        for person in people:
            name, right = person[0], person[field]
            scores = {o: logprob(model, tok, template.format(name), " " + o) for o in options}
            pmi = {o: scores[o] - prior[o] for o in options}
            raw += max(scores, key=scores.get) == right
            pick = max(pmi, key=pmi.get)
            calibrated += pick == right
            margins.append(pmi[right] - max(v for o, v in pmi.items() if o != right))
            answers[name] = pick
        out[level] = {"raw": raw / len(people), "calibrated": calibrated / len(people),
                      "margin": st.mean(margins), "chance": 1 / len(options), "answers": answers}
    return out

def show(path, result):
    print(f"\n{path}")
    for level, r in result.items():
        print(f"   {level:10s} raw {r['raw']:4.0%}   calibrated {r['calibrated']:4.0%}"
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
                got = [result[level]["answers"][name] for level in LEVELS]
                marks = ["ok" if g == t else f"said {g}" for g, t in zip(got, truth)]
                print(f"      {name:24s} {truth[0]:11s} {marks[0]:16s} {truth[1]:17s} {marks[1]}")
        del model
        if args.device == "mps":
            torch.mps.empty_cache()
