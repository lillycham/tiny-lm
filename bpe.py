"""Byte-pair encoding (BPE): the tokeniser that GPT-2, Llama and most other LLMs use.

So far every model read one character at a time: 65 tokens. BPE makes a bigger
vocabulary by merging. It starts from the 256 byte values, then again and again:

    1. Count every pair of neighbouring tokens in the training text.
    2. Take the most common pair, say ("t", "h"), and give it a new token ID.
    3. Replace every "t" "h" in the text with that new token.

After 256 merges there are 512 tokens: bytes, then "th", "the", " the", ... Common
words become one token, and rare words stay as smaller pieces, so any text still
works. The same text becomes fewer tokens, so a model with the same context length
sees more of it.

    python bpe.py
"""
import time
from collections import Counter

from data import split, text

VOCAB_SIZE = 512    # 256 bytes + 256 merges

# ---------- the two building blocks ----------
def pair_counts(ids):
    """How often each pair of neighbours appears. [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}."""
    # TODO(Lilly): one line. zip(ids, ids[1:]) gives every neighbour pair:
    #   (ids[0], ids[1]), (ids[1], ids[2]), ... and Counter(...) counts them, like
    #   the bigram counts at the very start. Return the Counter.
    raise NotImplementedError

def merge(ids, pair, new_id):
    """Replace every pair in ids with new_id. merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]."""
    # TODO(Lilly): walk through ids with an index i and build a new list, out.
    #   - If ids[i] and ids[i + 1] are the pair, append new_id and jump 2 ahead.
    #   - If not, append ids[i] and go 1 ahead.
    #   Use a while loop, not a for loop, because i sometimes jumps by 2.
    #   Careful at the end: the last item has no ids[i + 1].
    raise NotImplementedError

# ---------- the tokeniser ----------
class BPE:
    def __init__(self, merges):
        """merges: {(a, b): new_id}, in the order they were learned."""
        self.merges = merges
        # The bytes of every token: a merged token is its two parts joined.
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @classmethod
    def train(cls, text, vocab_size):
        """Learn vocab_size - 256 merges from text."""
        ids = list(text.encode("utf-8"))
        merges = {}
        for new_id in range(256, vocab_size):
            # TODO(Lilly): the three steps from the top of the file.
            #   1. Count the pairs in ids.
            #   2. Find the most common pair: max(counts, key=counts.get) gives the
            #      key with the biggest value.
            #   3. Merge that pair in ids, and record it: merges[pair] = new_id.
            raise NotImplementedError
        return cls(merges)

    def encode(self, s):
        """Text -> token IDs, with the merges that training learned."""
        ids = list(s.encode("utf-8"))
        # TODO(Lilly): do the merges again, in the order they were learned. Merge
        #   256 can use pairs that merge 255 made, so the order matters.
        #   While ids has at least 2 items:
        #     1. Count the pairs in ids.
        #     2. Of those pairs, find the one that was learned first: the one with the
        #        smallest ID in self.merges. Pairs that aren't in self.merges must lose,
        #        so give them infinity:
        #            min(counts, key=lambda p: self.merges.get(p, float("inf")))
        #     3. If that pair isn't in self.merges, no pair can merge any more: break.
        #     4. Merge it, with self.merges[pair] as the new ID.
        raise NotImplementedError
        return ids

    def decode(self, ids):
        # errors="replace": a token list can end halfway through a multi-byte character.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def show(self, ids):
        """The tokens of ids, with | between them."""
        return "|".join(self.decode([i]) for i in ids)

# ---------- checks ----------
if __name__ == "__main__":
    print(f"1. pair_counts([1, 2, 3, 1, 2]) = {dict(pair_counts([1, 2, 3, 1, 2]))}")
    print(f"   merge([1, 2, 3, 1, 2], (1, 2), 99) = {merge([1, 2, 3, 1, 2], (1, 2), 99)}")
    print("   (expected {(1, 2): 2, (2, 3): 1, (3, 1): 1} and [99, 3, 99])")

    train_text, val_text = text[:split], text[split:]
    t = time.time()
    bpe = BPE.train(train_text, VOCAB_SIZE)
    print(f"\n2. Learned {len(bpe.merges)} merges in {time.time() - t:.0f}s. The first 20:")
    print("   " + "  ".join(repr(bpe.vocab[i].decode()) for i in range(256, 276)))
    longest = sorted(bpe.vocab.values(), key=len)[-10:]
    print("   The 10 longest tokens: " + "  ".join(repr(b.decode()) for b in longest))

    t = time.time()
    val_ids = bpe.encode(val_text)
    print(f"\n3. Validation text: {len(val_text):,} characters -> {len(val_ids):,} tokens,"
          f" {len(val_text) / len(val_ids):.2f} characters per token ({time.time() - t:.0f}s)")
    for s in (val_text, "Héllo, wörld! 🐱"):
        print(f"   decode(encode(s)) == s: {bpe.decode(bpe.encode(s)) == s}  ({s[:20]!r})")

    line = "ROMEO: But soft, what light through yonder window breaks?"
    print(f"\n4. {bpe.show(bpe.encode(line))}")
