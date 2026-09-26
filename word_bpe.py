"""A BPE tokeniser fast enough for TinyStories: 2.2 GB of text.

bpe.py works on the whole text as one long list, and every merge walks the whole
list. That's fine for 1 MB of Shakespeare, but TinyStories would take days. Three
changes make it fast:

    1. Split the text into words first, GPT-2 style: " the", " cat", ".", "\\n".
       A space stays at the front of the word after it. Merges never cross from
       one word into the next, so "t, " can't become a token any more.
    2. Train on the unique words and how often each appears. " the" appears
       millions of times, but it only needs to be merged once, with its count
       as a weight. And a merge only touches the words that contain the pair.
    3. Encode each unique word once, and remember the result.

Each story also ends with a special token, <|endoftext|>, so the model can learn
where stories end.

    python word_bpe.py          # train the merges, then encode TinyStories to disk
"""
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from bpe import BPE, merge

VOCAB_SIZE = 4096                 # 256 bytes + 3839 merges + <|endoftext|>
EOT = "<|endoftext|>"
EOT_ID = VOCAB_SIZE - 1
TRAIN_FILE = Path("data/TinyStoriesV2-GPT4-train.txt")
VAL_FILE = Path("data/TinyStoriesV2-GPT4-valid.txt")
TOKENISER = Path("checkpoints/stories_bpe.json")
TOKENS = Path("data/tinystories_{split}.npy")   # token IDs, saved as 16-bit numbers

# A word is an optional space and then letters, or numbers, or punctuation. Or it is
# spaces and newlines. The first part splits off endings like 's and 'll.
WORD = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")

def split_words(text):
    """"Tom's cat sat." -> ["Tom", "'s", " cat", " sat", "."]."""
    return WORD.findall(text)

def stories(path, max_bytes=None):
    """The stories in a TinyStories file, one at a time, without the <|endoftext|> lines.

    One at a time, so the 2.2 GB file never has to fit in memory. With max_bytes,
    stop after about that many bytes.
    """
    lines, n = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n += len(line)
            if line.strip() != EOT:
                lines.append(line)
                continue
            story = "".join(lines).strip()
            if story:
                yield story
            lines = []
            if max_bytes and n >= max_bytes:
                return
    story = "".join(lines).strip()
    if story:
        yield story

def weighted_pair_counts(words):
    """words: {tuple of IDs: count}. How often each pair appears in the whole text."""
    # TODO(Lilly): like pair_counts in bpe.py, but for many words, each with a weight.
    #   Start with counts = Counter(). For each word's ids and count n in words.items(),
    #   add n to counts[pair] for every neighbour pair in ids (zip(ids, ids[1:]) again).
    #   A Counter starts every missing key at 0, so counts[pair] += n just works.
    counts = Counter()

    for ids, n in words.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += n
    return counts

class WordBPE(BPE):
    def __init__(self, merges):
        super().__init__(merges)
        self.vocab[EOT_ID] = EOT.encode()
        self.cache = {}

    @classmethod
    def train(cls, texts, vocab_size):
        """Learn vocab_size - 257 merges from a list of texts."""
        words = Counter()
        for text in texts:
            words.update(split_words(text))
        # Each unique word as a tuple of byte IDs, with how often it appears.
        words = {tuple(w.encode("utf-8")): n for w, n in words.items()}
        counts = weighted_pair_counts(words)
        merges = {}
        for new_id in range(256, vocab_size - 1):
            pair = max(counts, key=counts.get)
            merges[pair] = new_id
            changed = {}
            for ids, n in words.items():
                # Most words don't have the pair, so skip them quickly.
                if pair[0] not in ids or pair not in zip(ids, ids[1:]):
                    continue

                new_ids = tuple(merge(list(ids), pair, new_id))

                for p in zip(ids, ids[1:]):
                    counts[p] -= n

                for p in zip(new_ids, new_ids[1:]):
                    counts[p] += n

                changed[ids] = new_ids
            for ids, new_ids in changed.items():
                words[new_ids] = words.get(new_ids, 0) + words.pop(ids)
            del counts[pair]
        return cls(merges)

    @classmethod
    def load(cls, path):
        pairs = json.loads(path.read_text())
        return cls({(a, b): 256 + i for i, (a, b) in enumerate(pairs)})

    def encode_word(self, word):
        """One word's token IDs. Each word is encoded only once, then remembered."""
        if word not in self.cache:
            self.cache[word] = super().encode(word)
        return self.cache[word]

    def encode(self, s):
        """Text -> token IDs: split into words, then encode each word."""
        ids = []
        for word in split_words(s):
            ids.extend(self.encode_word(word))

        return ids

    def encode_stories(self, texts):
        """Every story's tokens, then <|endoftext|>, all in one 16-bit array."""
        # A Python list costs 8 bytes per ID, so turn it into a 2-byte array every
        # 10 million IDs.
        chunks, ids = [], []
        for text in texts:
            ids.extend(self.encode(text))
            ids.append(EOT_ID)
            if len(ids) > 10_000_000:
                chunks.append(np.array(ids, dtype=np.uint16))
                ids = []
        chunks.append(np.array(ids, dtype=np.uint16))
        return np.concatenate(chunks)

if __name__ == "__main__":
    s = "Tom's cat sat.\n\n\"Hi,\" she said.  Bye!"
    print(f"1. {split_words(s)}")
    print(f"   Joined again, the same text: {''.join(split_words(s)) == s}")

    t = time.time()
    train_texts = list(stories(TRAIN_FILE, max_bytes=10_000_000))
    tok = WordBPE.train(train_texts, VOCAB_SIZE)
    tok.save(TOKENISER)
    print(f"\n2. Learned {len(tok.merges)} merges from {len(train_texts):,} stories in {time.time() - t:.0f}s")
    print("   The first 20: " + "  ".join(repr(tok.vocab[i].decode()) for i in range(256, 276)))
    print("   Some later ones: " + "  ".join(repr(tok.vocab[i].decode(errors="replace")) for i in range(4000, 4010)))
    slow = BPE(tok.merges)
    print(f"   Same tokens as bpe.py for each word: {all(tok.encode_word(w) == slow.encode(w) for w in split_words(train_texts[0]))}")

    t = time.time()
    val_texts = list(stories(VAL_FILE))
    val_ids = tok.encode_stories(val_texts)
    n_chars = sum(len(s) for s in val_texts)
    print(f"\n3. Validation: {n_chars:,} characters -> {len(val_ids):,} tokens,"
          f" {n_chars / len(val_ids):.2f} characters per token ({time.time() - t:.0f}s)")
    print(f"   decode(encode(s)) == s: {tok.decode(tok.encode(val_texts[0])) == val_texts[0]}")
    print(f"   {tok.show(tok.encode(val_texts[1][:120]))}")
    np.save(str(TOKENS).format(split="val"), val_ids)

    t = time.time()
    train_ids = tok.encode_stories(stories(TRAIN_FILE))
    np.save(str(TOKENS).format(split="train"), train_ids)
    print(f"\n4. Training: {len(train_ids):,} tokens in {time.time() - t:.0f}s,"
          f" saved to {str(TOKENS).format(split='train')} ({train_ids.nbytes / 1e9:.1f} GB)")
