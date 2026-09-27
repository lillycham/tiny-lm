"""FineWeb-Edu for Part 3: a 16,384-token tokeniser, and the web text as token files.

word_bpe.py did this for TinyStories. FineWeb-Edu is general English from the web,
filtered for educational pages: 10B tokens in 14 Parquet files. A Parquet file is
split into row groups of about 1,000 documents; each one can be read on its own.

    python -m web.web_data tokeniser                                  # ~5 min, from the sample file
    python -m web.web_data encode val                                 # the validation split
    python -m web.web_data encode train data/fineweb_edu_000.parquet  # and more files after it

The tokeniser: word_bpe.py's WordBPE with 16,384 tokens, not 4,096. On held-out
FineWeb-Edu text that is 4.20 characters per token, against 4.57 for GPT-2's
50,257; 32,768 would give 4.49 but double the embedding.

The token files are raw uint16, one after another, with <|endoftext|> after each
document, read with np.memmap. Not .npy: np.save needs the whole array in memory
first, and 2.5B tokens are 5 GB. Here each row group's tokens go to the end of the
file as soon as they are ready.

Encoding runs on every CPU core: one row group per job, in a multiprocessing.Pool.
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from core.word_bpe import WordBPE

VOCAB_SIZE = 16384
EOT_ID = VOCAB_SIZE - 1               # 256 bytes + 16,127 merges + <|endoftext|>
TOKENISER = Path("checkpoints/web/web_bpe.json")
TOKENS = Path("data/web_{split}.bin")
SAMPLE = Path("data/fineweb_edu_013.parquet")    # the smallest of the 14 files, 541 MB
VAL_GROUPS = 5                        # its first 5 row groups (~5,000 documents) are validation
TOKENISER_GROUPS = range(10, 30)      # ~20,000 documents (~95 MB) train the tokeniser

# ---------- reading ----------
def read_group(path, group, column="text"):
    """The texts of one row group of a Parquet file. SimpleStories keeps them in "story"."""
    return pq.ParquetFile(path).read_row_group(group, columns=[column]).column(column).to_pylist()

def jobs(split, files=()):
    """(path, row group) for every row group in the split, in order.

    val is the first VAL_GROUPS row groups of SAMPLE. train is every row group of
    files, except those, if SAMPLE is one of the files.
    """
    if split == "val":
        return [(SAMPLE, g) for g in range(VAL_GROUPS)]
    out = []
    for path in map(Path, files):
        skip = VAL_GROUPS if path.resolve() == SAMPLE.resolve() else 0
        out += [(path, g) for g in range(skip, pq.ParquetFile(path).num_row_groups)]
    return out

def load_tokens(split):
    """The token IDs that encode() saved for split, read from disk only as needed."""
    return np.memmap(str(TOKENS).format(split=split), dtype=np.uint16, mode="r")

# ---------- the tokeniser ----------
def train_tokeniser(path=SAMPLE, groups=TOKENISER_GROUPS, vocab_size=VOCAB_SIZE):
    texts = [s for g in groups for s in read_group(path, g)]
    print(f"Training on {len(texts):,} documents, {sum(map(len, texts)) / 1e6:.0f} MB", flush=True)
    return WordBPE.train(texts, vocab_size)

# ---------- encoding, on every core ----------
tok = None      # each worker process's own tokeniser, set by start_worker
column = "text" # and the Parquet column its texts are in

def start_worker(tokeniser_path, text_column="text"):
    """Runs once in each worker process, before its first job.

    Every process has its own memory, so each needs its own tokeniser. Loading it
    here, once per process, is much cheaper than sending it along with every job.
    Each copy also keeps its own cache of encoded words, which grows as it works.
    """
    global tok, column
    tok, column = WordBPE.load(tokeniser_path), text_column

def encode_group(job):
    """One job, (path, row group) -> its tokens: each document, then <|endoftext|>,
    as one uint16 array. Runs in a worker process, with the globals tok and column."""
    path, group = job

    texts = read_group(path, group, column)
    return np.array([id for text in texts 
                     for id in (tok.encode(text) + [EOT_ID])],
                     dtype=np.uint16)

def encode(job_list, out, processes=None, tokeniser_path=TOKENISER, max_tokens=None, column="text"):
    """Encode every job in a Pool of worker processes, and append the tokens to out,
    in job order. Stop after the job that passes max_tokens. Returns the count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    total, start = 0, time.time()

    with open(out, "wb") as f, mp.Pool(processes, initializer=start_worker,
                                 initargs=(tokeniser_path, column)) as pool:
        for count, a in enumerate(pool.imap(encode_group, job_list), 1):
            a.tofile(f)
            total += len(a)
            if max_tokens is not None and max_tokens <= total:
                break
            if count % 20 == 0:
                print(f"{count} jobs completed, tokens: {total:,}, tps: {total/(time.time() - start):,.0f}",
                      flush=True)

    print(f"{total:,} tokens in {time.time() - start:.0f}s, saved to {out}", flush=True)
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tokeniser", help=f"train the tokeniser on part of {SAMPLE}, save to {TOKENISER}")
    e = sub.add_parser("encode", help="encode Parquet files into data/web_{split}.bin")
    e.add_argument("split", choices=["train", "val"])
    e.add_argument("files", nargs="*", type=Path, help="Parquet files, for train")
    e.add_argument("--processes", type=int, default=mp.cpu_count(), help="default: every core")
    e.add_argument("--max-tokens", type=float, help="stop after about this many tokens, e.g. 2.7e9")
    args = parser.parse_args()

    if args.command == "tokeniser":
        t = time.time()
        tokeniser = train_tokeniser()
        tokeniser.save(TOKENISER)
        print(f"{len(tokeniser.merges):,} merges in {time.time() - t:.0f}s, saved to {TOKENISER}")
        held = [s for g in range(VAL_GROUPS) for s in read_group(SAMPLE, g)]
        ids = [i for s in held for i in tokeniser.encode(s)]
        print(f"Validation text: {sum(map(len, held)) / len(ids):.2f} characters per token")
    else:
        if args.split == "train" and not args.files:
            sys.exit("encode train needs one or more Parquet files")
        todo = jobs(args.split, args.files)
        print(f"{len(todo):,} row groups on {args.processes} processes", flush=True)
        encode(todo, Path(str(TOKENS).format(split=args.split)), args.processes,
               max_tokens=int(args.max_tokens) if args.max_tokens else None)
