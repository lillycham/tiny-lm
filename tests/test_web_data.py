"""web_data.py: the splits, the encoding worker and the parallel encoder.

A tiny Parquet file (3 row groups of 2 documents) and a tiny tokeniser stand in for
FineWeb-Edu, so these run in a second or two.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import web_data as wd
from word_bpe import WordBPE

DOCS = ["The cat sat on the mat.", "Water boils at 100 °C.", "Photosynthesis makes sugar.",
        "Rivers carry water to the sea.", "“Quotes” and dashes — too.", "The end."]

@pytest.fixture
def files(tmp_path):
    """A Parquet file of DOCS, 2 per row group, and a small tokeniser trained on them."""
    path = tmp_path / "docs.parquet"
    pq.write_table(pa.table({"text": DOCS, "id": [str(i) for i in range(len(DOCS))]}), path, row_group_size=2)
    tok = WordBPE.train(DOCS * 3, 300)
    tok_path = tmp_path / "bpe.json"
    tok.save(tok_path)
    return path, tok_path, WordBPE.load(tok_path)

def expected(tok, docs):
    return [i for d in docs for i in tok.encode(d) + [wd.EOT_ID]]

def test_read_group(files):
    path, _, _ = files
    assert pq.ParquetFile(path).num_row_groups == 3
    assert wd.read_group(path, 1) == DOCS[2:4]

def test_val_is_kept_out_of_train(files, monkeypatch):
    path, _, _ = files
    monkeypatch.setattr(wd, "SAMPLE", path)
    monkeypatch.setattr(wd, "VAL_GROUPS", 1)
    assert wd.jobs("val") == [(path, 0)]
    assert wd.jobs("train", [path]) == [(path, 1), (path, 2)]      # without the val group

def test_encode_group(files):
    """Every document, then <|endoftext|>, as uint16: IDs above 32,767 need unsigned."""
    path, tok_path, tok = files
    wd.start_worker(tok_path)                                     # as a worker process would
    got = wd.encode_group((path, 2))
    assert isinstance(got, np.ndarray) and got.dtype == np.uint16
    assert got.tolist() == expected(tok, DOCS[4:6])
    assert (got == wd.EOT_ID).sum() == 2
    assert tok.decode([i for i in got.tolist() if i != wd.EOT_ID]) == DOCS[4] + DOCS[5]

@pytest.mark.parametrize("processes", [1, 2])
def test_encode_writes_every_job_in_order(files, tmp_path, processes):
    """imap gives results back in job order, whichever worker finishes first."""
    path, tok_path, tok = files
    out = tmp_path / "web_train.bin"
    job_list = [(path, 2), (path, 0), (path, 1)]                  # not in file order, on purpose
    total = wd.encode(job_list, out, processes, tok_path)
    got = np.memmap(out, dtype=np.uint16, mode="r")
    assert total == len(got)
    assert got.tolist() == expected(tok, DOCS[4:6] + DOCS[0:2] + DOCS[2:4])

def test_encode_stops_at_max_tokens(files, tmp_path):
    path, tok_path, tok = files
    out = tmp_path / "web_train.bin"
    first = len(expected(tok, DOCS[0:2]))
    total = wd.encode([(path, 0), (path, 1), (path, 2)], out, 1, tok_path, max_tokens=first + 1)
    assert total == len(expected(tok, DOCS[0:4]))                  # stops after the job that passed it
    assert len(np.memmap(out, dtype=np.uint16, mode="r")) == total

def test_encode_replaces_an_old_file(files, tmp_path):
    """A second run must not add to the end of the first one's tokens."""
    path, tok_path, tok = files
    out = tmp_path / "web_train.bin"
    wd.encode([(path, 0)], out, 1, tok_path)
    wd.encode([(path, 0)], out, 1, tok_path)
    assert np.memmap(out, dtype=np.uint16, mode="r").tolist() == expected(tok, DOCS[0:2])
