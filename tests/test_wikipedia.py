"""web/wikipedia.py: reading the Vital Articles lists, and filtering the dump."""
import pyarrow as pa
import pyarrow.parquet as pq

from tools import knowledge
from web import wikipedia as wiki

PAGE = """<noinclude>{{Short description|Larger list}}</noinclude>
See [[Wikipedia:Vital articles/Level/4|Level 4]].
==Scientists==
===Physicists===
# {{Icon|GA}} {{Icon|FFA}} '''[[Albert Einstein]]'''<!--b. 1879--> ([[Wikipedia:Vital articles/Level 3|Level 3]])
# {{Icon|C}} [[Pierre Curie]] <!--b. 1859-->
# [[Category:Physicists]] [[Isaac_Newton|Newton]]
# {{Icon|B}} [[Big Bang#History|The Big Bang]]
== Painters ==
# [[Claude Monet]]
* [[Not a list line]]
#: [[Wikipedia:Notes]] only
"""

def test_entries_take_the_first_article_link_of_each_list_line():
    assert wiki.entries(PAGE) == [("Albert Einstein", "Physicists"), ("Pierre Curie", "Physicists"),
                                  ("Isaac Newton", "Physicists"), ("Big Bang", "Physicists"),
                                  ("Claude Monet", "Painters")]

def test_filter_keeps_only_wanted_titles(tmp_path):
    dump = tmp_path / "train-00000.parquet"
    rows = [{"id": str(i), "url": "", "title": t, "text": f"Text of {t}."}
            for i, t in enumerate(["Aardvark", "Albert Einstein", "Banana", "Claude Monet", "Zebra"])]
    pq.write_table(pa.Table.from_pylist(rows), dump, row_group_size=2)
    out = tmp_path / "articles.parquet"
    n = wiki.filter_dump([dump], {"Claude Monet": "Arts", "Albert Einstein": "People", "Missing": "X"}, out)
    got = pq.read_table(out).to_pylist()
    assert n == 2 and [r["title"] for r in got] == ["Albert Einstein", "Claude Monet"]
    assert got[1] == {"title": "Claude Monet", "topic": "Arts", "text": "Text of Claude Monet."}

def test_split_jobs_puts_every_fiftieth_row_group_in_val(tmp_path):
    path = tmp_path / "a.parquet"
    pq.write_table(pa.table({"text": [str(i) for i in range(100)]}), path, row_group_size=1)
    train, val = wiki.split_jobs(path, val_share=0.02)
    assert [g for _, g in val] == [49, 99] and len(train) == 98
    assert not {g for _, g in train} & {g for _, g in val}

def test_split_jobs_always_has_a_val_group(tmp_path):
    path = tmp_path / "small.parquet"
    pq.write_table(pa.table({"text": [str(i) for i in range(4)]}), path, row_group_size=1)
    train, val = wiki.split_jobs(path, val_share=0.02)
    assert [g for _, g in val] == [3] and len(train) == 3

def test_half_the_probe_people_are_held_out():
    names = [p[0] for p in knowledge.PEOPLE]
    assert len(knowledge.WIKI_HELD_OUT) == len(names) // 2
    held = wiki.held_out_titles()
    assert "Napoleon" in held or "Napoleon Bonaparte" not in knowledge.WIKI_HELD_OUT
    assert "Albert Einstein" not in held                     # the first person is read

def test_merge_keeps_one_row_per_title(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pq.write_table(pa.Table.from_pylist([{"title": "B", "topic": "x", "text": "dump B"},
                                         {"title": "A", "topic": "x", "text": "dump A"}]), a)
    pq.write_table(pa.Table.from_pylist([{"title": "A", "topic": "x", "text": "api A"},
                                         {"title": "C", "topic": "y", "text": "api C"}]), b)
    out = tmp_path / "full.parquet"
    assert wiki.merge([a, b], out) == 3
    assert [(r["title"], r["text"]) for r in pq.read_table(out).to_pylist()] == \
        [("A", "dump A"), ("B", "dump B"), ("C", "api C")]

def test_fetch_missing_skips_what_it_has(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(wiki, "extract", lambda t: asked.append(t) or (None if t == "Gone" else f"Text of {t}."))
    out = tmp_path / "api.parquet"
    n, todo = wiki.fetch_missing({"A": "x", "B": "y", "Gone": "z"}, ["A"], out)
    assert asked == ["B", "Gone"] and (n, todo) == (1, 2)
    assert pq.read_table(out).to_pylist() == [{"title": "B", "topic": "y", "text": "Text of B."}]
