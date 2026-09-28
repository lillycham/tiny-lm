"""Wikipedia's Vital Articles for an anneal: ~10,000 of its most important articles.

Vital Articles are lists that editors keep of the articles Wikipedia most needs. Level 4
has ~10,000, in 11 topics (People, History, Geography, Arts, ...): a clean, factual
slice across everything, ~60M tokens. The lists are wiki pages; each article is a line
that starts with "#", and its first link is the article:

    # {{Icon|GA}} '''[[Albert Einstein]]'''<!--b. 1879--> ([[Wikipedia:Vital articles/Level 3|Level 3]])

The text comes from Hugging Face's plain-text copy of English Wikipedia (Nov 2023,
41 Parquet files, 11.6 GB), filtered to the listed titles. CC BY-SA 4.0.

Half of the knowledge probe's people are left out (tools/knowledge.py WIKI_HELD_OUT), so
the probe can compare people the anneal read about with people it didn't.

    python -m web.wikipedia list                # data/vital_articles_4.json, ~1 min of API requests
    python -m web.wikipedia download            # the dump, 11.6 GB: on a rented server
    python -m web.wikipedia filter              # data/vital_articles_4.parquet
    python -m web.wikipedia encode              # data/wiki_{train,val}.bin
    python -m web.anneal --data wiki --share 0.25 --tag wiki25 --compile
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.knowledge import PEOPLE, WIKI_HELD_OUT
from web import web_data

API = "https://en.wikipedia.org/w/api.php"
AGENT = "tiny-lm/0.1 (a learning project; https://github.com/lillycham/tiny-lm)"   # Wikipedia asks for one
TOPICS = ["People", "History", "Geography", "Arts", "Philosophy and religion", "Everyday life",
          "Society and social sciences", "Biology and health sciences", "Physical sciences",
          "Technology", "Mathematics"]
LIST = Path("data/vital_articles_4.json")
DUMP_DIR = Path("data/wikipedia")
DUMP_URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-{i:05d}-of-00041.parquet"
DUMP_FILES = 41
ARTICLES = Path("data/vital_articles_4.parquet")
VAL_SHARE = 0.02            # of the articles, for the anneal's val loss
TOKENS = Path("data/wiki_{split}.bin")
# Probe names whose article has another title.
ARTICLE_TITLES = {"Napoleon Bonaparte": "Napoleon", "Pyotr Tchaikovsky": "Pyotr Ilyich Tchaikovsky"}

# ---------- the list ----------
LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$")

def entries(wikitext):
    """[(title, section)] for each list line: the first link that isn't to another
    namespace (those have a ":", like "Wikipedia:..." or "Category:...")."""
    out, section = [], ""
    for line in wikitext.splitlines():
        if m := HEADING.match(line):
            section = m.group(2)
        elif line.startswith("#"):
            for target in LINK.findall(line):
                if ":" not in target:
                    out.append((target.strip().replace("_", " "), section))
                    break
    return out

def api(**params):
    """One request to the Wikipedia API, as JSON."""
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2, "maxlag": 5})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=60) as r:
        return json.load(r)

def page_wikitext(title):
    return api(action="parse", page=title, prop="wikitext")["parse"]["wikitext"]

def resolve(titles, batch=50):
    """{title: the article's current title}, following redirects. Missing pages are left out."""
    out = {}
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        q = api(action="query", titles="|".join(chunk), redirects=1)["query"]
        follow = {r["from"]: r["to"] for r in q.get("redirects", [])}
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        found = {p["title"] for p in q["pages"] if not p.get("missing")}
        for t in chunk:
            final = follow.get(norm.get(t, t), norm.get(t, t))
            if final in found:
                out[t] = final
        time.sleep(0.2)                              # be polite
    return out

def build_list():
    """[{"title", "topic", "section"}] for every Level 4 article, titles resolved."""
    rows = []
    for topic in TOPICS:
        found = entries(page_wikitext(f"Wikipedia:Vital articles/Level 4/{topic}"))
        rows += [{"title": t, "topic": topic, "section": s} for t, s in found]
        print(f"{topic}: {len(found):,} articles", flush=True)
        time.sleep(0.5)
    titles = sorted({r["title"] for r in rows})
    canonical = resolve(titles)
    out, seen = [], set()
    for r in rows:
        t = canonical.get(r["title"])
        if t and t not in seen:
            seen.add(t)
            out.append({**r, "title": t})
    print(f"{len(out):,} articles after redirects ({len(rows) - len(out)} duplicates or missing)")
    return out

def held_out_titles():
    """The articles of the probe people left out of the anneal."""
    return {ARTICLE_TITLES.get(name, name) for name in WIKI_HELD_OUT}

# ---------- the text ----------
def filter_dump(files, wanted, out, row_group_size=200):
    """The rows of the dump's Parquet files whose title is in wanted ({title: topic}),
    as a Parquet file of title, topic and text. Reads the titles first, and the text
    only of the row groups with a match."""
    kept = []
    for path in files:
        f = pq.ParquetFile(path)
        for g in range(f.num_row_groups):
            titles = f.read_row_group(g, columns=["title"]).column("title").to_pylist()
            if not any(t in wanted for t in titles):
                continue
            table = f.read_row_group(g, columns=["title", "text"])
            for t, text in zip(table.column("title").to_pylist(), table.column("text").to_pylist()):
                if t in wanted:
                    kept.append({"title": t, "topic": wanted[t], "text": text})
        print(f"{path.name}: {len(kept):,} articles so far", flush=True)
    kept.sort(key=lambda r: r["title"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(kept), out, row_group_size=row_group_size)
    return len(kept)

def split_jobs(path, val_share=VAL_SHARE):
    """(train jobs, val jobs) for web_data.encode: every 1 / val_share-th row group is val."""
    n = pq.ParquetFile(path).num_row_groups
    every = max(2, round(1 / val_share))
    val = [(path, g) for g in range(n) if g % every == every - 1]
    return [(path, g) for g in range(n) if g % every != every - 1], val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "download", "filter", "encode"])
    parser.add_argument("--files", type=int, default=DUMP_FILES, help="dump files to use (default all 41)")
    args = parser.parse_args()

    if args.command == "list":
        rows = build_list()
        LIST.parent.mkdir(parents=True, exist_ok=True)
        LIST.write_text(json.dumps(rows, ensure_ascii=False, indent=0))
        missing = {ARTICLE_TITLES.get(p[0], p[0]) for p in PEOPLE} - {r["title"] for r in rows}
        print(f"Saved to {LIST}. Probe people not in the list: {sorted(missing) or 'none'}")
    elif args.command == "download":
        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(args.files):
            path = DUMP_DIR / f"train-{i:05d}.parquet"
            if not path.exists():
                tmp = path.with_suffix(".tmp")
                urllib.request.urlretrieve(DUMP_URL.format(i=i), tmp)
                tmp.replace(path)
            print(f"{path} {path.stat().st_size / 1e6:,.0f} MB", flush=True)
    elif args.command == "filter":
        skip = held_out_titles()
        wanted = {r["title"]: r["topic"] for r in json.loads(LIST.read_text()) if r["title"] not in skip}
        files = sorted(DUMP_DIR.glob("train-*.parquet"))[:args.files]
        n = filter_dump(files, wanted, ARTICLES)
        print(f"{n:,} of {len(wanted):,} listed articles found in {len(files)} files; "
              f"{len(skip)} probe people held out. Saved to {ARTICLES}")
    else:
        train, val = split_jobs(ARTICLES)
        for split, jobs in (("val", val), ("train", train)):
            web_data.encode(jobs, Path(str(TOKENS).format(split=split)), column="text")
