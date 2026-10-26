"""Which heads decide how a made-up name ends: a causal test on teacher-forced prefixes.

A name like Glegley is split into the pieces G|le|g|le|y, so "le" comes twice: once
followed by g, once by y. When the model writes the name again, it reaches the last
"le" and must choose: y ends the name, g starts another "gle" (a stutter). The score
of a prefix is

    logit(y) - logit(g)        at the last "le": above 0, the name ends

Every prefix is fixed text (teacher-forced), so nothing is sampled. The prefixes vary:

    name      repeated piece (Glegley, Plegley) or all pieces unique (Gegley, Kegley)
    mention   the 1st, 2nd or 3rd time the name is written again after the first
    polluted  the first mention already stuttered (Gleglegley), or not
    frame     20 short sentence starts ("One day,", "Mom said to", ...)

Two matrices, each head switched off by mean-ablation (its output replaced with its
average output on ordinary stories, so the model is not pushed into an unnatural state)
and, for comparison, by zero-ablation (interp.Ablate):

    heads        head x condition: the change in the score
    composition  head A x head B: the change in B's attention to the earlier "y"s when A
                 is off. If A feeds B, this is large.

    python -m tools.circuit                       # the 384-wide story model, on the CPU
    python -m tools.circuit --checkpoint checkpoints/stories/stories_gpt.pt
"""
import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core import gpt
from core.word_bpe import EOT_ID, TOKENISER, WordBPE
from stories.stories_gpt import load_tokens
from tools.interp import Ablate

MODEL = Path("checkpoints/stories/stories_gpt_384w6l_full.pt")
OUT = Path("logs/circuit")

NAMES = {"repeated": ["Glegley", "Plegley"], "unique": ["Gegley", "Kegley"]}
FRAMES = ["One day,", "Then", "After lunch,", "In the morning,", "Mom said to", "Later,", "The dog ran to",
          "Everyone loved", "So", "At night,", "The bird sang for", "Every day,", "Soon", "Her friend hugged",
          "The teacher smiled at", "Suddenly,", "That night,", "The cat followed", "Finally,", "Dad called"]
FILLERS = ["{} was very kind.", "{} liked to play in the park.", "{} had a red ball."]

# ---------- prefixes ----------
def stutter(name):
    """Glegley -> Gleglegley: one extra "gle", as the model writes it when it slips."""
    return name[:-4] + "gle" + name[-4:]

def prefix(name, mention, polluted, frame):
    """The story up to the last "le" of the name's mention-th reuse (1 = the first after
    the introduction). With polluted, the introduction already wrote the name with a stutter."""
    first = stutter(name) if polluted else name
    text = f"Once upon a time, there was a person named {first}."
    for i in range(mention - 1):
        text += " " + FILLERS[i].format(name)
    return f"{text} {frame} {name[:-1]}"          # (without the final "y")

def conditions():
    """Every (group, name, mention, polluted, frame), in a fixed order."""
    return [(group, name, mention, polluted, frame)
            for group, names in NAMES.items() for name in names
            for mention in (1, 2, 3) for polluted in (False, True) for frame in FRAMES]

def column(group, mention, polluted):
    """The name of a matrix column: the conditions it averages over (both names, every frame)."""
    return f"{group} m{mention}" + (" polluted" if polluted else "")

def encode(tok, text):
    """[<|endoftext|>] + the text, as stories began in training."""
    return [EOT_ID] + tok.encode(text)

def end_ids(tok):
    """The token IDs of "y" and "g" inside a name, from how the tokeniser splits Glegley."""
    ids = tok.encode(" Glegley")
    pieces = [tok.decode([i]) for i in ids]
    if pieces != [" G", "le", "g", "le", "y"]:
        raise ValueError(f"Glegley splits as {pieces}, not G|le|g|le|y: the test assumes that split")
    return ids[4], ids[2]

# ---------- switching heads off ----------
@torch.no_grad()
def head_means(model, tokens, chunks=32, seed=0):
    """Each layer's average input to attn.proj, (layers, C): the heads' outputs side by side,
    averaged over ordinary text. Mean-ablation puts a head's slice of this in place of its output."""
    rng = np.random.default_rng(seed)
    T = model.block
    starts = rng.integers(0, len(tokens) - T, size=chunks)
    X = torch.from_numpy(np.stack([np.asarray(tokens[s:s + T], dtype=np.int64) for s in starts]))
    sums, handles = {}, []
    for i, block in enumerate(model.blocks):
        def hook(module, inputs, i=i):
            sums[i] = inputs[0].float().mean(dim=(0, 1))
        handles.append(block.attn.proj.register_forward_pre_hook(hook))
    try:
        model(X.to(next(model.parameters()).device))
    finally:
        for h in handles:
            h.remove()
    return torch.stack([sums[i] for i in range(len(model.blocks))])

class MeanAblate:
    """Replace one head's output with its average (from head_means) while inside.

        with MeanAblate(model, means, layer=4, head=1):
            ...
    """
    def __init__(self, model, means, layer, head):
        self.attn = model.blocks[layer].attn
        self.hs = self.attn.proj.in_features // self.attn.n_head
        self.mean = means[layer, head * self.hs:(head + 1) * self.hs]
        self.head, self.handle = head, None

    def __enter__(self):
        self.handle = self.attn.proj.register_forward_pre_hook(self.replace)
        return self

    def __exit__(self, *exc):
        self.handle.remove()

    def replace(self, module, inputs):
        x = inputs[0].clone()
        x[:, :, self.head * self.hs:(self.head + 1) * self.hs] = self.mean.to(x.dtype)
        return x

def switched_off(model, how, means, heads):
    """A context that switches off every (layer, head) in heads, by "mean" or "zero"."""
    stack = ExitStack()
    for layer, head in heads:
        stack.enter_context(MeanAblate(model, means, layer, head) if how == "mean" else Ablate(model, layer, head))
    return stack

# ---------- measuring ----------
@torch.no_grad()
def scores(model, prefixes, y, g):
    """logit(y) - logit(g) at the end of each prefix, (N,). Prefixes of one length run as a batch."""
    out = torch.zeros(len(prefixes))
    device = next(model.parameters()).device
    by_length = {}
    for i, ids in enumerate(prefixes):
        by_length.setdefault(len(ids), []).append(i)
    for idx in by_length.values():
        X = torch.tensor([prefixes[i] for i in idx], device=device)
        logits = model(X, last_only=True)[:, -1].float().cpu()
        out[idx] = logits[:, y] - logits[:, g]
    return out

@torch.no_grad()
def y_attention(model, prefixes, y):
    """Each head's attention from the last position to every earlier "y", summed: (N, layers, heads).
    The earlier "y"s are the ends of the name's earlier mentions, the place an induction-style
    head looks to learn that the name ends here."""
    rows, handles = {}, []
    for i, block in enumerate(model.blocks):
        def hook(module, inputs, output, i=i):
            x = inputs[0]
            q, k, _ = module.qkv(x).split(x.shape[-1], dim=-1)
            q = module.split_heads(q)[:, :, -1:]
            k = module.smear_keys(module.split_heads(k))
            rows[i] = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)[:, :, 0]   # (B, H, T)
        handles.append(block.attn.register_forward_hook(hook))
    device = next(model.parameters()).device
    n_layers, n_heads = len(model.blocks), model.blocks[0].attn.n_head
    out = torch.zeros(len(prefixes), n_layers, n_heads)
    try:
        by_length = {}
        for i, ids in enumerate(prefixes):
            by_length.setdefault(len(ids), []).append(i)
        for idx in by_length.values():
            X = torch.tensor([prefixes[i] for i in idx], device=device)
            model(X, last_only=True)
            mask = (X == y).float()                                            # (B, T)
            for l in range(n_layers):
                out[idx, l] = (rows[l] * mask[:, None, :]).sum(-1).float().cpu()
    finally:
        for h in handles:
            h.remove()
    return out

def run(model, tok, how_list=("mean", "zero")):
    """Both matrices, as plain lists for JSON."""
    y, g = end_ids(tok)
    conds = conditions()
    prefixes = [encode(tok, prefix(name, m, p, f)) for _, name, m, p, f in conds]
    for ids in prefixes:
        if tok.decode(ids[-1:]) != "le":
            raise ValueError(f"A prefix does not end on 'le': {tok.decode(ids[-6:])!r}")
    cols = sorted({column(gr, m, p) for gr, _, m, p, _ in conds}, key=lambda c: (c.split()[0] != "repeated", c))
    col_of = torch.tensor([cols.index(column(gr, m, p)) for gr, _, m, p, _ in conds])

    def per_column(values):
        return [values[col_of == j].mean().item() for j in range(len(cols))]

    means = head_means(model, load_tokens("val"))
    n_layers, n_heads = len(model.blocks), model.blocks[0].attn.n_head
    heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    clean = scores(model, prefixes, y, g)
    result = {"columns": cols, "heads": [f"L{l}H{h}" for l, h in heads], "clean": per_column(clean)}
    for how in how_list:
        rows = []
        for head in heads:
            with switched_off(model, how, means, [head]):
                rows.append([a - b for a, b in zip(per_column(scores(model, prefixes, y, g)), result["clean"])])
            print(f"  {how}: L{head[0]}H{head[1]} done", flush=True)
        result[f"heads_{how}"] = rows

    # Composition, on the repeated names only (the stutter's home): A off -> B's y-attention.
    rep = [i for i, c in enumerate(conds) if c[0] == "repeated" and not c[3]]
    rep_prefixes = [prefixes[i] for i in rep]
    base = y_attention(model, rep_prefixes, y).mean(0)                         # (layers, heads)
    comp = []
    for head in heads:
        with switched_off(model, "mean", means, [head]):
            comp.append((y_attention(model, rep_prefixes, y).mean(0) - base).flatten().tolist())
    result["y_attention"] = base.flatten().tolist()
    result["composition"] = comp
    return result

# ---------- report ----------
def colour(v, scale):
    """A diverging cell colour: blue below 0, orange above, transparent at 0."""
    a = min(1.0, abs(v) / scale) if scale else 0.0
    hue = "59,91,219" if v < 0 else "240,140,0"
    return f"rgba({hue},{a * 0.85:.3f})"

def table(title, note, rows, row_names, col_names, fmt="{:+.2f}"):
    flat = [abs(v) for r in rows for v in r if v == v]
    scale = max(flat) if flat else 1.0
    head = "".join(f"<th>{c}</th>" for c in col_names)
    body = "".join(f"<tr><th class=r>{n}</th>" + "".join(
        f"<td style='background:{colour(v, scale)}'>{fmt.format(v)}</td>" if v == v else "<td></td>" for v in r)
        + "</tr>" for n, r in zip(row_names, rows))
    return f"<h2>{title}</h2><p>{note}</p><div class=s><table><tr><th></th>{head}</tr>{body}</table></div>"

def report(result, path):
    heads = result["heads"]
    parts = [table("Score with every head on",
                   "logit(y) - logit(g) at the last \"le\", averaged over 2 names and 20 frames. Above 0: the name ends.",
                   [result["clean"]], ["clean"], result["columns"])]
    for how in ("mean", "zero"):
        if f"heads_{how}" in result:
            parts.append(table(f"Head x condition ({how}-ablation)",
                               "The change in the score when one head is off. Blue: less y (more stutter); "
                               "orange: more y.", result[f"heads_{how}"], heads, result["columns"]))
    n = len(heads)
    layer = [int(h[1:].split("H")[0]) for h in heads]
    comp = [[v if layer[a] < layer[b] else float("nan") for b, v in enumerate(row)]
            for a, row in enumerate(result["composition"])]
    parts.append(table("Composition: head A off (row) x head B's attention to the earlier \"y\"s (column)",
                       "Mean-ablation, repeated names only. Only later layers are shown. Blue: A's output helps B "
                       "find the earlier name ends.", comp, heads, heads, "{:+.2f}"))
    parts.append(table("Each head's attention to the earlier \"y\"s, every head on", "Summed weight.",
                       [result["y_attention"][i:i + 6] for i in range(0, n, 6)],
                       [f"L{l}" for l in range(n // 6)], [f"H{h}" for h in range(6)], "{:.2f}"))
    page = ("<!doctype html><meta charset=utf-8><title>Name circuit</title><style>"
            ":root{color-scheme:light dark;--bg:#fff;--t:#1d1d1f;--m:#6b6b70}"
            "@media(prefers-color-scheme:dark){:root{--bg:#151517;--t:#ececee;--m:#9a9aa2}}"
            "body{background:var(--bg);color:var(--t);font:14px system-ui;margin:24px}"
            "p{color:var(--m)} .s{overflow-x:auto} table{border-collapse:collapse;font:11px ui-monospace,monospace}"
            "td,th{padding:2px 5px;text-align:right;white-space:nowrap} th{color:var(--m);font-weight:normal}"
            "</style><h1>Which heads decide how Glegley ends</h1>" + "".join(parts))
    path.write_text(page)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=MODEL, help=f"default {MODEL}")
    parser.add_argument("--device", default="cpu", choices=gpt.DEVICES, help="default cpu (short prefixes)")
    args = parser.parse_args()
    torch.manual_seed(0)
    model = gpt.load(args.device, args.checkpoint)
    tok = WordBPE.load(TOKENISER)
    result = run(model, tok)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.checkpoint.stem
    (OUT / f"{stem}.json").write_text(json.dumps(result))
    report(result, OUT / f"{stem}.html")
    print(f"saved {OUT / stem}.json and .html")
    clean = dict(zip(result["columns"], result["clean"]))
    print("score with every head on:", ", ".join(f"{c} {v:+.2f}" for c, v in clean.items()))
    for how in ("mean", "zero"):
        rows = result[f"heads_{how}"]
        top = sorted(range(len(rows)), key=lambda i: -max(abs(v) for v in rows[i]))[:8]
        print(f"largest effects ({how}):", ", ".join(f"{result['heads'][i]} {max(rows[i], key=abs):+.2f}" for i in top))
