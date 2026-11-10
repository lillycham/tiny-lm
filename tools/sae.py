"""A sparse autoencoder (SAE) on the story GPT's residual stream.

The residual stream after a block is a C-dimensional vector per token, and each
direction in it can mean many things at once. An SAE rewrites that vector as a
sum of a few directions out of many more (M = C x expansion), learned so that
each one, hopefully, means one thing:

    f  = TopK(ReLU(W_enc (x - b_dec) + b_enc))     M numbers, at most k of them not 0
    x̂ = W_dec f + b_dec                            back to C numbers

Column i of W_dec is feature i's direction. TopK keeps the k largest numbers of
each token and sets the rest to 0, so the sparsity is fixed at k: there is no L1
weight to tune, and the loss is only how well x̂ matches x.

    python -m tools.sae train --layer 3                       # the 384-wide story model, after block 3
    python -m tools.sae train --layer 3 --steps 200           # a quick try
    python -m tools.sae show checkpoints/sae/stories_gpt_384w6l_full_L3_x16_k32.pt
    python -m tools.sae show checkpoints/sae/stories_gpt_384w6l_full_L3_x16_k32.pt --features 12,345
    python -m tools.sae label checkpoints/sae/stories_gpt_384w6l_full_L3_x16_k32.pt

train also saves each feature's top contexts and how often it fires next to the SAE
(.examples.json), for show, label and the Features view in the playground
(python -m tools.playground). label writes the names you give features to
.labels.json, so they survive retraining and can be edited by hand.

There are 6,000 features per layer, which is more than one sitting, so a pass can be
handed out. slice writes the next features of the shortlist, their contexts and the
exact file to write names to, into a staging/ folder next to the SAE; merge folds what
comes back into .labels.json, and reports where two passes disagree instead of picking
for you:

    python -m tools.sae slice  checkpoints/sae/stories_gpt_384w6l_full_L5_x16_k32.pt --start 40 --n 80
    python -m tools.sae merge  checkpoints/sae/stories_gpt_384w6l_full_L5_x16_k32.pt --name features40-119

audit turns that round: it shows each name already written beside its top contexts in
different text, so a name can be checked without reusing the places it came from, and
report prints the verdicts. That is the cheap way to find out which of your own names
are wrong before writing more:

    python -m tools.sae audit  checkpoints/sae/stories_gpt_384w6l_full_L5_x16_k32.pt
    python -m tools.sae report checkpoints/sae/stories_gpt_384w6l_full_L5_x16_k32.pt
"""
import argparse
import json
import math
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import gpt
from core.word_bpe import TOKENISER, WordBPE
from stories.stories_gpt import load_tokens

MODEL = Path("checkpoints/stories/stories_gpt_384w6l_full.pt")
OUT = Path("checkpoints/sae")

# ---------- the SAE ----------
class SAE(nn.Module):
    """x (..., C) -> (x̂ (..., C), f (..., M)). See the module docstring.

    scale is a fixed number, set once from the data before training (set_scale): it
    makes the average |x| equal sqrt(C), so the learning rate does not depend on how
    large one layer's vectors happen to be. encode() and decode() apply it, so the
    caller always works with the model's own vectors."""

    def __init__(self, C, M, k):
        super().__init__()
        self.config = dict(C=C, M=M, k=k)
        self.k = k
        self.enc = nn.Linear(C, M)
        self.dec = nn.Linear(M, C)               # dec.weight is (C, M): one column per feature
        self.register_buffer("scale", torch.tensor(1.0))
        with torch.no_grad():
            # Start the decoder with random unit columns, and the encoder as its
            # transpose: then every feature starts out reading the same direction it
            # writes, which leaves fewer features that never fire ("dead").
            self.dec.weight.normal_()
            self.normalise_decoder()
            self.enc.weight.copy_(self.dec.weight.T)
            self.enc.bias.zero_()
            self.dec.bias.zero_()

    def encode(self, x):
        pre = F.relu(self.enc(x * self.scale - self.dec.bias))
        values, indices = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, values)

    def decode(self, f):
        return self.dec(f) / self.scale

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def normalise_decoder(self):
        """Make every decoder column length 1. Without this the SAE could make f smaller
        and the columns longer, and the sizes of the features would mean nothing."""
        self.dec.weight /= self.dec.weight.norm(dim=0, keepdim=True)

    @torch.no_grad()
    def set_scale(self, x):
        self.scale.fill_(math.sqrt(x.shape[-1]) / x.norm(dim=-1).mean().item())

def save(sae, path, **info):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": sae.config, "weights": sae.state_dict(), "info": info}, path)

def load(path, device="cpu"):
    saved = torch.load(path, map_location=device)
    sae = SAE(**saved["config"]).to(device)
    sae.load_state_dict(saved["weights"])
    return sae, saved["info"]

# ---------- activations ----------
def chunks(tokens, B, T, rng):
    """B random windows of T tokens from a 1-D array of token IDs, as a (B, T) tensor."""
    starts = rng.integers(0, len(tokens) - T, size=B)
    return torch.from_numpy(np.stack([np.asarray(tokens[s:s + T], dtype=np.int64) for s in starts]))

@torch.no_grad()
def residuals(model, layer, X):
    """The residual stream after block `layer` (0 = the first block) for X (B, T):
    (B, T, C). The model runs only as far as it needs to."""
    x = model.tok(X) + model.pos(torch.arange(X.shape[1], device=X.device))
    for block in model.blocks[:layer + 1]:
        x = block(x)
    return x

def stream(model, layer, tokens, B, T, device, seed=0):
    """An endless stream of (B * (T - 1), C) batches of residual vectors, shuffled.

    Position 0 is left out: every window's first token sees only itself, and its
    vector is much larger than the rest, so it would pull the SAE towards it."""
    rng = np.random.default_rng(seed)
    while True:
        X = chunks(tokens, B, T, rng).to(device)
        x = residuals(model, layer, X)[:, 1:].reshape(-1, model.config["emb"])
        yield x[torch.randperm(len(x), device=device)]

# ---------- evaluation ----------
@torch.no_grad()
def variance_explained(sae, x):
    """1 - (squared error) / (variance of x): 1 is perfect, 0 is no better than the mean vector."""
    x_hat, _ = sae(x)
    return 1 - ((x - x_hat) ** 2).sum().item() / ((x - x.mean(0)) ** 2).sum().item()

@torch.no_grad()
def spliced_loss(model, layer, X, Y, replace):
    """The model's loss on X -> Y, with block `layer`'s output x replaced by replace(x).

    A forward hook that returns a value replaces the module's output, so everything
    after the block sees the replacement."""
    handle = model.blocks[layer].register_forward_hook(lambda module, inputs, output: replace(output))
    try:
        return gpt.lm_loss(model, X, Y).item()
    finally:
        handle.remove()

@torch.no_grad()
def evaluate(model, sae, layer, tokens, device, B=16, T=256, seed=1):
    """The numbers to judge an SAE by, on held-out text.

      variance  how much of the residual stream x̂ keeps
      clean     the model's own loss
      sae       the loss with x̂ in place of x after the block
      mean      the loss with the mean vector in place of x (all the layer's information gone)
      recovered (mean - sae) / (mean - clean): the share of the loss that the SAE keeps
      dead      the share of features that did not fire on any of these tokens"""
    rng = np.random.default_rng(seed)
    XY = chunks(tokens, B, T + 1, rng).to(device)
    X, Y = XY[:, :-1], XY[:, 1:]
    x = residuals(model, layer, X)
    flat = x[:, 1:].reshape(-1, x.shape[-1])
    mean = flat.mean(0)
    _, f = sae(flat)
    clean = gpt.lm_loss(model, X, Y).item()
    with_sae = spliced_loss(model, layer, X, Y, lambda out: torch.cat([out[:, :1], sae(out[:, 1:])[0]], 1))
    with_mean = spliced_loss(model, layer, X, Y, lambda out: torch.cat([out[:, :1], mean.expand_as(out[:, 1:])], 1))
    return dict(variance=variance_explained(sae, flat), clean=clean, sae=with_sae, mean=with_mean,
                recovered=(with_mean - with_sae) / (with_mean - clean),
                dead=(f.amax(0) == 0).float().mean().item())

def format_eval(e):
    return (f"variance explained {e['variance']:.3f} | loss {e['clean']:.3f} clean, {e['sae']:.3f} with the SAE, "
            f"{e['mean']:.3f} with the mean | {e['recovered']:.1%} recovered | {e['dead']:.1%} dead")

# ---------- training ----------
def train(model, layer, expansion, k, steps, lr, B, device, log_every=250):
    C = model.config["emb"]
    sae = SAE(C, C * expansion, k).to(device)
    acts = stream(model, layer, load_tokens("train"), B, model.block, device)
    first = next(acts)
    sae.set_scale(first)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    # The learning rate stays at lr, then falls to 0 over the last fifth of the steps.
    decay_from = int(steps * 0.8)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 1.0 if s < decay_from else (steps - s) / (steps - decay_from))
    batch = 4096
    buffer, fired = first, torch.zeros(sae.config["M"], dtype=torch.bool, device=device)
    start = time.time()
    for step in range(steps):
        if len(buffer) < batch:
            buffer = torch.cat([buffer, next(acts)])
        x, buffer = buffer[:batch], buffer[batch:]
        x_hat, f = sae(x)
        # The loss is the error in the SAE's own units (x * scale), so its size does not depend on the layer.
        loss = ((x - x_hat) * sae.scale).pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        schedule.step()
        sae.normalise_decoder()
        fired |= f.amax(0) > 0
        if (step + 1) % log_every == 0 or step == steps - 1:
            print(f"step {step + 1:>6}: loss {loss.item():.4f}, {(~fired).float().mean().item():.1%} of features "
                  f"not fired since the last log ({time.time() - start:.0f} s)", flush=True)
            fired.zero_()
    return sae

# ---------- looking at features ----------
Scan = namedtuple("Scan", "values where windows counts seen")

@torch.no_grad()
def top_contexts(model, sae, layer, tokens, device, n_tokens=500_000, top=8, B=32, seed=2):
    """For every feature, the `top` tokens it fires on most over about n_tokens of text.

    Returns a Scan: values (top, M) are those tokens' activations, where (top, M) their
    places in windows (n, T), counts (M) how many tokens each feature fired on out of
    seen. The two give the firing rate, which labelling needs: a feature on 40% of
    tokens says something different from one on 0.01%, even with the same contexts."""
    rng = np.random.default_rng(seed)
    T, M = model.block, sae.config["M"]
    best = torch.full((top, M), -1.0, device=device)
    where = torch.zeros((top, M), dtype=torch.long, device=device)
    counts = torch.zeros(M, dtype=torch.long, device=device)
    seen, windows = 0, []
    for _ in range(max(1, n_tokens // (B * T))):
        X = chunks(tokens, B, T, rng).to(device)
        _, f = sae(residuals(model, layer, X))           # (B, T, M)
        f[:, 0] = -1.0                                   # (position 0: see stream())
        counts += (f > 0).sum(dim=(0, 1))
        seen += B * (T - 1)
        flat = f.reshape(-1, M)
        # Each token's place in all the windows so far: window number * T + position.
        ids = torch.arange(len(flat), device=device) + sum(len(w) for w in windows) * T
        values, rows = torch.cat([best, flat]).topk(top, dim=0)
        where = torch.cat([where, ids.unsqueeze(1).expand(-1, M)]).gather(0, rows)
        best = values
        windows.append(X.cpu())
    return Scan(best.cpu(), where.cpu(), torch.cat(windows), counts.cpu(), seen)

def snippets(tok, values, where, windows, T, before=24, after=3):
    """The top contexts as text: for each feature, a list of [value, left, token, right],
    strongest first, without the places where it did not fire."""
    out = []
    for i in range(values.shape[1]):
        rows = []
        for v, flat in zip(values[:, i].tolist(), where[:, i].tolist()):
            if v <= 0:
                break
            w, t = divmod(flat, T)
            ids = windows[w].tolist()
            rows.append([round(v, 2), tok.decode(ids[max(0, t - before):t]).replace("\n", " "),
                         tok.decode(ids[t:t + 1]).replace("\n", "⏎"),
                         tok.decode(ids[t + 1:t + 1 + after]).replace("\n", " ")])
        out.append(rows)
    return out

def examples_path(path):
    """Where the top contexts of an SAE are saved: next to it, as .examples.json."""
    return Path(path).with_suffix(".examples.json")

def write_examples(path, device, n_tokens, top=8):
    """Find every feature's top contexts and firing rate once and save them, so show,
    label and the playground can read them without running the model over half a
    million tokens again. Returns what it wrote, as load_examples does."""
    sae, info = load(path, device)
    model = gpt.load(device, Path(info["model"]))
    print(f"finding each feature's top contexts and firing rate over {n_tokens:,} tokens…", flush=True)
    scan = top_contexts(model, sae, info["layer"], load_tokens("val"), device, n_tokens, top)
    rows = snippets(WordBPE.load(TOKENISER), scan.values, scan.where, scan.windows, model.block)
    data = {"tokens": n_tokens, "seen": scan.seen, "counts": scan.counts.tolist(), "features": rows}
    examples_path(path).write_text(json.dumps(data, ensure_ascii=False))
    return data

def load_examples(path, device, n_tokens):
    """The saved examples, written again if there are none, they cover another amount
    of text, or they are from before firing rates were saved."""
    saved = examples_path(path)
    data = json.loads(saved.read_text()) if saved.exists() else None
    if data and data["tokens"] == n_tokens and "counts" in data:
        return data
    return write_examples(path, device, n_tokens)

def labels_path(path):
    """Where the names of an SAE's features are saved: next to it, as .labels.json."""
    return Path(path).with_suffix(".labels.json")

def load_labels(path):
    """{feature number as a string: label} for an SAE; empty if none were written."""
    saved = labels_path(path)
    return json.loads(saved.read_text()) if saved.exists() else {}

def write_labels(path, labels):
    """Save a whole set of labels, sorted, so a change is a diff of names and not of order."""
    labels_path(path).write_text(json.dumps(labels, ensure_ascii=False, indent=1, sort_keys=True) + "\n")

def set_label(path, index, label):
    """Save one feature's label next to the SAE, keeping the others. An empty label
    removes it, so a mistake can be undone. Returns all the labels as they now are."""
    labels = load_labels(path)
    key = str(int(index))
    if label.strip():
        labels[key] = label.strip()
    else:
        labels.pop(key, None)
    write_labels(path, labels)
    return labels

def feature_count(path):
    """How many features the SAE has, from the saved examples, or None if they are not
    written yet. Cheap: it reads a file instead of loading torch to ask the SAE."""
    saved = examples_path(path)
    return len(json.loads(saved.read_text())["features"]) if saved.exists() else None

def shortlist(data, n=20, min_rate=1e-4, max_rate=0.3, min_distinct=3):
    """The features to label first: alive, not on nearly every token, strongest first.

    Features whose top contexts are mostly the same token are left out, because they
    are just that token (or a position) and there is little to name; min_distinct sets
    how many different tokens they need, and 1 keeps them. Features that never fire and
    ones on more than max_rate of tokens (usually a common token) are left out too.

    An examples file from before counts were saved has no rates, so it ranks on peak
    and distinct tokens alone."""
    rows, counts, seen = data["features"], data.get("counts"), data.get("seen")
    scored = []
    for i, row in enumerate(rows):
        if not row:
            continue
        if counts and seen:
            rate = counts[i] / seen
            if not min_rate <= rate <= max_rate:
                continue
        if len({x[2] for x in row}) < min(min_distinct, len(row)):
            continue
        scored.append((row[0][0], i))
    return [i for _, i in sorted(scored, reverse=True)[:n]]

def slice_shortlist(data, start=0, n=80, min_rate=1e-4, max_rate=0.3, min_distinct=3):
    """`n` features from position `start` of the shortlist.

    The shortlist is a fixed ranking, so two passes given ranges that do not overlap
    name different features whatever order they work in, and neither has to know
    anything about the other. That is how labelling is split over several people or
    agents; merge_staged puts their answers together."""
    return shortlist(data, start + n, min_rate, max_rate, min_distinct)[start:]

def rate_text(data, i):
    """'fires on 0.42% of 499,968 tokens', or '' when the examples have no counts."""
    counts, seen = data.get("counts"), data.get("seen")
    return f"fires on {counts[i] / seen:.2%} of {seen:,} tokens" if counts else ""

def label_features(path, data, features):
    """Ask for a name for each feature in turn, after its top contexts and firing rate.
    Enter keeps the current label or skips, '-' clears it, 'q' stops. Each name is saved
    as soon as it is typed, so a stop halfway keeps the work."""
    labels = load_labels(path)
    for k, i in enumerate(features, 1):
        row = data["features"][i]
        current = labels.get(str(i), "")
        peak = f"peak {row[0][0]:.2f}  " if row else ""
        print(f"\nfeature {i}  [{k}/{len(features)}]  {peak}{rate_text(data, i)}"
              f"  label: {current or '(none)'}")
        for v, left, here, right in row:
            print(f"  {v:6.2f}  …{left}[{here}]{right}")
        if not row:
            print("  (never fired)")
        answer = input("label (Enter keeps, '-' clears, q quits): ").strip()
        if answer.lower() == "q":
            break
        if answer == "-":
            labels = set_label(path, i, "")
        elif answer:
            labels = set_label(path, i, answer)

# ---------- work written for someone else to do ----------
STYLE = """\
How to name: a short, lowercase phrase for what the feature reacts to in these
contexts, not what it does for the model ("hyphenated compounds", "dialogue verbs",
"animal words"). A token that a feature simply reproduces is a fine label ("the token
'ing'"). How often it fires matters: one on 30% of tokens is doing something broad.
When the contexts disagree, do not force it: say "unclear (a or b)". Leave a feature
out rather than guess at it.
"""

STAGING = "staging"

def staging_dir(path):
    """Where work handed to someone else is written and collected: a staging folder
    next to the SAE. Each file carries the name of its pass, so several passes can run
    at once without touching the labels already agreed on."""
    return Path(path).parent / STAGING

def work_path(path, name):
    """The readable list of features and contexts that a labelling pass works from."""
    return staging_dir(path) / f"{Path(path).stem}.{name}.work.txt"

def staged_path(path, name):
    """Where a labelling pass writes its names, before merge_staged folds them in."""
    return staging_dir(path) / f"{Path(path).stem}.{name}.labels.json"

def audit_path(path, name):
    """Where a check of names already written writes its verdicts."""
    return staging_dir(path) / f"{Path(path).stem}.{name}.verdicts.json"

def label_line(data, i, label):
    """'feature 12  peak 3.41  fires on 0.42% of 499,968 tokens  label: questions'."""
    row = data["features"][i]
    peak = f"peak {row[0][0]:.2f}" if row else "never fired"
    rate = rate_text(data, i)
    return f"feature {i}  {peak}  {rate + '  ' if rate else ''}label: {label or '(none)'}"

def context_lines(data, i):
    """One feature's top contexts, as the lines a person reads."""
    return [f"  {v:6.2f}  …{left}[{here}]{right}" for v, left, here, right in data["features"][i]]

def write_slice(path, data, features, name, out=None):
    """Write one pass's features, their contexts and rates, and where to put the names,
    for somebody else to label (see staging_dir). Returns the file written.

    The header is the whole briefing: what the labels are for, the exact file to write,
    its shape, and STYLE. A pass needs nothing else, and cannot lose the style guide."""
    out = Path(out) if out else work_path(path, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = load_labels(path)
    lines = [
        f"# Names for {len(features)} features of {path}",
        "#",
        "# Each line below is one place the feature fired, strongest first:",
        "#   activation  …text before[the token it fired on]text after",
        "#",
        f"# Write the names to {staged_path(path, name)}",
        "# as one JSON object from feature number to name, like",
        '#   {"12": "questions", "13": "past-tense verbs"}',
        "# The file does not exist yet, and nothing else belongs in it.",
        "#",
        *(("# " + line).rstrip() for line in STYLE.splitlines()),
        "",
    ]
    for i in features:
        lines.append(label_line(data, i, labels.get(str(i), "")))
        lines.extend(context_lines(data, i))
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

def write_audit(path, model, sae, layer, tokens, device, top=8, seed=5, offset=3_000_000,
                n_tokens=500_000, name="audit", tok=None):
    """Write each name already given to a feature beside its top contexts in *other*
    text, so the name can be checked against places it was not written from. Returns
    (file written, the features asked about).

    The text is a different stretch of the validation set at a different seed, which is
    as held out as a resample gets; a feature that never fires there is still listed,
    because that on its own is a verdict on a name that says it is common."""
    labels = load_labels(path)
    if not labels:
        raise ValueError(f"{path} has no names to check")
    features = sorted(int(k) for k in labels)
    scan = top_contexts(model, sae, layer, tokens[offset:], device, n_tokens, top, seed=seed)
    rows = snippets(tok or WordBPE.load(TOKENISER), scan.values, scan.where, scan.windows, model.block)
    data = {"seen": scan.seen, "counts": scan.counts.tolist(), "features": rows}
    out = work_path(path, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# A check of the {len(features)} names written for {path}",
        "#",
        "# These are fresh contexts: the same features, found over different text at a",
        "# different seed, from the places the names were written from. A name that only",
        "# fits its own contexts does not fit these.",
        "#",
        f"# Write the verdicts to {audit_path(path, name)}",
        "# as one JSON object from feature number to verdict, like",
        '#   {"12": {"verdict": "agrees"},',
        '#    "13": {"verdict": "disagrees", "was": "past tense", "label": "past-tense verbs"},',
        '#    "14": {"verdict": "partly", "was": "unclear", "label": "unclear (past tense or finished actions)"}}',
        '# verdict is "agrees", "partly" or "disagrees"; give "label" with anything but agrees,',
        '# and "was" is the name as written above, so a later change can be told from the record.',
        "# Judge the name as written, not against a better name you could think of:",
        '# "questions" agrees if these are questions, even if "wh- questions" is better.',
        "",
    ]
    for i in features:
        lines.append(label_line(data, i, labels[str(i)]))
        lines.extend(context_lines(data, i))
        if not rows[i]:
            lines.append("  (never fired in this text)")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out, features

def staged_files(path, suffix, names=()):
    """The staging files of one kind for this SAE, in name order; names keeps only the
    passes listed, so one answer can be merged before the others are in."""
    stem = Path(path).stem
    found = sorted(staging_dir(path).glob(f"{stem}.*.{suffix}"))
    return [f for f in found if not names or f.name[len(stem) + 1:].split(".")[0] in names]

Merge = namedtuple("Merge", "added same conflicts")

def merge_staged(path, names=(), force=False):
    """Fold the names written by labelling passes in staging/ into the SAE's own labels.

    names limits it to those passes; by default every one staged for this SAE is used.
    A feature that two passes named differently, or that already has another name, is a
    conflict: nothing is written for it and the report says who called it what. That is
    what staging is for: where agents disagree, a person decides. force takes the first
    pass in name order instead, for when that pass is trusted.

    Returns Merge(added, same, conflicts): the names now written, the features that
    already said the same, and {feature: [the names that competed, current first]}.
    Every file is read and checked before anything is written, so one pass's broken
    file cannot half-apply the others."""
    m = feature_count(path)
    candidates = {}
    for f in staged_files(path, "labels.json", names):
        try:
            staged = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{f} is not JSON: {e}") from e
        if not isinstance(staged, dict):
            raise ValueError(f"{f} should be one object from feature number to name")
        for key, name in staged.items():
            if not str(key).lstrip("-").isdigit():
                raise ValueError(f"{f}: {key!r} is not a feature number")
            if m is not None and not 0 <= int(key) < m:
                raise ValueError(f"{f}: feature {key} is outside 0..{m - 1}")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{f}: feature {key} has no name")
            candidates.setdefault(str(int(key)), []).append(name.strip())

    labels = load_labels(path)
    added, same, conflicts = {}, set(), {}
    for key, spoken in candidates.items():
        names_ = list(dict.fromkeys(spoken))                # (two passes agreeing is not a conflict)
        current = labels.get(key)
        if len(names_) == 1 and current in (None, names_[0]):
            if current is None:
                labels[key] = added[key] = names_[0]
            else:
                same.add(key)
        elif force and names_:
            if current != names_[0]:
                labels[key] = added[key] = names_[0]
            else:
                same.add(key)
        else:
            conflicts[key] = ([current] if current else []) + names_
    if added:
        write_labels(path, labels)
    return Merge(added, same, conflicts)

def read_audits(path, names=()):
    """{feature: (verdict, suggested name, name as checked)} from every check staged for
    the SAE. The suggested name is '' when the verdict is agrees, and a bare verdict
    string is accepted in place of the object, since that is easy to write by accident."""
    out = {}
    for f in staged_files(path, "verdicts.json", names):
        try:
            staged = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{f} is not JSON: {e}") from e
        for key, answer in staged.items():
            said = answer.get("verdict") if isinstance(answer, dict) else answer
            label = answer.get("label") if isinstance(answer, dict) else ""
            was = answer.get("was") if isinstance(answer, dict) else ""
            if said not in ("agrees", "partly", "disagrees"):
                raise ValueError(f"{f}: feature {key} has verdict {said!r}")
            out[str(int(key))] = (said, (label or "").strip(), (was or "").strip())
    return out

def outstanding(verdicts, labels):
    """The verdicts still worth reading. Agrees is done; so is a check whose suggested name
    was written, or whose name has changed since it was checked, because then the row is a
    record of an older name and not a complaint about this one."""
    return {i: v for i, v in verdicts.items()
            if v[0] != "agrees"
            and not (v[1] and labels.get(i) == v[1])
            and not (v[2] and labels.get(i) != v[2])}

def show(path, features, device, n_tokens):
    sae, info = load(path, device)
    data = load_examples(path, device, n_tokens)
    rows, labels = data["features"], load_labels(path)
    if not features:
        # By default, 20 features spread over how strongly they fire: the strongest first.
        alive = sorted((i for i, r in enumerate(rows) if r), key=lambda i: -rows[i][0][0])
        features = [alive[round(j * (len(alive) - 1) / 19)] for j in range(min(20, len(alive)))]
    dead = sum(not r for r in rows) / len(rows)
    print(f"{path}: layer {info['layer']}, {sae.config['M']} features, k = {sae.config['k']}; "
          f"{dead:.1%} never fired on {n_tokens:,} tokens, {len(labels)} labelled\n")
    for i in features:
        name = labels.get(str(i))
        print(f"feature {i}:" + (f" {name}" if name else ""))
        for v, left, here, right in rows[i]:
            print(f"  {v:6.2f}  …{left}[{here}]{right}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train an SAE on one layer, then evaluate it")
    gpt.add_device_option(t)
    t.add_argument("--checkpoint", type=Path, default=MODEL, help=f"the model (default {MODEL})")
    t.add_argument("--layer", type=int, default=3, help="after which block, from 0 (default 3)")
    t.add_argument("--expansion", type=int, default=16, help="features per residual dimension (default 16)")
    t.add_argument("--k", type=int, default=32, help="features on per token (default 32)")
    t.add_argument("--steps", type=int, default=5000, help="batches of 4096 vectors (default 5000)")
    t.add_argument("--lr", type=float, default=3e-4, help="default 3e-4")
    t.add_argument("--B", type=int, default=32, help="windows per model run (default 32)")
    s = sub.add_parser("show", help="print the texts that features fire on most")
    gpt.add_device_option(s)
    s.add_argument("sae", type=Path)
    s.add_argument("--features", default="", help="comma-separated feature numbers (default: 20 spread out)")
    s.add_argument("--tokens", type=int, default=500_000, help="how much validation text to search (default 500,000)")
    l = sub.add_parser("label", help="give features a name, one at a time, from their top contexts")
    gpt.add_device_option(l)
    l.add_argument("sae", type=Path)
    l.add_argument("--n", type=int, default=20, help="how many features to label (default 20)")
    l.add_argument("--features", default="", help="comma-separated feature numbers (default: the shortlist)")
    l.add_argument("--tokens", type=int, default=500_000, help="how much validation text to search (default 500,000)")
    l.add_argument("--min-rate", type=float, default=1e-4, help="ignore features rarer than this (default 0.01%%)")
    l.add_argument("--max-rate", type=float, default=0.3, help="ignore features commoner than this (default 30%%)")
    l.add_argument("--min-distinct", type=int, default=3,
                   help="different tokens the top contexts need, to skip token features (default 3; 1 keeps all)")
    sl = sub.add_parser("slice", help="write the next features of the shortlist for someone else to label")
    gpt.add_device_option(sl)
    sl.add_argument("sae", type=Path)
    sl.add_argument("--n", type=int, default=80, help="how many features (default 80)")
    sl.add_argument("--start", type=int, default=0, help="skip this many of the shortlist, to split the work (default 0)")
    sl.add_argument("--name", default="", help="what to call the pass (default features<start>-<end>)")
    sl.add_argument("--tokens", type=int, default=500_000, help="how much validation text to search (default 500,000)")
    sl.add_argument("--min-rate", type=float, default=1e-4, help="ignore features rarer than this (default 0.01%%)")
    sl.add_argument("--max-rate", type=float, default=0.3, help="ignore features commoner than this (default 30%%)")
    sl.add_argument("--min-distinct", type=int, default=3,
                    help="different tokens the top contexts need, to skip token features (default 3; 1 keeps all)")
    mg = sub.add_parser("merge", help="fold the labels written in staging/ into the SAE's own labels")
    mg.add_argument("sae", type=Path)
    mg.add_argument("--name", default="", help="only these passes, comma-separated (default: every one)")
    mg.add_argument("--force", action="store_true", help="let a staged name replace one already there")
    au = sub.add_parser("audit", help="show each named feature with fresh contexts, to check its name")
    gpt.add_device_option(au)
    au.add_argument("sae", type=Path)
    au.add_argument("--name", default="audit", help="what to call this check (default audit)")
    au.add_argument("--top", type=int, default=8, help="contexts per feature (default 8)")
    au.add_argument("--tokens", type=int, default=500_000, help="how much validation text to search (default 500,000)")
    au.add_argument("--seed", type=int, default=5, help="a different seed finds different places (default 5)")
    au.add_argument("--offset", type=int, default=3_000_000,
                    help="how far into the validation text to start, to keep it fresh (default 3,000,000)")
    r = sub.add_parser("report", help="print the verdicts of the checks written in staging/")
    r.add_argument("sae", type=Path)
    r.add_argument("--name", default="", help="only these checks, comma-separated (default: every one)")
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.cmd == "train":
        model = gpt.load(args.device, args.checkpoint)
        for p in model.parameters():
            p.requires_grad_(False)
        sae = train(model, args.layer, args.expansion, args.k, args.steps, args.lr, args.B, args.device)
        e = evaluate(model, sae, args.layer, load_tokens("val"), args.device)
        print(format_eval(e))
        path = OUT / f"{args.checkpoint.stem}_L{args.layer}_x{args.expansion}_k{args.k}.pt"
        save(sae, path, model=str(args.checkpoint), layer=args.layer, steps=args.steps, eval=e)
        write_examples(path, args.device, 500_000)
        print(f"saved {path} and {examples_path(path).name}")
    elif args.cmd == "label":
        data = load_examples(args.sae, args.device, args.tokens)
        features = [int(x) for x in args.features.split(",") if x] or shortlist(data, args.n, args.min_rate,
                                                                                 args.max_rate, args.min_distinct)
        label_features(args.sae, data, features)
    elif args.cmd == "slice":
        data = load_examples(args.sae, args.device, args.tokens)
        features = slice_shortlist(data, args.start, args.n, args.min_rate, args.max_rate, args.min_distinct)
        name = args.name or f"features{args.start}-{args.start + len(features) - 1}"
        out = write_slice(args.sae, data, features, name)
        print(f"wrote {out} ({len(features)} features)")
        print(f"the names go to {staged_path(args.sae, name)}")
    elif args.cmd == "merge":
        merged = merge_staged(args.sae, [x for x in args.name.split(",") if x], args.force)
        print(f"{len(merged.added)} new, {len(merged.same)} already the same, {len(merged.conflicts)} conflicts"
              + (f"; wrote {labels_path(args.sae)}" if merged.added else "; nothing written"))
        if merged.conflicts:
            print("nothing written for these (the name already there is first, if there was one):")
            for key, spoken in sorted(merged.conflicts.items(), key=lambda kv: int(kv[0])):
                print(f"  feature {key}  " + " / ".join(repr(n) for n in spoken))
    elif args.cmd == "audit":
        sae, info = load(args.sae, args.device)
        model = gpt.load(args.device, Path(info["model"]))
        out, features = write_audit(args.sae, model, sae, info["layer"], load_tokens("val"), args.device,
                                    top=args.top, seed=args.seed, offset=args.offset,
                                    n_tokens=args.tokens, name=args.name)
        print(f"wrote {out} ({len(features)} names checked against other text)")
        print(f"the verdicts go to {audit_path(args.sae, args.name)}")
    elif args.cmd == "report":
        verdicts = read_audits(args.sae, [x for x in args.name.split(",") if x])
        labels = load_labels(args.sae)
        said = [v for v, _, _ in verdicts.values()]
        print(f"{len(verdicts)} names checked: {said.count('agrees')} agree, "
              f"{said.count('partly')} partly, {said.count('disagrees')} disagree")
        left = outstanding(verdicts, labels)
        if len(left) < len(verdicts) - said.count("agrees"):
            print(f"{len(verdicts) - said.count('agrees') - len(left)} since dealt with")
        for verdict in ("disagrees", "partly"):
            rows = [(i, labels.get(i, "?"), s) for i, (v, s, _) in left.items() if v == verdict]
            if rows:
                print(f"\n{verdict}:")
                for i, name, suggestion in sorted(rows, key=lambda row: int(row[0])):
                    print(f"  feature {i}  {name!r}" + (f" -> {suggestion!r}" if suggestion else ""))
    else:
        features = [int(x) for x in args.features.split(",") if x]
        show(args.sae, features, args.device, args.tokens)
