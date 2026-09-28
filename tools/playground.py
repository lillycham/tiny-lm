"""A playground in the browser for the story and web models: write, compare and look inside.

    python -m tools.playground                  # then open http://localhost:8000
    python -m tools.playground --port 8080 --device cpu

- Write: a story from any model in checkpoints/ (one folder per part), token by token as it is made.
  The web models (web_gpt_*, Part 3) use their own tokeniser; write the start of a text.
- Compare: two models side by side, with the same request and the same seed.
- Tokens: the text split into its tokens, coloured by how likely each one was.
- Attention: click a token to see where each head looked from it.

Four kinds of request, as in training:
    story      plain text to continue, after <|endoftext|> as in pretraining
    name       "User: Tell me a story about Zork.\\nAssistant: "  (sft.py)
    instruct   "User: Tell me a story that uses the words ...\\nAssistant: "  (instruct_sft.py)
    chat       "User: <question>\\n\\n<context, if any>\\nAssistant: "  (chat_sft.py, web models),
               after a system prompt and the earlier turns of the conversation, if given

The server is Python's own http.server, so there is nothing new to install. It
listens on 127.0.0.1 only: the models run on this machine, for this machine.
"""
import argparse
import codecs
import json
import math
import re
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F

from web import chat_sft
from core import gpt
from stories import instruct_sft
from stories import sft
from web import web_data
from core import word_bpe
from core.word_bpe import EOT, WordBPE

ROOT = Path("checkpoints")
PAGE = Path(__file__).parent / "playground.html"
MAX_TOKENS = 1000
TOKENISERS = {"stories": word_bpe.TOKENISER, "web": web_data.TOKENISER}

# ---------- models ----------
def kind(path):
    """The kind of request a model was trained for, from its file name."""
    if "_chat" in path.name or "_system" in path.name:     # chat_sft.py, system_sft.py
        return "chat"
    if path.name.startswith("stories_sft"):
        return "name"
    if path.name.startswith("stories_instruct"):
        return "instruct"
    return "story"

def family(path):
    """Which tokeniser a model uses: "web" for web_gpt_* and models fine-tuned from one
       (stories_instruct_web_gpt_*), else "stories"."""
    return "web" if "web_gpt" in Path(path).name else "stories"

def step_of(path):
    """stories_gpt_384w6l_step1024.pt -> 1024, so snapshots sort by step, not as text."""
    m = re.search(r"_step(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1

def list_models(root=ROOT):
    """Every story and web model under root, then the snapshots (in any snapshots/ folder).
       Not resume files, and not the Shakespeare models, which use character tokens."""
    found = [p for pattern in ("stories_*.pt", "web_gpt_*.pt") for p in root.rglob(pattern)
             if not p.name.endswith(".resume.pt")]
    models = sorted(p for p in found if p.parent.name != "snapshots")
    snaps = sorted((p for p in found if p.parent.name == "snapshots"),
                   key=lambda p: (p.name.split("_step")[0], step_of(p)))
    return ([{"path": p.relative_to(root).as_posix(), "kind": kind(p), "family": family(p), "group": "models"}
             for p in models]
            + [{"path": p.relative_to(root).as_posix(), "kind": kind(p), "family": family(p), "group": "snapshots"}
               for p in snaps])

class Models:
    """Load each model on first use, and keep the most recent keep of them (2: enough
       for Compare). A web model is ~395 MB, so keeping every one ran out of memory.
       One lock for all model work: two requests at once (compare) take turns, a token
       at a time."""

    def __init__(self, root=ROOT, device="cpu", keep=2):
        self.root, self.device, self.keep = root.resolve(), device, keep
        self.cache, self.lock = OrderedDict(), threading.RLock()

    def get(self, name):
        path = (self.root / name).resolve()
        if path.suffix != ".pt" or self.root not in path.parents or not path.exists():
            raise ValueError(f"No model {name!r} in {self.root}")
        with self.lock:
            if path not in self.cache:
                while len(self.cache) >= self.keep:
                    self.cache.popitem(last=False)        # the least recently used
                self.cache[path] = gpt.load(self.device, path).eval()
                free_cache(torch.device(self.device))
            self.cache.move_to_end(path)
            return self.cache[path]

# ---------- requests ----------
def build_prompt(mode, text="", name="", words="", features=(), question="", context="", history=(), system=""):
    """(prompt text, whether it starts after <|endoftext|>) for one kind of request."""
    if mode == "story":
        return text, True
    if mode == "name":
        if not name.strip():
            raise ValueError("A name SFT request needs a name")
        return sft.prompt(name.strip()), False
    if mode == "instruct":
        ws = [w.strip() for w in words.split(",") if w.strip()]
        if not ws:
            raise ValueError("An instruct request needs at least one word")
        unknown = [f for f in features if f not in instruct_sft.FEATURES]
        if unknown:
            raise ValueError(f"Unknown features {unknown}; use {list(instruct_sft.FEATURES)}")
        return instruct_sft.prompt(instruct_sft.request(ws, list(features))), False
    if mode == "chat":
        if not question.strip():
            raise ValueError("A chat request needs a question")
        if not all(isinstance(t, (list, tuple)) and len(t) == 3 and all(isinstance(x, str) for x in t)
                   for t in history):
            raise ValueError("history must be a list of [question, context, answer]")
        return chat_sft.conversation(history, question, context, system), False
    raise ValueError(f"Unknown mode {mode!r}")

def piece(tok, i):
    """One token's own text. A token can hold half of a character; that half shows as �."""
    return EOT if i == tok.eot_id else tok.vocab[i].decode("utf-8", errors="replace")

def token_bytes(tok, i):
    return b"" if i == tok.eot_id else tok.vocab[i]

# ---------- sampling ----------
def free_cache(device):
    """Give the MPS allocator's unused blocks back.

    The context grows by one token a step, so every step's tensors have a new size.
    MPS keeps each freed block for a later tensor of the same size, which never
    comes: 1,000 tokens from a web model held ~15 GB, with only ~0.4 GB in use.
    Emptying the cache every token keeps it near 1 GB, at no measurable cost.
    """
    if device.type == "mps":
        torch.mps.empty_cache()

@torch.no_grad()
def sample(model, tok, prompt_ids, n, temperature=0.8, seed=0, top=5, lock=None):
    """Yield one dict per new token, up to n tokens, ending early at <|endoftext|>.

    Each dict has the token's id, its piece, "text" (the new characters it completes:
    empty while a character is still half made), its probability p, the entropy H of
    the whole distribution (in nats: 0 = certain), and the top alternatives. p is the model's own probability, at temperature 1, whatever the
    temperature used to choose. Temperature 0 always takes the most likely token.
    """
    lock = lock or threading.RLock()
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    text = codecs.getincrementaldecoder("utf-8")(errors="replace")
    ids = list(prompt_ids)
    for _ in range(n):
        with lock:
            x = torch.tensor([ids[-model.block:]], device=device)
            logits = model(x, last_only=True)[0, -1].float().cpu()
            free_cache(device)
        p = F.softmax(logits, dim=-1)
        if temperature > 0:
            nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1, generator=g).item()
        else:
            nxt = logits.argmax().item()
        top_p, top_i = p.topk(top)
        yield {"id": nxt, "piece": piece(tok, nxt), "text": text.decode(token_bytes(tok, nxt)),
               "p": p[nxt].item(), "H": entropy(p),
               "top": [[piece(tok, i), q] for i, q in zip(top_i.tolist(), top_p.tolist())]}
        if nxt == tok.eot_id:
            return
        ids.append(nxt)

def entropy(p):
    """-sum p log p, in nats: 0 when one token has all the probability, ln(V) when all are equal."""
    return -(p * p.clamp_min(1e-12).log()).sum().item()

@torch.no_grad()
def attention_from(model, ids, t, lock=None):
    """Where every head looked from position t: (offset, [layer][head][weights]).

    Only the last block tokens up to t are in view, as in generation; offset is the
    position of the first of them. Each list of weights sums to 1.
    """
    lock = lock or threading.RLock()
    if not 0 <= t < len(ids):
        raise ValueError(f"Token {t} is outside the {len(ids)} tokens")
    start = max(0, t + 1 - model.block)
    device = next(model.parameters()).device
    rows, handles = {}, []

    def last_row(i):
        # Only the row the page shows: t's query against every key. interp.Recorder keeps
        # every row, T x T per head, ~600 MB for a web model at 1,024 tokens. The last
        # row may attend to every position, so it needs no causal mask.
        def hook(module, inputs, output):
            x = inputs[0]
            q, k, _ = module.qkv(x).split(x.shape[-1], dim=-1)
            q = module.split_heads(q)[:, :, -1:]                      # (1, H, 1, hs)
            k = module.smear_keys(module.split_heads(k))              # (1, H, T, hs)
            rows[i] = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)[0, :, 0]
        return hook

    with lock:
        try:
            handles = [b.attn.register_forward_hook(last_row(i)) for i, b in enumerate(model.blocks)]
            model(torch.tensor([ids[start:t + 1]], device=device), last_only=True)
        finally:
            for h in handles:
                h.remove()
        weights = [[[round(w, 4) for w in rows[i][h].tolist()] for h in range(rows[i].shape[0])]
                   for i in range(len(model.blocks))]
        free_cache(device)
    return start, weights

# ---------- visualisations ----------
# Each works on a window of the text, start to end, seen with the context before it,
# up to the model's block of tokens. The page asks for a few dozen tokens: a whole
# 1,024-token logit lens would be 13 x 1,024 x 16,384 logits, ~870 MB.

def view(model, ids, start, end):
    """(first position the model sees, start, end), checked: the window must end in ids,
       and start no earlier than the block of tokens that ends at end."""
    if not (0 < end <= len(ids)) or start >= end:
        raise ValueError(f"No window {start}-{end} in {len(ids)} tokens")
    first = max(0, end - model.block)
    return first, max(start, first), end

def query_key(module, x):
    """A CausalSelfAttention's q and k, (B, H, T, hs), as its forward makes them."""
    q, k, _ = module.qkv(x).split(x.shape[-1], dim=-1)
    return module.split_heads(q), module.smear_keys(module.split_heads(k))

def causal_weights_for(q, k, rows):
    """Softmax attention weights of query positions rows (a slice of 0..T) against every
    key: (B, H, len(rows), T). A key after its query gets 0."""
    T = k.shape[-2]
    scores = q[:, :, rows] @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    pos = torch.arange(T, device=q.device)
    scores = scores.masked_fill(pos[None, :] > pos[rows][:, None], float("-inf"))
    return F.softmax(scores, dim=-1)

@torch.no_grad()
def logit_lens(model, tok, ids, start, end, lock=None):
    """What each layer would predict after each position start..end-1: the residual
    stream there, through the final LayerNorm and output layer, as if the model stopped.

    Returns (start, rows): one row per stream (0 = the embeddings, i = after block i),
    each {"top": top pieces, "p": their probabilities, "next": the probability of the
    token that really came next, or None at the last position}. The last row is the
    model's own prediction.
    """
    lock = lock or threading.RLock()
    first, start, end = view(model, ids, start, end)
    device = next(model.parameters()).device
    streams, handles = {}, []

    def save(i):
        def hook(module, inputs, output):
            streams[i] = output[0, start - first:].detach()
        return hook

    with lock:
        try:
            handles.append(model.drop.register_forward_hook(save(0)))
            handles += [b.register_forward_hook(save(i + 1)) for i, b in enumerate(model.blocks)]
            model(torch.tensor([ids[first:end]], device=device), last_only=True)
        finally:
            for h in handles:
                h.remove()
        nxt = ids[start + 1:end + 1]
        rows = []
        for i in range(len(model.blocks) + 1):
            p = F.softmax(model.out(model.ln(streams[i])).float(), dim=-1)      # (W, V)
            top_p, top_i = p.max(dim=-1)
            rows.append({"top": [piece(tok, j) for j in top_i.tolist()], "p": [round(x, 4) for x in top_p.tolist()],
                         "next": [round(p[j, n].item(), 4) for j, n in enumerate(nxt)] + [None] * (end - start - len(nxt))})
        free_cache(device)
    return start, rows

@torch.no_grad()
def attention_matrix(model, ids, layer, head, start, end, lock=None):
    """One head's attention weights among positions start..end-1: (start, matrix, outside),
    matrix[r][c] the weight from start + r to start + c, and outside[r] the weight row r
    gives to earlier tokens outside the window."""
    lock = lock or threading.RLock()
    if not (0 <= layer < len(model.blocks) and 0 <= head < model.blocks[layer].attn.n_head):
        raise ValueError(f"No layer {layer} head {head}")
    first, start, end = view(model, ids, start, end)
    device = next(model.parameters()).device
    got = {}

    def hook(module, inputs, output):
        q, k = query_key(module, inputs[0])
        got["w"] = causal_weights_for(q[:, head:head + 1], k[:, head:head + 1], slice(start - first, None))[0, 0]

    with lock:
        h = model.blocks[layer].attn.register_forward_hook(hook)
        try:
            model(torch.tensor([ids[first:end]], device=device), last_only=True)
        finally:
            h.remove()
        w = got["w"]                                                          # (W, T)
        inside = w[:, start - first:]
        matrix = [[round(x, 4) for x in row] for row in inside.tolist()]
        outside = [round(x, 4) for x in (1 - inside.sum(dim=-1)).clamp_min(0).tolist()]
        free_cache(device)
    return start, matrix, outside

@torch.no_grad()
def head_map(model, ids, window=256, lock=None):
    """Each head's typical behaviour on the last window tokens, as (layers, heads) grids:
    prev: mean attention to the token just before; first: to the first token in view (the
    attention sink); induction: where the token appeared earlier, to the token that came
    after that earlier copy (None if no token repeats). Returns (offset, grids, repeats)."""
    lock = lock or threading.RLock()
    offset = max(0, len(ids) - min(window, model.block))
    x = ids[offset:]
    if len(x) < 2:
        raise ValueError("The head map needs at least 2 tokens")
    # For each position t, the token after the last earlier copy of x[t], if any.
    last, after = {}, []
    for t, token in enumerate(x):
        after.append(last[token] + 1 if token in last and last[token] + 1 < t else None)
        last[token] = t
    rep_t = [t for t, a in enumerate(after) if a is not None]
    device = next(model.parameters()).device
    grids = {"prev": [], "first": [], "induction": []}
    handles = []

    def hook(module, inputs, output):
        q, k = query_key(module, inputs[0])
        w = causal_weights_for(q, k, slice(None))[0]                            # (H, T, T)
        t = torch.arange(1, w.shape[-1], device=w.device)
        grids["prev"].append(w[:, t, t - 1].mean(dim=-1).tolist())
        grids["first"].append(w[:, t, 0].mean(dim=-1).tolist())
        if rep_t:
            rt = torch.tensor(rep_t, device=w.device)
            at = torch.tensor([after[i] for i in rep_t], device=w.device)
            grids["induction"].append(w[:, rt, at].mean(dim=-1).tolist())
        else:
            grids["induction"].append([None] * w.shape[0])

    with lock:
        try:
            handles = [b.attn.register_forward_hook(hook) for b in model.blocks]
            model(torch.tensor([x], device=device), last_only=True)
        finally:
            for h in handles:
                h.remove()
        free_cache(device)
    rounded = {k: [[None if v is None else round(v, 4) for v in row] for row in g] for k, g in grids.items()}
    return offset, rounded, len(rep_t)

# ---------- the server ----------
class Handler(BaseHTTPRequestHandler):
    models = None       # set by make_server
    toks = None         # {family: WordBPE}

    def log_message(self, format, *args):     # quiet: no line for every request
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def do_GET(self):
        if self.path == "/":
            body = PAGE.read_bytes()       # read each time, so edits show on reload
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/models":
            self.send_json({"models": list_models(self.models.root), "device": self.models.device,
                            "features": list(instruct_sft.FEATURES)})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            req = self.read_json()
            if self.path == "/api/generate":
                return self.generate(req)
            if self.path == "/api/attention":
                model = self.models.get(req["model"])
                offset, weights = attention_from(model, [int(i) for i in req["ids"]], int(req["t"]), self.models.lock)
                return self.send_json({"offset": offset, "weights": weights})
            if self.path in ("/api/lens", "/api/matrix", "/api/headmap"):
                model, tok = self.models.get(req["model"]), self.toks[family(req["model"])]
                ids = [int(i) for i in req["ids"]]
                if self.path == "/api/lens":
                    start, rows = logit_lens(model, tok, ids, int(req["start"]), int(req["end"]), self.models.lock)
                    return self.send_json({"start": start, "rows": rows})
                if self.path == "/api/matrix":
                    start, matrix, outside = attention_matrix(model, ids, int(req["layer"]), int(req["head"]),
                                                              int(req["start"]), int(req["end"]), self.models.lock)
                    return self.send_json({"start": start, "matrix": matrix, "outside": outside})
                offset, grids, repeats = head_map(model, ids, lock=self.models.lock)
                return self.send_json({"offset": offset, **grids, "repeats": repeats})
            self.send_json({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError) as e:
            self.send_json({"error": str(e)}, 400)

    def generate(self, req):
        """Stream one JSON object per line: start, then one per token, then end."""
        model, tok = self.models.get(req["model"]), self.toks[family(req["model"])]
        prompt, after_eot = build_prompt(req.get("mode", "story"), req.get("text", ""), req.get("name", ""),
                                         req.get("words", ""), req.get("features", []),
                                         req.get("question", ""), req.get("context", ""), req.get("history", []),
                                         req.get("system", ""))
        after_eot = req.get("after_eot", after_eot)
        prompt_ids = ([tok.eot_id] if after_eot else []) + tok.encode(prompt)
        if not prompt_ids:
            raise ValueError("Give some text to start from")
        n = max(1, min(int(req.get("max_tokens", 300)), MAX_TOKENS))
        temperature, seed = float(req.get("temperature", 0.8)), int(req.get("seed", 0))

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        write = lambda obj: (self.wfile.write((json.dumps(obj) + "\n").encode()), self.wfile.flush())
        c = model.config
        try:
            write({"type": "start", "prompt": prompt, "prompt_ids": prompt_ids,
                   "prompt_pieces": [piece(tok, i) for i in prompt_ids],
                   "params": sum(p.numel() for p in model.parameters()),
                   "layers": c["layers"], "heads": c["heads"],
                   "shape": f"{c['layers']} layers x {c['heads']} heads, width {c['emb']}"
                            + (", smeared keys" if c.get("smear") else "")})
            t, count, reason = time.time(), 0, "length"
            for token in sample(model, tok, prompt_ids, n, temperature, seed, lock=self.models.lock):
                count += 1
                if token["id"] == tok.eot_id:
                    reason = "eot"
                write({"type": "token", **token})
            write({"type": "end", "reason": reason, "tokens": count, "seconds": round(time.time() - t, 2)})
        except (BrokenPipeError, ConnectionResetError):
            pass        # the page pressed Stop, or closed: stop making tokens
        except Exception as e:
            # The 200 and some tokens are already sent, so a new HTTP response can't
            # follow. Say what went wrong as one more line of the stream.
            write({"type": "error", "error": f"{type(e).__name__}: {e}"})

def make_server(root=ROOT, device="cpu", port=8000, tokenisers=TOKENISERS):
    """tokenisers: {family: path}. A family whose tokeniser file is missing is left out."""
    toks = {f: WordBPE.load(Path(p)) for f, p in tokenisers.items() if Path(p).exists()}
    handler = type("PlaygroundHandler", (Handler,), {"models": Models(root, device), "toks": toks})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    parser.add_argument("--port", type=int, default=8000, help="default 8000")
    args = parser.parse_args()
    server = make_server(device=args.device, port=args.port)
    print(f"{len(list_models())} models on {args.device}. Open http://localhost:{server.server_port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
