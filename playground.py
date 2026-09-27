"""A playground in the browser for the story models: write, compare and look inside.

    python playground.py                  # then open http://localhost:8000
    python playground.py --port 8080 --device cpu

- Write: a story from any model in checkpoints/, token by token as it is made.
- Compare: two models side by side, with the same request and the same seed.
- Tokens: the text split into its tokens, coloured by how likely each one was.
- Attention: click a token to see where each head looked from it.

Three kinds of request, as in training:
    story      plain text to continue, after <|endoftext|> as in pretraining
    name       "User: Tell me a story about Zork.\\nAssistant: "  (sft.py)
    instruct   "User: Tell me a story that uses the words ...\\nAssistant: "  (instruct_sft.py)

The server is Python's own http.server, so there is nothing new to install. It
listens on 127.0.0.1 only: the models run on this machine, for this machine.
"""
import argparse
import codecs
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F

import gpt
import instruct_sft
import sft
from interp import Recorder
from word_bpe import EOT, EOT_ID, TOKENISER, WordBPE

ROOT = Path("checkpoints")
PAGE = Path(__file__).parent / "playground.html"
MAX_TOKENS = 1000

# ---------- models ----------
def kind(path):
    """The kind of request a model was trained for, from its file name."""
    if path.name.startswith("stories_sft"):
        return "name"
    if path.name.startswith("stories_instruct"):
        return "instruct"
    return "story"

def step_of(path):
    """stories_gpt_384w6l_step1024.pt -> 1024, so snapshots sort by step, not as text."""
    m = re.search(r"_step(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1

def list_models(root=ROOT):
    """Every story model in root, then the snapshots in root/snapshots. Not resume files,
       and not the Shakespeare models, which use other tokenisers."""
    models = sorted(p for p in root.glob("stories_*.pt") if not p.name.endswith(".resume.pt"))
    snaps = sorted((root / "snapshots").glob("stories_*.pt"), key=lambda p: (p.name.split("_step")[0], step_of(p)))
    return ([{"path": p.relative_to(root).as_posix(), "kind": kind(p), "group": "models"} for p in models]
            + [{"path": p.relative_to(root).as_posix(), "kind": kind(p), "group": "snapshots"} for p in snaps])

class Models:
    """Load each model once, on first use, and keep it. One lock for all model work:
       two requests at once (compare) take turns, a token at a time."""

    def __init__(self, root=ROOT, device="cpu"):
        self.root, self.device = root.resolve(), device
        self.cache, self.lock = {}, threading.RLock()

    def get(self, name):
        path = (self.root / name).resolve()
        if path.suffix != ".pt" or self.root not in path.parents or not path.exists():
            raise ValueError(f"No model {name!r} in {self.root}")
        with self.lock:
            if path not in self.cache:
                self.cache[path] = gpt.load(self.device, path).eval()
            return self.cache[path]

# ---------- requests ----------
def build_prompt(mode, text="", name="", words="", features=()):
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
    raise ValueError(f"Unknown mode {mode!r}")

def piece(tok, i):
    """One token's own text. A token can hold half of a character; that half shows as �."""
    return EOT if i == EOT_ID else tok.vocab[i].decode("utf-8", errors="replace")

def token_bytes(tok, i):
    return b"" if i == EOT_ID else tok.vocab[i]

# ---------- sampling ----------
@torch.no_grad()
def sample(model, tok, prompt_ids, n, temperature=0.8, seed=0, top=5, lock=None):
    """Yield one dict per new token, up to n tokens, ending early at <|endoftext|>.

    Each dict has the token's id, its piece, "text" (the new characters it completes:
    empty while a character is still half made), its probability p, and the top
    alternatives. p is the model's own probability, at temperature 1, whatever the
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
            logits = model(x)[0, -1].float().cpu()
        p = F.softmax(logits, dim=-1)
        if temperature > 0:
            nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1, generator=g).item()
        else:
            nxt = logits.argmax().item()
        top_p, top_i = p.topk(top)
        yield {"id": nxt, "piece": piece(tok, nxt), "text": text.decode(token_bytes(tok, nxt)),
               "p": p[nxt].item(), "top": [[piece(tok, i), q] for i, q in zip(top_i.tolist(), top_p.tolist())]}
        if nxt == EOT_ID:
            return
        ids.append(nxt)

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
    with lock, Recorder(model) as rec:
        model(torch.tensor([ids[start:t + 1]], device=device))
    weights = [[[round(w, 4) for w in rec.attn[i][0, h, -1].tolist()] for h in range(rec.attn[i].shape[1])]
               for i in range(len(model.blocks))]
    return start, weights

# ---------- the server ----------
class Handler(BaseHTTPRequestHandler):
    models = None       # set by make_server
    tok = None

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
            self.send_json({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError) as e:
            self.send_json({"error": str(e)}, 400)

    def generate(self, req):
        """Stream one JSON object per line: start, then one per token, then end."""
        model = self.models.get(req["model"])
        prompt, after_eot = build_prompt(req.get("mode", "story"), req.get("text", ""), req.get("name", ""),
                                         req.get("words", ""), req.get("features", []))
        after_eot = req.get("after_eot", after_eot)
        prompt_ids = ([EOT_ID] if after_eot else []) + self.tok.encode(prompt)
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
                   "prompt_pieces": [piece(self.tok, i) for i in prompt_ids],
                   "params": sum(p.numel() for p in model.parameters()),
                   "shape": f"{c['layers']} layers x {c['heads']} heads, width {c['emb']}"
                            + (", smeared keys" if c.get("smear") else "")})
            t, count, reason = time.time(), 0, "length"
            for token in sample(model, self.tok, prompt_ids, n, temperature, seed, lock=self.models.lock):
                count += 1
                if token["id"] == EOT_ID:
                    reason = "eot"
                write({"type": "token", **token})
            write({"type": "end", "reason": reason, "tokens": count, "seconds": round(time.time() - t, 2)})
        except (BrokenPipeError, ConnectionResetError):
            pass        # the page pressed Stop, or closed: stop making tokens

def make_server(root=ROOT, device="cpu", port=8000, tokeniser=TOKENISER):
    handler = type("PlaygroundHandler", (Handler,), {"models": Models(root, device), "tok": WordBPE.load(tokeniser)})
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
