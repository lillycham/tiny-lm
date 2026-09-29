"""OpenAI's GPT-2 small (124M), loaded into our own GPT class, to compare with ours.

GPT-2 small has the web model's shape: 12 layers, width 768, 12 heads, GELU, tied
embeddings, 1,024 tokens of context. Two things differ:
  - It has a bias on q, k and v (GPT's qkv_bias option).
  - Its tokeniser has 50,257 tokens, not 16,384. It is byte-level BPE with the same word
    split as our WordBPE, so GPT2Tokenizer turns its merges into a WordBPE.

The files, from huggingface.co/openai-community/gpt2, in checkpoints/gpt2/:
  model.safetensors   the weights (548 MB): a JSON header, then the raw numbers
  vocab.json          token string -> ID
  merges.txt          the 50,000 merges, in the order they were learned

    python -m tools.gpt2          # load it, then a few predictions and a sample
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from core import gpt
from core.word_bpe import WordBPE

DIR = Path("checkpoints/gpt2")
CONFIG = dict(block=1024, emb=768, heads=12, layers=12, dropout=0.0, vocab=50257, gelu=True, tied=True,
              qkv_bias=True)

# ---------- the weights ----------
def read_safetensors(path):
    """A .safetensors file -> {name: tensor}.

    The file is:
      - 8 bytes: the length of the header, as a little-endian unsigned integer
      - the header: JSON, {name: {"dtype": "F32", "shape": [768, 2304], "data_offsets": [start, end]}},
        plus a "__metadata__" entry that is not a tensor
      - the raw numbers of every tensor, one after another. The offsets count from the
        end of the header, not from the start of the file.
    """
    # TODO(Lilly): read it. Some tools:
    #   - struct.unpack("<Q", eight_bytes)[0]: "<" is little-endian, "Q" an unsigned 8-byte integer.
    #   - json.loads works on bytes.
    #   - torch.frombuffer(some_bytes, dtype=torch.float32) makes a flat tensor from raw bytes;
    #     .view(shape) gives it its shape. GPT-2 has only "F32".
    #   - frombuffer warns if the bytes can't be written to (bytes can't). A bytearray can.
    #   - Read the whole file once, then slice it. 548 MB fits in memory.
    raise NotImplementedError

def convert(hf, layers=CONFIG["layers"]):
    """GPT-2's {name: tensor} -> a state_dict for our GPT with qkv_bias and tied.

    GPT-2's names: wte, wpe, h.{i}.ln_1, h.{i}.attn.c_attn, h.{i}.attn.c_proj, h.{i}.ln_2,
    h.{i}.mlp.c_fc, h.{i}.mlp.c_proj, ln_f. Ours: print(GPT(...).state_dict().keys()).
    """
    # TODO(Lilly): for each of our names, which GPT-2 tensor, and transposed or not?
    #   - OpenAI's Conv1D stores (in, out) and computes x @ W + b. nn.Linear stores (out, in)
    #     and computes x @ W.T + b.
    #   - h.{i}.attn.bias is the causal mask, not a weight. Leave it out.
    #   - load_state_dict (strict) wants every key of our state_dict, so think about out.weight.
    raise NotImplementedError

def load(device="cpu", path=DIR / "model.safetensors"):
    model = gpt.GPT(**CONFIG)
    model.load_state_dict(convert(read_safetensors(path)))
    return model.to(device).eval()

# ---------- the tokeniser ----------
def bytes_to_unicode():
    """GPT-2's printable stand-in for each byte: {byte: character}.

    vocab.json and merges.txt are text, so every byte of a token needs a visible character.
    The printable bytes stand for themselves; the others (space, newline, control bytes)
    get characters from 256 up. So a space is "Ġ" (256 + 32) and " Paris" is "ĠParis".
    """
    keep = {*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)}
    chars, extra = {}, 0
    for b in range(256):
        if b in keep:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + extra)
            extra += 1
    return chars

class GPT2Tokenizer:
    """GPT-2's tokeniser as a WordBPE, with its IDs mapped to GPT-2's.

    A WordBPE numbers the 256 bytes 0-255 in byte order, then each merge 256, 257, ...
    in order. GPT-2 numbers its merges the same way, but its bytes in another order, so
    the two need a table between them. <|endoftext|> is 50,256 in both.
    """

    def __init__(self, dir=DIR):
        vocab = json.loads((dir / "vocab.json").read_text())
        byte_to_char = bytes_to_unicode()
        char_to_byte = {c: b for b, c in byte_to_char.items()}
        to_bytes = lambda token: bytes(char_to_byte[c] for c in token)
        ids = {bytes([b]): b for b in range(256)}               # token bytes -> our ID
        merges = {}
        for line in (dir / "merges.txt").read_text().splitlines()[1:]:     # line 0 is "#version"
            a, b = (to_bytes(t) for t in line.split(" "))
            merges[ids[a], ids[b]] = ids[a + b] = 256 + len(merges)
        self.bpe = WordBPE(merges)
        self.eot_id = self.bpe.eot_id
        ours = {v: k for k, v in ids.items()}
        self.to_gpt2 = [vocab["".join(byte_to_char[b] for b in ours[i])] for i in range(len(ours))]
        self.to_gpt2.append(vocab["<|endoftext|>"])
        self.from_gpt2 = {g: i for i, g in enumerate(self.to_gpt2)}

    def encode(self, s):
        return [self.to_gpt2[i] for i in self.bpe.encode(s)]

    def decode(self, ids):
        return self.bpe.decode([self.from_gpt2[i] for i in ids])

    def show(self, ids):
        return "|".join(self.decode([i]) for i in ids)

# ---------- a first look ----------
@torch.no_grad()
def top_next(model, tok, prompt, k=5):
    """The k likeliest next tokens after <|endoftext|> and prompt, with their probabilities."""
    ids = torch.tensor([[tok.eot_id] + tok.encode(prompt)], device=next(model.parameters()).device)
    p = F.softmax(model(ids, last_only=True)[0, -1].float(), dim=-1)
    top = p.topk(k)
    return [(tok.decode([i]), v) for v, i in zip(top.values.tolist(), top.indices.tolist())]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    args = parser.parse_args()
    model, tok = load(args.device), GPT2Tokenizer()
    print(f"GPT-2 small: {sum(p.numel() for p in model.parameters()):,} parameters on {args.device}")
    for prompt in ["The capital of France is", "Albert Einstein was born in", "1, 2, 3, 4,"]:
        print(f"\n{prompt!r}: " + "  ".join(f"{t!r} {p:.0%}" for t, p in top_next(model, tok, prompt)))
    torch.manual_seed(0)
    start = "In a shocking finding, scientists discovered"
    print(f"\n{start}" + gpt.generate(model, 60, start, 0.8, tok.encode, tok.decode, stop=tok.eot_id))
