"""Export a checkpoint as Hugging Face files, to upload to the Hub.

Our .pt files are pickles. The Hub prefers .safetensors, which holds only numbers. The story
and web models (GELU, tied embeddings, no smearing) are GPT-2 shaped, so they become a
standard GPT2LMHeadModel and load with plain transformers, with no code from this repo:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("user/repo")

Three things change on the way:
  - OpenAI's Conv1D stores (in, out) and nn.Linear stores (out, in), so every matrix is
    transposed.
  - GPT-2 has a bias on q, k and v. Ours has none, so the export writes zeros.
  - The tokeniser becomes a tokenizer.json: the same merges, written as byte-level BPE, with
    our word-split regex in front. check() compares it with WordBPE on real text.

The other models (ReLU, untied output with a bias, smeared keys) can't be a GPT2LMHeadModel.
--native saves their weights as safetensors, with our sizes in config.json, for
`gpt.GPT(**config)`.

    python -m tools.hf_export checkpoints/web/web_gpt_768w12l.pt --tokenizer checkpoints/web/web_bpe.json --out hf/web-98m --check
    python -m tools.hf_export checkpoints/shakespeare/gpt_bpe.pt --native --out hf/shakespeare
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from core import gpt
from core.word_bpe import EOT, WORD, WordBPE
from tools.gpt2 import bytes_to_unicode

# ---------- the weights ----------
def check_gpt2_shaped(config):
    """Raise unless a checkpoint's sizes fit a GPT2LMHeadModel."""
    wrong = [name for name, ok in [("gelu", config.get("gelu")), ("tied", config.get("tied")),
                                   ("no smear", not config.get("smear"))] if not ok]
    if wrong:
        raise ValueError(f"not GPT-2 shaped (needs {', '.join(wrong)}): use --native")

def to_hf_weights(weights, layers):
    """Our state_dict -> a GPT2LMHeadModel state_dict, without lm_head (it is tied to wte)."""
    out = {
        "transformer.wte.weight": weights["tok.weight"],
        "transformer.wpe.weight": weights["pos.weight"],
        "transformer.ln_f.weight": weights["ln.weight"],
        "transformer.ln_f.bias": weights["ln.bias"],
    }
    for i in range(layers):
        ours, hf = f"blocks.{i}.", f"transformer.h.{i}."
        for a, b in [("ln1", "ln_1"), ("ln2", "ln_2")]:
            out[f"{hf}{b}.weight"] = weights[f"{ours}{a}.weight"]
            out[f"{hf}{b}.bias"] = weights[f"{ours}{a}.bias"]
        qkv = weights[f"{ours}attn.qkv.weight"]
        out[f"{hf}attn.c_attn.weight"] = qkv.t()
        out[f"{hf}attn.c_attn.bias"] = weights.get(f"{ours}attn.qkv.bias", torch.zeros(qkv.shape[0]))
        for a, b in [("attn.proj", "attn.c_proj"), ("ffwd.net.0", "mlp.c_fc"), ("ffwd.net.2", "mlp.c_proj")]:
            out[f"{hf}{b}.weight"] = weights[f"{ours}{a}.weight"].t()
            out[f"{hf}{b}.bias"] = weights[f"{ours}{a}.bias"]
    return {k: v.detach().to(torch.float32).contiguous().clone() for k, v in out.items()}

def hf_config(config, eot_id):
    """Our sizes -> the config.json of a GPT2LMHeadModel."""
    return {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": config["vocab"],
        "n_positions": config["block"],
        "n_ctx": config["block"],
        "n_embd": config["emb"],
        "n_layer": config["layers"],
        "n_head": config["heads"],
        "n_inner": 4 * config["emb"],
        "activation_function": "gelu_new",     # the tanh GELU: nn.GELU(approximate="tanh")
        "layer_norm_epsilon": 1e-5,
        "resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0,
        "initializer_range": 0.02,
        "tie_word_embeddings": True,
        "bos_token_id": eot_id,
        "eos_token_id": eot_id,
    }

# ---------- the tokeniser ----------
def tokenizer_json(tok):
    """A WordBPE -> the contents of a tokenizer.json (byte-level BPE).

    Each token's bytes are written with GPT-2's printable stand-ins (tools.gpt2.bytes_to_unicode).
    A merge is "left right", ranked by the order it was learned.
    """
    to_char = bytes_to_unicode()
    name = lambda i: "".join(to_char[b] for b in tok.vocab[i])
    vocab = {name(i): i for i in range(tok.eot_id)}
    if len(vocab) != tok.eot_id:
        raise ValueError("two merges make the same bytes, so one token string has two IDs")
    merges = [f"{name(a)} {name(b)}" for a, b in tok.merges]
    return {
        "version": "1.0",
        "truncation": None, "padding": None,
        "added_tokens": [{"id": tok.eot_id, "content": EOT, "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": False, "special": True}],
        "normalizer": None,
        # Our split first (WORD, the same pattern as core.word_bpe), then bytes as characters.
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": WORD.pattern}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        ]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                  "ignore_merges": False, "vocab": vocab, "merges": merges},
    }

def tokenizer_config(block):
    return {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": block,
            "bos_token": EOT, "eos_token": EOT, "clean_up_tokenization_spaces": False}

# ---------- writing and checking ----------
def export(checkpoint, out, tokenizer=None, native=False):
    """Write the Hugging Face files for a checkpoint into out. Returns the sizes."""
    saved = torch.load(checkpoint, map_location="cpu")
    config, weights = saved["config"], saved["weights"]
    out.mkdir(parents=True, exist_ok=True)
    if native:
        # A tied output shares the token matrix, and safetensors keeps one copy of it.
        state = {k: v.detach().contiguous().clone() for k, v in weights.items()
                 if not (config.get("tied") and k == "out.weight")}
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    else:
        check_gpt2_shaped(config)
        if tokenizer is None:
            raise ValueError("--tokenizer is needed, or use --native")
        tok = WordBPE.load(tokenizer)
        if tok.eot_id + 1 != config["vocab"]:
            raise ValueError(f"tokeniser has {tok.eot_id + 1} tokens, the model {config['vocab']}")
        state = to_hf_weights(weights, config["layers"])
        (out / "config.json").write_text(json.dumps(hf_config(config, tok.eot_id), indent=2) + "\n")
        (out / "generation_config.json").write_text(json.dumps(
            {"bos_token_id": tok.eot_id, "eos_token_id": tok.eot_id}, indent=2) + "\n")
        (out / "tokenizer.json").write_text(json.dumps(tokenizer_json(tok), ensure_ascii=False) + "\n")
        (out / "tokenizer_config.json").write_text(json.dumps(tokenizer_config(config["block"]), indent=2) + "\n")
    save_file(state, out / "model.safetensors", metadata={"format": "pt"})
    return config

@torch.no_grad()
def check(checkpoint, out, tokenizer, texts):
    """Load the exported files with transformers. Returns (the largest logit difference to our
    model, how many texts the tokenisers disagree on)."""
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    ours = gpt.load("cpu", checkpoint)
    theirs = AutoModelForCausalLM.from_pretrained(out).eval()
    X = torch.randint(0, ours.config["vocab"], (2, min(ours.block, 64)))
    diff = (ours(X) - theirs(X).logits).abs().max().item()
    wordbpe = WordBPE.load(tokenizer)
    hf_tok = PreTrainedTokenizerFast(tokenizer_file=str(out / "tokenizer.json"))
    # A literal <|endoftext|> is one special token for Hugging Face, and plain text for WordBPE.
    texts = [t for t in texts if EOT not in t]
    wrong = [t for t in texts if wordbpe.encode(t) != hf_tok.encode(t)
             or hf_tok.decode(hf_tok.encode(t)) != t]
    return diff, wrong

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="the WordBPE merges, e.g. checkpoints/web/web_bpe.json")
    parser.add_argument("--native", action="store_true", help="our own layout, for models that are not GPT-2 shaped")
    parser.add_argument("--check", action="store_true", help="load the result with transformers and compare")
    args = parser.parse_args()

    config = export(args.checkpoint, args.out, args.tokenizer, args.native)
    print(f"wrote {args.out} ({(args.out / 'model.safetensors').stat().st_size / 1e6:.0f} MB)")
    if args.check and not args.native:
        texts = ([Path("README.md").read_text()]
                 + [line for p in Path("tools").glob("*.py") for line in p.read_text().splitlines() if line]
                 + ["Tom's cat_sat on 12³ mats…  \n\n\tdon't — “quoted” naïve 日本語 😀"])
        diff, wrong = check(args.checkpoint, args.out, args.tokenizer, texts)
        print(f"largest logit difference to our model: {diff:.2e}")
        print(f"tokeniser: {len(texts) - len(wrong)}/{len(texts)} texts match WordBPE")
