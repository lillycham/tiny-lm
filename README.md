# tiny-lm

A small language model, built up step by step on Tiny Shakespeare. It starts
from a bigram table in NumPy and ends at a 2.9M-parameter GPT with a BPE
tokeniser, trained on the Mac GPU.

Each file is one step. Each file reuses the parts from the steps before it,
and its docstring explains what changed and why.

## The steps

| File              | Model                                                      | Val loss (nats/char) |
|-------------------|------------------------------------------------------------|---------------------:|
| `bigram.py`       | Bigram table from counts, with add-one smoothing           |                 2.48 |
| `bigram_sgd.py`   | The same table, learned with gradient descent              |                 2.50 |
| `mlp.py`          | MLP over 8 characters of context (Bengio et al., 2003), NumPy |              1.72 |
| `mlp_torch.py`    | The same MLP in PyTorch, with autograd                     |                 1.72 |
| `attention.py`    | One causal self-attention head                             |                 2.34 |
| `transformer.py`  | Multi-head attention, feed-forward, LayerNorm, residuals   |                 1.71 |
| `gpt.py`          | Small GPT: batched heads, dropout, AdamW, MPS, checkpoints |                 1.53 |
| `bpe.py`          | Byte-pair encoding tokeniser, 512 tokens                   |                    — |
| `bpe_gpt.py`      | The GPT, trained on BPE tokens                             |                 1.50 |

The BPE model predicts tokens, not characters. Its loss is converted to nats
per character, so all the rows compare directly. The uniform guess over 65
characters is 4.17.

## Setup

The flake gives a Python with NumPy and PyTorch:

```sh
nix develop      # or: direnv allow
```

The data is in `data/tinyshakespeare.txt`.

## Usage

Run a step to check its parts and train its model:

```sh
python bigram.py
python gpt.py              # about 9 minutes on an M-series GPU
python bpe_gpt.py          # about 10 minutes
```

Generate text from any model with `main.py`:

```sh
python main.py counts
python main.py mlp --start "ROMEO:"
python main.py gpt --temperature 0.8
python main.py bpe -n 250
```

The `gpt` and `bpe` models load their checkpoints from `checkpoints/`, so run
`gpt.py` or `bpe_gpt.py` first. Use `python main.py --help` for all options.

## Licence

MIT. See `LICENSE`.
