# tiny-lm

A small language model, built up step by step. Part 1 goes from a bigram table
in NumPy to a GPT with a BPE tokeniser, on Tiny Shakespeare. Part 2 moves to
TinyStories: a GPT-2 shaped story model, fine-tuning to follow requests, and
interpretability tools to see what the model does inside. Everything trains on
the Mac GPU (MPS), and on NVIDIA GPUs too.

Each file is one step. Each file reuses the parts from the steps before it,
and its docstring explains what changed and why.

## Part 1: Tiny Shakespeare

| File              | Model                                                         | Val loss (nats/char) |
|-------------------|---------------------------------------------------------------|---------------------:|
| `bigram.py`       | Bigram table from counts, with add-one smoothing              |                 2.48 |
| `bigram_sgd.py`   | The same table, learned with gradient descent                 |                 2.50 |
| `mlp.py`          | MLP over 8 characters of context (Bengio et al., 2003), NumPy |                 1.72 |
| `mlp_torch.py`    | The same MLP in PyTorch, with autograd                        |                 1.72 |
| `attention.py`    | One causal self-attention head                                |                 2.34 |
| `transformer.py`  | Multi-head attention, feed-forward, LayerNorm, residuals      |                 1.71 |
| `gpt.py`          | Small GPT: batched heads, dropout, AdamW, MPS, checkpoints    |                 1.53 |
| `bpe.py`          | Byte-pair encoding tokeniser, 512 tokens                      |                    — |
| `bpe_gpt.py`      | The GPT, trained on BPE tokens                                |                 1.50 |

The BPE model predicts tokens, not characters. Its loss is converted to nats
per character, so all the rows compare directly. The uniform guess over 65
characters is 4.17.

## Part 2: TinyStories

| File              | What it does                                                            |
|-------------------|-------------------------------------------------------------------------|
| `word_bpe.py`     | Fast BPE: split into words first, GPT-2 style. 4,096 tokens, 3.95 characters per token. Encodes the 2.2 GB training file (553M tokens) in about 2 minutes |
| `stories_gpt.py`  | The story model: GPT-2 shaped (GELU, tied embeddings), 256 tokens of context. Can resume after a crash |
| `lr_sweep.py`     | Short runs to pick a learning rate before a long run                    |
| `sft.py`          | Supervised fine-tuning: "Tell me a story about Lily." With loss masking |
| `instruct_sft.py` | Fine-tuning on TinyStoriesInstruct: stories that must use given words   |
| `interp.py`       | Hooks that record the residual stream and attention, tests for copying, pronouns and "who gets the ball", and an induction-head test |

| Model                   | Parameters | Tokens seen | Val loss (nats/token) | Time on an M-series Mac |
|-------------------------|-----------:|------------:|----------------------:|------------------------:|
| Story GPT, width 192    |       3.5M |         61M |                 2.557 |                  45 min |
| Story GPT, width 384    |      12.3M |        262M |                 1.337 |                   7.6 h |

These losses are per token, on different text, so they don't compare with
Part 1.

### What we found

The 3.5M model can't copy a sequence from its context: it has no induction
head. It repeats names that are one token (`Lily`) but not names split into
pieces (`P|ri|y|a`), and fine-tuning doesn't change that. The 12M model has an
induction head (layer 5, head 3) by step 2,600, and with it:

| Test                                    |  3.5M |    12M |
|-----------------------------------------|------:|-------:|
| Copies an unseen name (nats)            | -0.13 |  +27.2 |
| "Who gets the ball" (nats)              | -0.48 |   +7.0 |
| After name SFT: uses an unseen name     |   0/9 |    5/9 |
| After instruct SFT: uses required words | 5/120 | 65/120 |

Fine-tuning doesn't build the copying mechanism. It teaches the model when to
use the one that pretraining built.

`gpt.py` also has an experimental option, `smear`: smeared keys (Olsson et al.,
2022), so that one head can do induction on its own. It isn't GPT-2 compatible.

## Setup

The flake gives a Python with NumPy and PyTorch:

```sh
nix develop      # or: direnv allow
```

Tiny Shakespeare is in `data/tinyshakespeare.txt`. TinyStories is too big for
the repo. Download it from Hugging Face
([roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
and [roneneldan/TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct))
into `data/`:

```sh
cd data
URL=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
curl -LO $URL/TinyStoriesV2-GPT4-train.txt       # 2.2 GB
curl -LO $URL/TinyStoriesV2-GPT4-valid.txt       # 22 MB

# For instruct_sft.py only. It uses the first 30 MB of the 2.7 GB training file.
URL=https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main
curl -LO $URL/TinyStories-Instruct-valid.txt     # 27 MB
curl -L $URL/TinyStories-Instruct-train.txt | head -c 31457280 > TinyStories-Instruct-train-30MB.txt
```

## Usage

Run a step to check its parts and train its model:

```sh
python gpt.py                    # Part 1: about 9 minutes on an M-series GPU
python bpe_gpt.py                # about 10 minutes

python word_bpe.py               # Part 2: train the tokeniser, save the tokens to data/
python stories_gpt.py            # the 3.5M story model, about 45 minutes
caffeinate -i python stories_gpt.py --emb 384 --steps 32000   # the 12M model, overnight
python sft.py --base checkpoints/stories_gpt_384w6l.pt        # fine-tune the 12M model
python interp.py --checkpoint checkpoints/stories_gpt_384w6l.pt
```

Generate text from any model with `main.py`:

```sh
python main.py mlp --start "ROMEO:"
python main.py gpt --temperature 0.8
python main.py stories --checkpoint checkpoints/stories_gpt_384w6l.pt
python main.py sft --start Hana --checkpoint checkpoints/stories_sft_384w6l.pt
python main.py instruct --start "dragon, soup, happy" --features Dialogue
python main.py tokens --start "Once upon a time, Tom's cat sat."
```

The trained models load their checkpoints from `checkpoints/`, so run the step
that trains them first. Use `python main.py --help` for all options.

Check the model code with the tests: tiny models on the CPU, a few seconds.

```sh
pytest
```

## Training on a rented GPU

The scripts use the fastest device they find: an NVIDIA GPU (`cuda`), then an
Apple GPU (`mps`), then the CPU. Use `--device` to choose one. Checkpoints
load on any device, so a model trained on a rented GPU runs on the Mac.

Push your commits first, because the rented machine clones from GitHub. Git
doesn't track the tokeniser or the token files, so copy them across. Replace
`HOST` with the machine's SSH address.

```sh
# On the rented machine
git clone https://github.com/lillycham/tiny-lm.git && cd tiny-lm
pip install -r requirements.txt
pytest                  # tests that need the token files skip until they are there

# On the Mac: the tokeniser (47 KB) and the tokens (1.1 GB)
rsync -avP checkpoints/stories_bpe.json HOST:tiny-lm/checkpoints/
rsync -avP data/tinystories_train.npy data/tinystories_val.npy HOST:tiny-lm/data/

# On the rented machine: time 200 steps, then start the real run
python stories_gpt.py --emb 384 --steps 200 --tag timing
nohup python stories_gpt.py --emb 384 --steps 67500 --tag full --snapshots > full.log 2>&1 &

# On the Mac: get the model and its snapshots back
rsync -avP HOST:tiny-lm/checkpoints/stories_gpt_384w6l_full.pt checkpoints/
rsync -avP HOST:tiny-lm/checkpoints/snapshots/ checkpoints/snapshots/
```

67,500 steps of 32 × 256 tokens is one pass over the training set. Always
give a rented run a `--tag`: without one, `--emb 384` saves to
`stories_gpt_384w6l.pt`, and the copy back replaces the 12M model on the Mac.
The run saves a resume file every 30 minutes. If the machine stops, run the
same command again to carry on. Keep the repo on a disk that outlives the
machine, if the provider has one.

## References

The work this repo builds on, in the order it appears.

**Data and path**
- Andrej Karpathy, [char-rnn](https://github.com/karpathy/char-rnn) (2015): the Tiny Shakespeare dataset. The
  bigram-to-GPT path follows the one in his [nanoGPT](https://github.com/karpathy/nanoGPT) and makemore work.
- Ronen Eldan and Yuanzhi Li, [TinyStories: How Small Can Language Models Be and Still Speak Coherent
  English?](https://arxiv.org/abs/2305.07759) (2023): the TinyStories and TinyStoriesInstruct datasets.

**Models**
- Yoshua Bengio et al., [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)
  (JMLR, 2003): `mlp.py`.
- Ashish Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017): attention and
  the transformer block.
- Kaiming He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (2016):
  residual connections.
- Jimmy Lei Ba, Jamie Ryan Kiros and Geoffrey Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450)
  (2016).
- Ruibin Xiong et al., [On Layer Normalization in the Transformer
  Architecture](https://arxiv.org/abs/2002.04745) (2020): pre-norm.
- Nitish Srivastava et al., [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) (JMLR, 2014).
- Alec Radford et al., [Language Models are Unsupervised Multitask
  Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  (2019): GPT-2, the shape of the story model.
- Dan Hendrycks and Kevin Gimpel, [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) (2016).
- Ofir Press and Lior Wolf, [Using the Output Embedding to Improve Language
  Models](https://arxiv.org/abs/1608.05859) (2017): tied embeddings.

**Tokenisers**
- Rico Sennrich, Barry Haddow and Alexandra Birch, [Neural Machine Translation of Rare Words with Subword
  Units](https://arxiv.org/abs/1508.07909) (2016): BPE for language models. `word_bpe.py` splits words the
  GPT-2 way.

**Training**
- Ilya Loshchilov and Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
  (2019): AdamW.
- Ilya Loshchilov and Frank Hutter, [SGDR: Stochastic Gradient Descent with Warm
  Restarts](https://arxiv.org/abs/1608.03983) (2017): the cosine learning-rate schedule.
- Jordan Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
  (2022): about 20 training tokens per parameter.
- Long Ouyang et al., [Training Language Models to Follow Instructions with Human
  Feedback](https://arxiv.org/abs/2203.02155) (2022): supervised fine-tuning.

**Interpretability**
- Nelson Elhage et al., [A Mathematical Framework for Transformer
  Circuits](https://transformer-circuits.pub/2021/framework/index.html) (2021): the residual stream view.
- Catherine Olsson et al., [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) (2022):
  the induction-head test, and smeared keys.
- Kevin Wang et al., [Interpretability in the Wild](https://arxiv.org/abs/2211.00593) (2022): indirect object
  identification, the "who gets the ball" test.

## Licence

MIT. See `LICENSE`.
