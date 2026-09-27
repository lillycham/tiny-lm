"""Look inside the story GPT: what each block computes, and whether it can copy.

A Recorder hooks into the model and keeps, for every block:
  - the residual stream after it: the (B, T, C) vectors that flow from block to block
  - each head's attention weights: which earlier tokens each position looks at

Then three small tests, scored with log-probabilities rather than right or wrong:
  1. Copy:      "...a girl named Zork. ... One day," -> is " Zork" more likely now?
  2. Pronouns:  "Lily and Tom ... Lily was sad because" -> " she" over " he"?
  3. Ball:      "Lily and Tom went to the park. Lily gave the ball to" -> " Tom" over " Lily"?
And the induction-head test: random tokens, repeated. A model that can copy
predicts the repeat, and some head looks back at the token after the first copy.

    python interp.py                                        # the base story model
    python interp.py --checkpoint checkpoints/stories_sft.pt
"""
import argparse
import math
import statistics as stats
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F

import gpt
from attention import attention
from stories_gpt import CHECKPOINT, load_tokens
from word_bpe import EOT_ID, TOKENISER, WordBPE

KNOWN = ["Lily", "Tom", "Ben", "Sue", "Max", "Mia", "Tim", "Anna", "Sam", "Lucy"]
UNSEEN = ["Zork", "Priya", "Kofi", "Ingrid", "Mateo", "Yuki", "Bram", "Oona", "Tariq", "Wren"]
GIRLS, BOYS = ["Lily", "Sue", "Mia", "Anna", "Lucy"], ["Tom", "Ben", "Max", "Tim", "Sam"]

# ---------- recording ----------
class Recorder:
    """Hooks that record the residual stream and the attention weights on every forward pass.

        with Recorder(model) as rec:
            model(X)
        rec.resid[0]   # (B, T, C): the embeddings, before any block
        rec.resid[i]   # (B, T, C): after block i - 1
        rec.attn[i]    # (B, n_head, T, T): block i's attention weights
    """

    def __init__(self, model):
        self.model = model
        self.resid, self.attn, self.handles = {}, {}, []

    def __enter__(self):
        # The embeddings are what comes out of model.drop (dropout is off in eval()).
        self.handles.append(self.model.drop.register_forward_hook(self.save_resid(0)))
        for i, block in enumerate(self.model.blocks):
            self.handles.append(block.register_forward_hook(self.save_resid(i + 1)))
            self.handles.append(block.attn.register_forward_hook(self.save_attn(i)))
        return self

    def __exit__(self, *exc):
        # Always take the hooks off again, or they stay on the model for good.
        for h in self.handles:
            h.remove()

    def save_resid(self, i):
        """A hook that saves a module's output as resid[i]."""
        # A forward hook is called as hook(module, inputs, output) after the module's
        # forward. inputs is a tuple of the arguments to forward.
        def hook(module, inputs, output):
            self.resid[i] = output.detach()
        return hook

    def save_attn(self, i):
        """A hook on a CausalSelfAttention that saves its attention weights as attn[i]."""
        def hook(module, inputs, output):
            x = inputs[0]
            q, k, v = module.qkv(x).split(x.shape[-1], dim=-1)
            q = module.split_heads(q)
            k = module.smear_keys(module.split_heads(k))
            v = module.split_heads(v)

            _, weights = attention(q, k, v)

            self.attn[i] = weights.detach()

        return hook

# ---------- head ablation ----------
class Ablate:
    """Hook that performs head ablation, wherein we switch off an attn head 
    within the model while active to test how scores drop as a result.

        with Ablate(model, layer=2, head=1):
            scores = copy_scores(model, tok, UNSEEN)
    """
    def __init__(self, model, layer, head):
        self.model = model
        self.layer = layer
        self.head = head

        self.attn_handle = None
        self.hs = None

    def __enter__(self):
        attn = self.model.blocks[self.layer].attn
        self.attn_handle = attn.proj.register_forward_pre_hook(self.zero_head)
        self.hs = attn.proj.in_features // attn.n_head

        return self

    def __exit__(self, *exc):
        self.attn_handle.remove()

    def zero_head(self, module, inputs):
        x = inputs[0].clone()
        x[:,:, self.head* self.hs:(self.head + 1) * self.hs] = 0
        return x

# ---------- scoring ----------
@torch.no_grad()
def logprob(model, tok, prompt, answer):
    """log P(answer | prompt), summed over the answer's tokens."""
    p, a = tok.encode(prompt), tok.encode(answer)
    ids = torch.tensor([[EOT_ID] + p + a])
    logp = F.log_softmax(model(ids[:, :-1]), dim=-1)[0]
    # The answer's tokens are the last len(a) targets.
    return sum(logp[len(p) + j, a[j]].item() for j in range(len(a)))

def copy_scores(model, tok, names):
    """For each name, how much more likely " name" gets when the story mentioned
       it before, in nats."""
    story = "Once upon a time, there was a little girl named {}. She liked to play with her red ball. One day,"
    gains = []
    for i, name in enumerate(names):
        other = names[(i + 1) % len(names)]
        gains.append(logprob(model, tok, story.format(name), " " + name)
                     - logprob(model, tok, story.format(other), " " + name))
    return gains

def diff_scores(model, tok, prompts):
    """logprob(right) - logprob(wrong) for each (prompt, right, wrong)."""
    return [logprob(model, tok, p, r) - logprob(model, tok, p, w)
             for p, r, w in prompts]

def summary(scores):
    """ list[float] -> tuple[float, float]
    Given a list of scores, return a mean and std. error."""
    SE = stats.stdev(scores) / math.sqrt(len(scores))

    return (stats.mean(scores), SE)

def pronoun_prompts():
    out = []
    for g, b in zip(GIRLS, BOYS):
        for first, second in ((g, b), (b, g)):
            for subject in (first, second):
                right = " she" if subject in GIRLS else " he"
                wrong = " he" if right == " she" else " she"
                out.append((f"{first} and {second} played with a ball. {subject} was sad because", right, wrong))
    return out

def ball_prompts():
    out = []
    for g, b in zip(GIRLS, BOYS):
        for giver, taker in ((g, b), (b, g)):
            for first, second in ((giver, taker), (taker, giver)):
                out.append((f"{first} and {second} went to the park. {giver} gave the ball to", " " + taker, " " + giver))
    return out

# ---------- induction heads ----------
def common_tokens(n=1000):
    """The n most common tokens in the validation text, without <|endoftext|>."""
    counts = np.bincount(np.asarray(load_tokens("val")[:2_000_000]), minlength=EOT_ID + 1)
    counts[EOT_ID] = 0
    return torch.tensor(np.argsort(-counts)[:n].copy())

@torch.no_grad()
def induction(model, pool, L=50, B=20, seed=0):
    """Random tokens from pool, repeated once.

    Returns the loss on the first copy, the loss on the repeat, and the induction score
    of every head, (layers, n_head). Rare tokens would test the model on tokens it
    barely knows, so pool is common ones.
    """
    g = torch.Generator().manual_seed(seed)
    r = pool[torch.randint(0, len(pool), (B, L), generator=g)]
    ids = torch.cat([torch.full((B, 1), EOT_ID), r, r], dim=1)       # (B, 1 + 2L)
    with Recorder(model) as rec:
        logits = model(ids[:, :-1])
    loss = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")   # (B, 2L)
    first, repeat = loss[:, :L - 1].mean().item(), loss[:, L:].mean().item()
    # ids: <eot>, then r at positions 1 to L, then r again at L + 1 to 2L. So the token
    # at a repeat position t first appeared at t - L. An induction head at t looks at
    # the token after that one, t - L + 1: the token that came next last time.
    n_layer, n_head = len(model.blocks), rec.attn[0].shape[1]
    scores = torch.zeros(n_layer, n_head)
    for i in range(n_layer):
        for t in range(L + 1, 2 * L):
            scores[i] += rec.attn[i][:,:,t,t - L + 1].mean(dim=0)
    return first, repeat, scores / (L - 1)

def show_head(model, tok, text, layer, head):
    """For each token of text, the earlier token this head looks at most."""
    ids = torch.tensor([[EOT_ID] + tok.encode(text)])
    with Recorder(model) as rec, torch.no_grad():
        model(ids)
    w = rec.attn[layer][0, head]
    names = ["<eot>"] + [tok.decode([i]) for i in ids[0, 1:].tolist()]
    return "  ".join(f"{names[t]!r}->{names[w[t].argmax()]!r}" for t in range(1, len(names)))

def format_stderr(pair):
    """tuple[float, float] -> string
    given a tuple pair of (mean, se), return a formatted string {mean:+.2f} ± {se:.2f}"""
    return f"{pair[0]:+.2f} ± {pair[1]:.2f}"


# ---------- checks ----------
@torch.no_grad()
def check_recorder(model, tok):
    ids = torch.tensor([[EOT_ID] + tok.encode("Once upon a time, there was a cat.")])
    with Recorder(model) as rec:
        logits = model(ids)
    T = ids.shape[1]
    print(f"1. {len(rec.resid)} residual streams of {tuple(rec.resid[0].shape)},"
          f" {len(rec.attn)} attention maps of {tuple(rec.attn[0].shape)}")
    w = rec.attn[0]
    print(f"   Rows sum to 1: {torch.allclose(w.sum(-1), torch.ones(1))},"
          f" nothing above the diagonal: {bool((w.triu(1) == 0).all())}")
    # The last residual stream, through the final LayerNorm and output layer, must give the logits.
    print(f"   Last stream -> logits matches: {torch.allclose(model.out(model.ln(rec.resid[len(model.blocks)])), logits, atol=1e-5)}")
    # Rebuild block 0's attention output from the recorded weights and compare.
    attn = model.blocks[0].attn
    x = model.blocks[0].ln1(rec.resid[0])
    v = attn.split_heads(attn.qkv(x).split(x.shape[-1], dim=-1)[2])
    rebuilt = attn.proj(attn.join_heads(w @ v))
    print(f"   Weights @ v rebuilds block 0's attention: {torch.allclose(rebuilt, attn(x), atol=1e-5)}")
    print(f"   Hooks removed afterwards: {all(len(m._forward_hooks) == 0 for m in model.modules())}")

def run_interp(model, tok, path):
    print(f"{path}: {sum(p.numel() for p in model.parameters()):,} parameters\n")
    check_recorder(model, tok)

    print("\n2. Name tokens:  " + "  ".join(tok.show(tok.encode(" " + n)) for n in KNOWN[:3] + UNSEEN[:5]))
    print(f"   Copy gain, names from the data: {format_stderr(summary(copy_scores(model, tok, KNOWN)))} nats")
    print(f"   Copy gain, unseen names:        {format_stderr(summary(copy_scores(model, tok, UNSEEN)))} nats")
    print(f"   Pronoun, she/he right minus wrong:  {format_stderr(summary(diff_scores(model, tok, pronoun_prompts())))} nats")
    print(f"   Ball, taker minus giver:            {format_stderr(summary(diff_scores(model, tok, ball_prompts())))} nats")
    print("   (0 = no preference; above 0 = the right way)")

    first, repeat, scores = induction(model, common_tokens())
    print(f"\n3. Random common tokens: loss {first:.2f} on the first copy, {repeat:.2f} on the repeat")
    print("   (a model that copies gets the repeat almost right: a loss near 0)")
    print("   Induction score of each head (attention to the token after the earlier copy):")
    print("          " + "".join(f"  head {h}" for h in range(scores.shape[1])))
    for i, row in enumerate(scores):
        print(f"   layer {i}" + "".join(f"  {s:6.2f}" for s in row.tolist()))
    layer, head = divmod(scores.argmax().item(), scores.shape[1])
    print(f"\n   The strongest, layer {layer} head {head}, on a story:")
    print("   " + show_head(model, tok, "Tom had a red ball. Tom gave the red", layer, head))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT, help=f"default {CHECKPOINT}")
    parser.add_argument("--against", type=Path, default=None, help="default off")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = gpt.load("cpu", args.checkpoint)
    tok = WordBPE.load(TOKENISER)

    run_interp(model, tok, args.checkpoint)

    if args.against:
        model_against = gpt.load("cpu", args.against)
        run_interp(model_against, tok, args.against)

