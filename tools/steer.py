"""Steer the story model with one feature of its sparse autoencoder (SAE).

A feature of the SAE is a direction in the residual stream after one block: when the
feature fires at 9, the stream gets 9 times that direction (tools/sae.py). So writing
with the direction added at every token asks the model what it would say if the feature
fired harder, and a negative strength asks what it would say without it:

    python -m tools.steer checkpoints/sae/stories_gpt_384w6l_full_L5_x16_k32.pt --feature 1822
    python -m tools.steer <sae> --feature 1822 --strengths -20,-10,0,10,20 --tokens 200
    python -m tools.steer <sae> --feature 1822 --prompt "The little dog" --temperature 0.7

The strength is in the feature's own units, so the peak saved with the contexts is the
natural scale: a feature with peak 9.2 is at full strength when it fires at 9.2, and a
strength of 20 is far past anything the text ever produced. Start near the peak and go
outwards, and read the same strength on two or three prompts before you believe it.

A feature that the SAE does not use much is a weak lever, however clear its name. A dead
feature does nothing at all. A feature on 30% of tokens does something broad and hard to
see, because it is already on nearly everywhere.

The seed is fixed before each strength, so the samples differ because of the steering
and not because of the dice. They still diverge quickly, which is why the text is not a
clean measurement: it shows the direction of the change, not its size.
"""
import argparse
import json
from pathlib import Path

import torch

from core import gpt
from core.word_bpe import TOKENISER, WordBPE
from stories.stories_gpt import EOT_ID
from tools import sae as sae_tools

def feature_direction(sae, index, device="cpu"):
    """What the feature adds to the residual stream at activation 1, in the model's own
    units: column `index` of the decoder, divided by scale, because decode() divides the
    whole of f by it. A strength of s is then "as if this feature fired at s"."""
    return (sae.dec.weight[:, int(index)] / sae.scale).detach().to(device)

@torch.no_grad()
def steered(model, layer, vector, strength, run):
    """Run `run()` with strength * vector added to the residual stream after block `layer`.

    A forward hook that returns a value replaces the block's output, so every block after
    this one sees the steered stream, and so does the SAE. The hook goes on every token,
    the prompt included: it is the context that the next token is read from that matters."""
    def add(module, inputs, output):
        return output + strength * vector

    handle = model.blocks[layer].register_forward_hook(add)
    try:
        return run()
    finally:
        handle.remove()

def write(model, tok, prompt, n, temperature=1.0, stop=None, steer=None):
    """Sample n tokens after `prompt`. steer is (layer, vector, strength), or None for the
    model's own text. The caller sets the seed, so two runs can share the dice."""
    def run():
        return gpt.generate(model, n, prompt, temperature, tok.encode, tok.decode, stop=stop)

    return run() if steer is None or steer[2] == 0 else steered(model, *steer, run)

def describe(data, labels, index, layer):
    """One line on what is being steered, so the text below it can be read in context."""
    row = data["features"][index] if data else []
    name = labels.get(str(index), "")
    peak = f"peak {row[0][0]:.2f}" if row else "never fired"
    rate = sae_tools.rate_text(data, index) if data else ""
    return (f"feature {index}  {name or '(no name)'}\n"
            f"layer {layer}, {peak}" + (f", {rate}" if rate else "")) + (
            "\nstrength is in the feature's own units: 1 is one activation, and the peak "
            "is the strongest it ever fired" if row else
            "\nthis feature never fired in the saved text, so it can only push, not steer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gpt.add_device_option(parser)
    parser.add_argument("sae", type=Path)
    parser.add_argument("--feature", type=int, required=True, help="the feature to steer with")
    parser.add_argument("--strengths", default="-8,0,8",
                        help="comma-separated strengths, in the feature's own units (default -8,0,8)")
    parser.add_argument("--prompt", default="Once upon a time", help="the text to write after")
    parser.add_argument("--tokens", type=int, default=120, help="tokens to write (default 120)")
    parser.add_argument("--temperature", type=float, default=0.8, help="1.0 is the model as trained (default 0.8)")
    parser.add_argument("--seed", type=int, default=0, help="the dice, shared by every strength (default 0)")
    args = parser.parse_args()

    sae, info = sae_tools.load(args.sae, args.device)
    model = gpt.load(args.device, Path(info["model"]))
    tok = WordBPE.load(TOKENISER)
    saved = sae_tools.examples_path(args.sae)
    data = json.loads(saved.read_text()) if saved.exists() else None
    labels = sae_tools.load_labels(args.sae)
    vector = feature_direction(sae, args.feature, args.device)

    print(describe(data, labels, args.feature, info["layer"]))
    print(f"\nprompt: {args.prompt!r}")
    for strength in [float(s) for s in args.strengths.split(",") if s.strip()]:
        torch.manual_seed(args.seed)
        text = write(model, tok, args.prompt, args.tokens, args.temperature, EOT_ID,
                     (info["layer"], vector, strength))
        print(f"\n{'=' * 20} strength {strength:+.1f}\n{text}")
