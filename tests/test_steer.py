"""Checks of steering (tools/steer.py): tiny sizes on the CPU, a few seconds in all."""
import torch

from core import gpt
from tools import sae as S
from tools import steer

TINY = dict(block=16, emb=12, heads=3, layers=2, dropout=0.0, vocab=50)

def tiny_model():
    torch.manual_seed(0)
    return gpt.GPT(**TINY).eval()

def fake_tok():
    """A tokeniser that maps any prompt to three tokens and every token to a letter."""
    return type("Tok", (), {"encode": lambda self, s: [1, 2, 3],
                            "decode": lambda self, ids: "".join(chr(97 + i % 26) for i in ids)})()

def test_a_direction_is_what_one_unit_of_the_feature_adds():
    """A strength of s has to mean "the feature fired at s", so the vector is exactly what
    decode() adds for one unit of the feature, and nothing else (not the bias)."""
    torch.manual_seed(0)
    sae = S.SAE(C=8, M=32, k=4)
    sae.set_scale(torch.randn(100, 8))
    one, none = torch.zeros(1, 32), torch.zeros(1, 32)
    one[0, 5] = 1
    assert torch.allclose(sae.decode(one)[0] - sae.decode(none)[0], steer.feature_direction(sae, 5))

def test_steering_adds_the_vector_at_the_layer():
    """The hook replaces the block's output, so every later block sees the steered stream;
    what it adds is the vector times the strength, on every token."""
    model = tiny_model()
    vector = torch.randn(TINY["emb"])
    X = torch.randint(0, TINY["vocab"], (2, TINY["block"]))
    plain = S.residuals(model, 1, X)
    pushed = steer.steered(model, 1, vector, 3.0, lambda: S.residuals(model, 1, X))
    assert torch.allclose(pushed - plain, (3.0 * vector).expand_as(plain))
    assert not model.blocks[1]._forward_hooks                   # (nothing left switched on)

def test_strength_zero_is_the_plain_text_and_only_the_caller_sets_the_dice():
    """The comparison is only worth reading if the dice are held still, so nothing in here
    seeds: the same seed gives the same text, with or without a strength of zero."""
    model, tok, vector = tiny_model(), fake_tok(), torch.randn(TINY["emb"])
    torch.manual_seed(0)
    plain = steer.write(model, tok, "hi", 5)
    torch.manual_seed(0)
    assert steer.write(model, tok, "hi", 5, steer=(1, vector, 0.0)) == plain
    torch.manual_seed(0)
    assert steer.write(model, tok, "hi", 20, steer=(1, vector, 50.0)) != plain
