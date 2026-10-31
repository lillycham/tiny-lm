import json

import pytest
import torch

from core import gpt
from tools import hf_export

CONFIG = dict(block=16, emb=32, heads=4, layers=2, dropout=0.0, vocab=64, gelu=True, tied=True)

def test_gpt2_shape_check():
    hf_export.check_gpt2_shaped(CONFIG)
    for wrong in (dict(CONFIG, gelu=False), dict(CONFIG, tied=False), dict(CONFIG, smear=True)):
        with pytest.raises(ValueError):
            hf_export.check_gpt2_shaped(wrong)

def test_weights_match_transformers(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = gpt.GPT(**CONFIG).eval()
    state = hf_export.to_hf_weights(model.state_dict(), CONFIG["layers"])
    theirs = transformers.GPT2LMHeadModel(transformers.GPT2Config.from_dict(hf_export.hf_config(CONFIG, 63))).eval()
    missing, unexpected = theirs.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(name.endswith("attn.bias") or name.endswith("masked_bias") or name == "lm_head.weight"
               for name in missing), missing
    X = torch.randint(0, CONFIG["vocab"], (2, 10))
    with torch.no_grad():
        assert torch.allclose(model(X), theirs(X).logits, atol=1e-5)

def test_config_is_json():
    assert json.loads(json.dumps(hf_export.hf_config(CONFIG, 63)))["n_embd"] == 32
