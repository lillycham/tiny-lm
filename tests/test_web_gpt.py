"""web_gpt.py: the model size, the step count, the checkpoint names, and val_B."""
import pytest
import torch

from core import gpt
from web import web_gpt as wg

def test_the_default_model_is_gpt2_small_sized():
    """GPT-2 small's blocks (85M), with our smaller vocabulary. Built on the meta device:
       shapes only, no memory."""
    with torch.device("meta"):
        model = gpt.GPT(**wg.CONFIG)
    total, blocks = wg.count(model)
    C = 768
    # Per block: weights 3C² (qkv) + C² (proj) + 8C² (feed-forward); biases C (proj) + 5C
    # (feed-forward), no qkv bias; two LayerNorms 4C. Then the final LayerNorm, 2C.
    assert blocks == 12 * (12 * C * C + 6 * C + 4 * C) + 2 * C
    assert 84e6 < blocks < 86e6 and 97e6 < total < 99e6
    assert model.out.weight is model.tok.weight                     # tied: the embedding counts once
    assert model.config["scaled_init"]                              # 12 layers: 24 writes to the stream

def test_steps_for():
    assert wg.steps_for(2.7e9, 32, 1024) == 82_397
    assert wg.steps_for(10, 32, 1024) == 1                          # never 0
    assert wg.steps_for(2.7e9, 32, 1024, accum=4) == 20_599            # 131k tokens a step

def test_checkpoint_path():
    assert wg.checkpoint_path(wg.CONFIG).name == "web_gpt_768w12l.pt"
    assert wg.checkpoint_path(dict(wg.CONFIG, emb=256, layers=4), "test").name == "web_gpt_256w4l_test.pt"

def test_train_model_passes_val_B(monkeypatch):
    """At 1,024 tokens and 16,384 classes, val_loss's default B=64 needs ~4 GB for its logits."""
    seen = []
    monkeypatch.setattr(gpt, "val_loss", lambda model, ids, device, B=64: seen.append(B) or 0.0)
    ids = torch.randint(0, 50, (500,))
    model = gpt.GPT(block=16, emb=12, heads=3, layers=1, dropout=0.0, vocab=50)
    gpt.train_model(model, "cpu", STEPS=4, B=2, WARMUP=1, log_every=2, train_ids=ids, val_ids=ids, val_B=3)
    assert seen == [3, 3]

def test_instruct_sft_picks_the_web_tokeniser():
    from pathlib import Path
    from stories import instruct_sft
    from web import web_data
    from core import word_bpe
    assert instruct_sft.tokeniser_for(Path("checkpoints/web/web_gpt_768w12l_anneal.pt")) == web_data.TOKENISER
    assert instruct_sft.tokeniser_for(Path("checkpoints/stories/stories_gpt_384w6l.pt")) == word_bpe.TOKENISER

def test_sft_example_ends_with_the_tokenisers_own_end_of_text():
    from stories import sft
    from core.word_bpe import WordBPE
    tok = WordBPE.train(["the cat sat on the mat and the dog ran"] * 5, 262)
    x, y = sft.make_example(tok, "User: hi\nAssistant: ", "the cat sat")
    answer = tok.encode("the cat sat")
    assert [t for t in y if t != sft.IGNORE] == answer + [tok.eot_id]
    assert x[-1] == tok.eot_id                   # padding
