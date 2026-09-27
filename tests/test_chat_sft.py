"""chat_sft.py: the Dolly prompt format, the split and the examples."""
import chat_sft
import sft
from word_bpe import WordBPE

ROWS = [{"instruction": f"Question {i}?", "context": "Some text." if i % 2 else "", "response": f" Answer {i}. "}
        for i in range(20)]

def test_prompt_with_and_without_context():
    assert chat_sft.prompt("Why?") == "User: Why?\nAssistant: "
    assert chat_sft.prompt(" Why? ", " A paragraph. ") == "User: Why?\n\nA paragraph.\nAssistant: "

def test_split_is_fixed_and_disjoint():
    train, val = chat_sft.split(ROWS, n_val=5)
    assert len(train) == 15 and len(val) == 5
    assert chat_sft.split(ROWS, n_val=5) == (train, val)
    assert not {r["instruction"] for r in train} & {r["instruction"] for r in val}

def test_dataset_learns_only_the_answer():
    tok = WordBPE.train([r["instruction"] + r["context"] + r["response"] for r in ROWS] * 3, 280)
    X, Y = chat_sft.dataset(tok, ROWS[:2], block=64)
    assert X.shape == Y.shape == (2, 64)
    learned = [t for t in Y[0].tolist() if t != sft.IGNORE]
    assert learned == tok.encode("Answer 0.") + [tok.eot_id]

def test_too_long_conversations_are_dropped():
    tok = WordBPE.train(["a b c"] * 5, 258)
    long = [{"instruction": "a " * 100, "context": "", "response": "b"}] + ROWS[:1]
    X, _ = chat_sft.dataset(tok, long, block=64)
    assert len(X) == 1

def test_fine_tune_with_accumulation_learns():
    """accum splits each step into micro-batches; a tiny model still learns one example."""
    import torch
    import gpt
    torch.manual_seed(0)
    model = gpt.GPT(block=16, emb=32, heads=2, layers=1, dropout=0.0, vocab=20, gelu=True, tied=True)
    x = torch.arange(16).remainder(20)[None].repeat(4, 1)
    y = x.roll(-1, dims=1)
    y[:, :4] = sft.IGNORE
    before = sft.answer_loss(model, x, y, "cpu")
    sft.fine_tune(model, x, y, "cpu", STEPS=60, B=4, LR=1e-2, WARMUP=5, val=(x, y), accum=2)
    assert sft.answer_loss(model, x, y, "cpu") < before / 4
