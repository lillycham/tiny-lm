"""anneal.py: the mix of stories and web text."""
import numpy as np
import pytest

import anneal

@pytest.mark.parametrize("fraction", [0.5, 0.25])
def test_mix(fraction):
    """All the stories, then one unbroken stretch of web text, in the right amounts."""
    stories = np.arange(100, dtype=np.uint16)
    web = np.arange(1000, 10000, dtype=np.uint16)
    out = anneal.mix(stories, web, fraction)
    assert out.dtype == np.uint16
    assert len(out) == round(100 / fraction)
    assert (out[:100] == stories).all()
    rest = out[100:]
    assert (np.diff(rest.astype(int)) == 1).all() and rest[0] >= 1000   # a stretch of web

def test_mix_differs_by_seed():
    stories, web = np.zeros(10, dtype=np.uint16), np.arange(10000, dtype=np.uint16)
    assert anneal.mix(stories, web, seed=0)[10] != anneal.mix(stories, web, seed=1)[10]
