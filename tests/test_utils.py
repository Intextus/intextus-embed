import numpy as np
from intextus.utils import compute_maxsim, get_punctuation_token_ids

def test_compute_maxsim():
    # 2 query tokens, dim 3
    q = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    # 3 doc tokens, dim 3
    d = np.array([
        [0.8, 0.2, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Dot products:
    # q[0] . d = [0.8, 0.1, 0.0] -> max = 0.8
    # q[1] . d = [0.2, 0.9, 0.0] -> max = 0.9
    # Sum of max = 0.8 + 0.9 = 1.7
    score = compute_maxsim(q, d)
    assert abs(score - 1.7) < 1e-6

def test_get_punctuation_token_ids():
    vocab = {
        "[CLS]": 101,
        "[SEP]": 102,
        "[Q]": 1,
        "[D]": 2,
        ".": 10,
        "Ġ.": 11,
        "##.": 12,
        "hello": 20,
        "world": 21,
        "##world": 22,
        "?": 13,
        "!": 14,
        "Ġ,": 15
    }
    
    skiplist = get_punctuation_token_ids(vocab, query_marker="[Q]", doc_marker="[D]")
    
    # Punctuation should be in skiplist
    assert 10 in skiplist  # "."
    assert 11 in skiplist  # "Ġ."
    assert 12 in skiplist  # "##."
    assert 13 in skiplist  # "?"
    assert 14 in skiplist  # "!"
    assert 15 in skiplist  # "Ġ,"
    
    # Standard words and special tokens should NOT be in skiplist
    assert 101 not in skiplist  # "[CLS]"
    assert 102 not in skiplist  # "[SEP]"
    assert 1 not in skiplist    # "[Q]"
    assert 2 not in skiplist    # "[D]"
    assert 20 not in skiplist   # "hello"
    assert 21 not in skiplist   # "world"
    assert 22 not in skiplist   # "##world"
