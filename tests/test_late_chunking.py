import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from limbed.late_chunking import (
    content_token_budget,
    encode_document_late_chunked,
    find_chunk_char_span,
    iter_token_windows,
    resolve_model_max_length,
    tokens_intersecting_span,
)
from limbed.utils import compute_maxsim_late_chunked


def test_resolve_model_max_length():
    assert resolve_model_max_length("jina-colbert-v2") == 8192
    assert resolve_model_max_length("colbertv2.0") == 512
    assert resolve_model_max_length("thlurte/lateon-onnx") == 512


def test_content_token_budget():
    assert content_token_budget(512) == 509
    assert content_token_budget(32) == 29


def test_iter_token_windows_single():
    assert iter_token_windows(100, 509, 64) == [(0, 100)]


def test_iter_token_windows_multiple():
    windows = iter_token_windows(600, 509, 64)
    assert windows[0] == (0, 509)
    assert windows[-1][1] == 600
    assert len(windows) >= 2


def test_tokens_intersecting_span():
    offsets = [(0, 4), (4, 7), (8, 12)]
    assert tokens_intersecting_span(offsets, 0, 4) == [0]
    assert tokens_intersecting_span(offsets, 3, 9) == [0, 1, 2]


def test_find_chunk_char_span():
    doc = "hello world today"
    assert find_chunk_char_span(doc, "world", False) == (6, 11)
    with pytest.raises(ValueError, match="Chunk not found"):
        find_chunk_char_span(doc, "missing", False)


def test_compute_maxsim_late_chunked():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    c1 = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    c2 = np.array([[0.0, 1.0]], dtype=np.float32)
    assert compute_maxsim_late_chunked(q, [c1, c2], aggregation="max") == pytest.approx(1.5)
    assert compute_maxsim_late_chunked(q, [c1, c2], aggregation="sum") == pytest.approx(2.5)
    assert compute_maxsim_late_chunked(q, []) == 0.0


@patch("limbed.late_chunking.tokenize_with_offsets")
def test_encode_document_late_chunked_single_window(mock_tokenize):
    embedder = MagicMock()
    embedder.do_lower_case = False
    embedder.model_max_length = 512
    embedder.tokenizer_path = "/fake/tokenizer.json"

    doc = "alpha beta gamma"
    chunks = ["alpha", "beta gamma"]
    offsets = [(0, 5), (6, 10), (11, 16)]
    mock_tokenize.return_value = ([1, 2, 3], offsets)

    def fake_encode(texts, max_length, normalize):
        text = texts[0]
        dim = 4
        seq_len = 5
        out = np.zeros((1, seq_len, dim), dtype=np.float32)
        if text == "alpha":
            out[0, 2] = [1, 0, 0, 0]
        elif text == "alpha beta gamma":
            out[0, 2] = [1, 0, 0, 0]
            out[0, 3] = [0, 1, 0, 0]
            out[0, 4] = [0, 0, 1, 0]
        return out

    embedder.encode_docs.side_effect = fake_encode

    result = encode_document_late_chunked(
        embedder, doc, chunks, max_length=512, window_overlap=64, normalize=True
    )
    assert len(result) == 2
    assert result[0].shape == (1, 4)
    assert result[1].shape == (2, 4)


@patch("limbed.late_chunking.tokenize_with_offsets")
def test_encode_document_late_chunked_warns_on_long_doc(mock_tokenize):
    embedder = MagicMock()
    embedder.do_lower_case = False
    embedder.model_max_length = 512
    embedder.tokenizer_path = "/fake/tokenizer.json"

    n_tokens = 600
    offsets = [(i * 4, (i + 1) * 4) for i in range(n_tokens)]
    mock_tokenize.return_value = (list(range(n_tokens)), offsets)

    embedder.encode_docs.return_value = np.zeros((1, 512, 8), dtype=np.float32)

    doc = "x" * (n_tokens * 4)
    chunk = "x" * 8
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encode_document_late_chunked(
            embedder,
            doc,
            [chunk],
            max_length=512,
            window_overlap=64,
        )
    assert any("model windows" in str(w.message) for w in caught)


def test_real_late_chunking_end_to_end():
    pytest.importorskip("tokenizers")
    from limbed import LateEmbedder, compute_maxsim_late_chunked

    model = LateEmbedder("mxbai-edge-colbert-v0-17m", num_threads=1)
    doc = (
        "ColBERT uses late interaction between query and document token embeddings. "
        "Late chunking encodes the full document before slicing retrieval chunks."
    )
    chunks = [
        "ColBERT uses late interaction between query and document token embeddings.",
        "Late chunking encodes the full document before slicing retrieval chunks.",
    ]

    chunk_embs = model.encode_doc_late_chunked(doc, chunks, max_length=128)
    assert len(chunk_embs) == 2
    assert chunk_embs[0].ndim == 2
    assert chunk_embs[0].shape[1] == 48
    assert chunk_embs[0].shape[0] > 0
    assert chunk_embs[1].shape[0] > 0

    q = model.encode_queries("What is late chunking?", max_length=32)
    score = compute_maxsim_late_chunked(q[0], chunk_embs, aggregation="max")
    assert isinstance(score, float)
    assert score > 0.0
