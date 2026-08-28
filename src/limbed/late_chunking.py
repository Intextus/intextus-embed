"""ColBERT late chunking: encode model windows, slice retrieval chunks."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .encoder import LateEmbedder

# Total sequence length limits (CLS + marker + tokens + SEP) per shipped alias.
MODEL_MAX_LENGTH: Dict[str, int] = {
    "mxbai-edge-colbert-v0-17m": 512,
    "mxbai-edge-colbert-v0-32m": 512,
    "colbertv2.0": 512,
    "colbertv2": 512,
    "answerai-colbert-small-v1": 512,
    "lateon": 512,
    "jina-colbert-v2": 8192,
}

DEFAULT_MODEL_MAX_LENGTH = 512


def resolve_model_max_length(model_name_or_path: str) -> int:
    key = model_name_or_path.lower()
    for alias, limit in MODEL_MAX_LENGTH.items():
        if alias in key:
            return limit
    if "jina" in key and "colbert" in key:
        return 8192
    return DEFAULT_MODEL_MAX_LENGTH


def preprocess_text(text: str, do_lower_case: bool) -> str:
    return text.lower() if do_lower_case else text


def content_token_budget(max_length: int, has_doc_marker: bool = True) -> int:
    """Max content tokens fitting in a packed ColBERT doc sequence."""
    overhead = 2 + (1 if has_doc_marker else 0)  # CLS + SEP + optional [D]
    return max(1, max_length - overhead)


def iter_token_windows(
    n_tokens: int,
    budget: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    """Return [start, end) token index ranges covering the document."""
    if n_tokens <= 0:
        return []
    if n_tokens <= budget:
        return [(0, n_tokens)]

    overlap = max(0, min(overlap, budget - 1))
    stride = max(1, budget - overlap)
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + budget, n_tokens)
        windows.append((start, end))
        if end >= n_tokens:
            break
        start += stride
    return windows


def _load_tokenizer(tokenizer_path: str):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ImportError(
            "late chunking requires the `tokenizers` package. "
            "Install with: pip install tokenizers"
        ) from exc
    tok = Tokenizer.from_file(tokenizer_path)
    tok.no_padding()
    tok.no_truncation()
    return tok


def tokenize_with_offsets(tokenizer_path: str, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    enc = _load_tokenizer(tokenizer_path).encode(text, add_special_tokens=False)
    offsets = [(s, e) for s, e in enc.offsets]
    return list(enc.ids), offsets


def find_chunk_char_span(doc: str, chunk: str, do_lower_case: bool) -> Tuple[int, int]:
    chunk_text = preprocess_text(chunk, do_lower_case)
    start = doc.find(chunk_text)
    if start < 0:
        raise ValueError(f"Chunk not found in document: {chunk_text[:120]!r}")
    return start, start + len(chunk_text)


def tokens_intersecting_span(
    offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    indices: List[int] = []
    for i, span in enumerate(offsets):
        if span is None:
            continue
        ts, te = span
        if ts < char_end and te > char_start:
            indices.append(i)
    return indices


def _encode_window_tokens(
    embedder: "LateEmbedder",
    doc: str,
    window_start: int,
    window_end: int,
    offsets: List[Tuple[int, int]],
    max_length: int,
    normalize: bool,
    token_embeddings: List[Optional[np.ndarray]],
    token_priority: List[float],
) -> None:
    if window_end <= window_start:
        return

    char_start = offsets[window_start][0]
    char_end = offsets[window_end - 1][1]
    window_text = doc[char_start:char_end]

    emb = embedder.encode_docs([window_text], max_length=max_length, normalize=normalize)[0]
    n_local = min(window_end - window_start, content_token_budget(max_length))
    content_start = 2  # [CLS] [D]
    window_center = window_start + n_local / 2.0

    for local_i in range(n_local):
        global_i = window_start + local_i
        seq_i = content_start + local_i
        if seq_i >= emb.shape[0]:
            break
        row = emb[seq_i]
        if float(np.linalg.norm(row)) < 1e-6:
            continue
        priority = -abs(global_i - window_center)
        if priority > token_priority[global_i]:
            token_embeddings[global_i] = row
            token_priority[global_i] = priority


def encode_document_late_chunked(
    embedder: "LateEmbedder",
    doc: str,
    chunks: List[str],
    max_length: Optional[int] = None,
    window_overlap: int = 64,
    normalize: bool = True,
) -> List[np.ndarray]:
    """
    Encode a document with ColBERT late chunking.

    Long documents are split into model-sized windows (each encoded once).
    Retrieval ``chunks`` (substrings of ``doc``) receive sliced multi-vector
    embeddings from those windows.
    """
    if not chunks:
        return []

    model_max = embedder.model_max_length
    effective_max = model_max if max_length is None else min(max_length, model_max)
    if max_length is not None and max_length > model_max:
        warnings.warn(
            f"max_length={max_length} exceeds model limit {model_max}; "
            f"using {effective_max}.",
            stacklevel=2,
        )

    doc_text = preprocess_text(doc, embedder.do_lower_case)
    token_ids, offsets = tokenize_with_offsets(embedder.tokenizer_path, doc_text)
    n_tokens = len(token_ids)
    if n_tokens == 0:
        dim = embedder.encode_docs([doc], max_length=effective_max, normalize=normalize).shape[-1]
        return [np.zeros((0, dim), dtype=np.float32) for _ in chunks]

    budget = content_token_budget(effective_max)
    windows = iter_token_windows(n_tokens, budget, window_overlap)

    if len(windows) > 1:
        warnings.warn(
            f"Document has {n_tokens} tokens (> {budget} content budget). "
            f"Encoding in {len(windows)} overlapping model windows; "
            "retrieval chunks spanning window boundaries lose cross-window context.",
            stacklevel=2,
        )

    token_embeddings: List[Optional[np.ndarray]] = [None] * n_tokens
    token_priority: List[float] = [float("-inf")] * n_tokens

    for window_start, window_end in windows:
        _encode_window_tokens(
            embedder,
            doc_text,
            window_start,
            window_end,
            offsets,
            effective_max,
            normalize,
            token_embeddings,
            token_priority,
        )

    encoded_rows = [row for row in token_embeddings if row is not None]
    if encoded_rows:
        dim = encoded_rows[0].shape[-1]
    else:
        probe = embedder.encode_docs([doc_text[:512] or " "], max_length=effective_max, normalize=normalize)
        dim = probe.shape[-1]

    results: List[np.ndarray] = []
    for chunk in chunks:
        char_start, char_end = find_chunk_char_span(doc_text, chunk, embedder.do_lower_case)
        token_indices = tokens_intersecting_span(offsets, char_start, char_end)
        rows = [token_embeddings[i] for i in token_indices if token_embeddings[i] is not None]
        if not rows:
            warnings.warn(
                f"No encoded tokens for chunk: {chunk[:120]!r} "
                "(outside model windows or punctuation-only).",
                stacklevel=2,
            )
            results.append(np.zeros((0, dim), dtype=np.float32))
        else:
            results.append(np.stack(rows, axis=0).astype(np.float32, copy=False))

    return results
