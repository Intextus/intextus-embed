import string
from typing import Dict, Set
import numpy as np

def compute_maxsim(query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> float:
    """
    Computes the late-interaction MaxSim score between query and document vectors.
    Uses the accelerated C++ backend if available, with a numpy fallback.
    
    Args:
        query_embeddings: Array of shape (Query_Tokens, Dim) representing query vector sequence.
        doc_embeddings: Array of shape (Doc_Tokens, Dim) representing document vector sequence.
        
    Returns:
        The float score representing late-interaction relevance.
    """
    try:
        from ._core import compute_maxsim as cpp_compute_maxsim
        # Ensure contiguous float32 arrays for C++ backend
        q = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        d = np.ascontiguousarray(doc_embeddings, dtype=np.float32)
        return cpp_compute_maxsim(q, d)
    except (ImportError, AttributeError, Exception):
        pass

    # Fallback to pure numpy implementation
    scores = np.dot(query_embeddings, doc_embeddings.T)
    max_scores_per_query_token = np.max(scores, axis=1)
    return float(np.sum(max_scores_per_query_token))

def get_punctuation_token_ids(
    vocab: Dict[str, int], 
    query_marker: str = "[Q]", 
    doc_marker: str = "[D]"
) -> Set[int]:
    """
    Identifies tokenizer vocabulary IDs that correspond to punctuation marks.
    This is used to construct a skiplist for document token masking.
    
    Args:
        vocab: Dictionary mapping token strings to their integer IDs.
        query_marker: Token representing query interaction.
        doc_marker: Token representing document interaction.
        
    Returns:
        A set of token IDs to be masked/skipped.
    """
    punctuation_chars = set(string.punctuation)
    skiplist_ids = set()
    
    # Common prefix/suffix subword markers used by various tokenizers
    clean_markers = ["##", "Ġ", " ", "</w>"]
    
    # Explicitly protect standard control tokens and query/doc markers
    protected_tokens = {
        query_marker, 
        doc_marker,
        "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]", 
        "<s>", "</s>", "<pad>", "<mask>", "<unk>"
    }
    
    for token, token_id in vocab.items():
        if token in protected_tokens:
            continue
            
        cleaned = token
        for marker in clean_markers:
            cleaned = cleaned.replace(marker, "")
            
        # Exclude special/control tokens (usually wrapped in [] or <> and longer than 1 char)
        if len(token) > 1 and (
            (token.startswith("[") and token.endswith("]")) or 
            (token.startswith("<") and token.endswith(">"))
        ):
            continue
            
        # A token is considered punctuation if its cleaned representation consists
        # entirely of standard punctuation characters (and is not empty).
        if cleaned and all(char in punctuation_chars for char in cleaned):
            skiplist_ids.add(token_id)
            
    return skiplist_ids
