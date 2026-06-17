import string
from typing import Dict, Set
import numpy as np

def compute_maxsim(query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> float:
    """
    Computes the late-interaction MaxSim score between query and document vectors.
    
    Args:
        query_embeddings: Array of shape (Query_Tokens, Dim) representing query vector sequence.
        doc_embeddings: Array of shape (Doc_Tokens, Dim) representing document vector sequence.
        
    Returns:
        The float score representing late-interaction relevance.
    """
    # Compute the dot product matrix between every query token and every document token
    # Resulting shape: (Query_Tokens, Doc_Tokens)
    scores = np.dot(query_embeddings, doc_embeddings.T)
    
    # Take the maximum score across the document tokens for each query token
    max_scores_per_query_token = np.max(scores, axis=1)
    
    # Sum the maximums together to get final relevance score
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
