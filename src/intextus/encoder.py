import os
from typing import List, Union
import numpy as np

# We import the C++ class under an alias to expose it via our Python wrapper
from ._core import IntextusEncoder as CppIntextusEncoder

class IntextusEncoder:
    def __init__(
        self, 
        model_name_or_path: str = "intextus/mxbai-edge-colbert-v0-17m-onnx", 
        tokenizer_path: str = None, 
        query_marker: str = "[Q]", 
        doc_marker: str = "[D]",
        do_lower_case: bool = True,
        provider: str = "CPUExecutionProvider"
    ):
        """
        Wrapper around the accelerated C++ ONNX engine for generic ColBERT execution.
        
        Args:
            model_name_or_path: Local path to a directory, an ONNX file, or a Hugging Face Hub model ID/alias.
            tokenizer_path: Optional path to tokenizer.json. If None, it is resolved automatically.
            query_marker: Special marker string used to denote query sequence.
            doc_marker: Special marker string used to denote document sequence.
            do_lower_case: Whether to lower case input texts.
            provider: Execution provider (configured in C++ as CPUExecutionProvider).
        """
        # Resolve paths dynamically
        model_path = None
        
        if os.path.exists(model_name_or_path):
            if os.path.isdir(model_name_or_path):
                model_path = os.path.join(model_name_or_path, "model.onnx")
                if tokenizer_path is None:
                    tokenizer_path = os.path.join(model_name_or_path, "tokenizer.json")
            else:
                model_path = model_name_or_path
                if tokenizer_path is None:
                    dir_name = os.path.dirname(model_name_or_path)
                    tokenizer_path = os.path.join(dir_name, "tokenizer.json")
        else:
            repo_id = model_name_or_path
            supported_mappings = {
                "mxbai-edge-colbert-v0-17m": "intextus/mxbai-edge-colbert-v0-17m-onnx",
                "mxbai-edge-colbert-v0-32m": "intextus/mxbai-edge-colbert-v0-32m-onnx",
                "lateon": "intextus/lateon-onnx"
            }
            if repo_id in supported_mappings:
                repo_id = supported_mappings[repo_id]
                
            try:
                from huggingface_hub import hf_hub_download
                print(f"Downloading model file from Hugging Face repository '{repo_id}'...")
                model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
                if tokenizer_path is None:
                    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
            except Exception as e:
                raise ValueError(
                    f"Could not load model '{model_name_or_path}' from local path or Hugging Face Hub.\n"
                    f"Underlying error: {e}"
                )
                
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at {model_path}")
            
        if tokenizer_path is None or not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        # Initialize C++ core encoder
        self._encoder = CppIntextusEncoder(
            model_path,
            tokenizer_path,
            query_marker,
            doc_marker,
            do_lower_case
        )

    @property
    def query_marker_id(self) -> int:
        return self._encoder.query_marker_id

    @property
    def doc_marker_id(self) -> int:
        return self._encoder.doc_marker_id

    @property
    def skiplist_arr(self) -> np.ndarray:
        # The C++ core returns a set of ids. We convert it to a NumPy array for compatibility.
        return np.array(list(self._encoder.skiplist_arr), dtype=np.int64)

    def encode_queries(self, queries: Union[str, List[str]], max_length: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encodes query texts into multi-vector embeddings using the accelerated C++ backend.
        
        Args:
            queries: A single query string or list of query strings.
            max_length: Maximum query sequence length (usually 32 for ColBERT).
            normalize: Whether to apply L2 normalization to the output vectors.
            
        Returns:
            A NumPy array of query embeddings with shape (Batch, Seq_Len, Dim).
            If a single query was passed, the shape is still (1, Seq_Len, Dim).
        """
        if isinstance(queries, str):
            queries = [queries]
        return self._encoder.encode_queries(queries, max_length, normalize)

    def encode_docs(self, docs: Union[str, List[str]], max_length: int = 256, normalize: bool = True) -> np.ndarray:
        """
        Encodes document texts into multi-vector embeddings using the accelerated C++ backend.
        Automatically zeroes out embeddings corresponding to punctuation tokens to reduce search noise.
        
        Args:
            docs: A single document string or list of document strings.
            max_length: Maximum document sequence length (usually 256 for ColBERT).
            normalize: Whether to apply L2 normalization to the output vectors.
            
        Returns:
            A NumPy array of document embeddings with shape (Batch, Seq_Len, Dim).
            If a single document was passed, the shape is still (1, Seq_Len, Dim).
        """
        if isinstance(docs, str):
            docs = [docs]
        return self._encoder.encode_docs(docs, max_length, normalize)
