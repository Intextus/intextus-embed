import os
from typing import List, Union
import numpy as np

# We import the C++ class under an alias to expose it via our Python wrapper
from ._core import LateEmbedder as CppLateEmbedder

class LateEmbedder:
    def __init__(
        self, 
        model_name_or_path: str = "thlurte/mxbai-edge-colbert-v0-17m-onnx", 
        tokenizer_path: str = None, 
        query_marker: str = "[Q]", 
        doc_marker: str = "[D]",
        do_lower_case: bool = None,
        provider: str = "CPUExecutionProvider",
        num_threads: int = 0,
        query_marker_id: int = None,
        doc_marker_id: int = None,
        cls_token_id: int = None,
        sep_token_id: int = None,
        pad_token_id: int = None,
        mask_token_id: int = None
    ):
        """
        Wrapper around the accelerated C++ ONNX engine for generic ColBERT execution.
        
        Args:
            model_name_or_path: Local path to a directory, an ONNX file, or a Hugging Face Hub model ID/alias.
            tokenizer_path: Optional path to tokenizer.json. If None, it is resolved automatically.
            query_marker: Special marker string (deprecated, mapped to query_marker_id).
            doc_marker: Special marker string (deprecated, mapped to doc_marker_id).
            do_lower_case: Whether to lower case input texts.
            provider: Execution provider (configured in C++ as CPUExecutionProvider).
            num_threads: Number of threads for ONNX Runtime intra-op parallelism.
                         0 (default) = auto-detect and use all available cores.
                         Set to 1 for single-threaded, or any positive int for explicit control.
            query_marker_id: Optional exact token ID for the query marker.
            doc_marker_id: Optional exact token ID for the document marker.
            cls_token_id: Optional exact token ID for CLS.
            sep_token_id: Optional exact token ID for SEP.
            pad_token_id: Optional exact token ID for PAD.
            mask_token_id: Optional exact token ID for MASK.
        """
        # Resolve paths dynamically
        model_path = None
        
        if os.path.exists(model_name_or_path):
            if os.path.isdir(model_name_or_path):
                model_path = os.path.join(model_name_or_path, "model.onnx")
                if not os.path.exists(model_path):
                    import glob
                    onnx_files = glob.glob(os.path.join(model_name_or_path, "**/*.onnx"), recursive=True)
                    if onnx_files:
                        model_path = onnx_files[0]
                if tokenizer_path is None:
                    t_path = os.path.join(os.path.dirname(model_path), "tokenizer.json") if model_path else None
                    if t_path and os.path.exists(t_path):
                        tokenizer_path = t_path
                    else:
                        tokenizer_path = os.path.join(model_name_or_path, "tokenizer.json")
            else:
                model_path = model_name_or_path
                if tokenizer_path is None:
                    dir_name = os.path.dirname(model_name_or_path)
                    tokenizer_path = os.path.join(dir_name, "tokenizer.json")
        else:
            repo_id = model_name_or_path
            supported_mappings = {
                "mxbai-edge-colbert-v0-17m": "thlurte/mxbai-edge-colbert-v0-17m-onnx",
                "mxbai-edge-colbert-v0-32m": "thlurte/mxbai-edge-colbert-v0-32m-onnx",
                "lateon": "thlurte/lateon-onnx",
                "colbertv2.0": "thlurte/colbertv2.0-onnx",
                "colbertv2": "thlurte/colbertv2.0-onnx",
                "jina-colbert-v2": "thlurte/jina-colbert-v2-onnx",
                "answerai-colbert-small-v1": "thlurte/answerai-colbert-small-v1-onnx"
            }
            if repo_id in supported_mappings:
                repo_id = supported_mappings[repo_id]
                
            try:
                from huggingface_hub import hf_hub_download
                print(f"Downloading model file from Hugging Face repository '{repo_id}'...")
                model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
                if tokenizer_path is None:
                    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
                
                # Check and download external model data file if present
                try:
                    hf_hub_download(repo_id=repo_id, filename="model.onnx.data")
                except Exception:
                    pass
            except Exception as e:
                raise ValueError(
                    f"Could not load model '{model_name_or_path}' from local path or Hugging Face Hub.\n"
                    f"Underlying error: {e}"
                )
                
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at {model_path}")
            
        if tokenizer_path is None or not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        # Resolve do_lower_case from tokenizer_config.json if not specified
        if do_lower_case is None:
            config_do_lower = None
            if tokenizer_path:
                config_path = os.path.join(os.path.dirname(tokenizer_path), "tokenizer_config.json")
                if os.path.exists(config_path):
                    try:
                        import json
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                        config_do_lower = config_data.get("do_lower_case")
                    except Exception:
                        pass
            if config_do_lower is not None:
                do_lower_case = config_do_lower
            else:
                # Default fallback rules
                do_lower_case = "uncased" in model_name_or_path.lower() or "colbertv2" in model_name_or_path.lower()

        # Determine token IDs dynamically if not provided
        is_jina = False
        vocab_size = 250000
        try:
            import json
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tok_data = json.load(f)
            vocab = tok_data.get("model", {}).get("vocab", {})
            if vocab:
                vocab_size = len(vocab)
                if isinstance(vocab, list):
                    vocab_tokens = {item[0] if isinstance(item, list) else item: i for i, item in enumerate(vocab)}
                else:
                    vocab_tokens = vocab
                
                if "<s>" in vocab_tokens and vocab_tokens["<s>"] == 0:
                    is_jina = True
        except Exception:
            pass

        if "jina" in model_name_or_path.lower():
            is_jina = True

        default_query_marker_id = 250002 if is_jina else 1
        default_doc_marker_id = 250003 if is_jina else 2
        default_cls_token_id = 0 if is_jina else 101
        default_sep_token_id = 2 if is_jina else 102
        default_pad_token_id = 1 if is_jina else 0
        default_mask_token_id = 250001 if is_jina else 103

        query_marker_id = query_marker_id if query_marker_id is not None else default_query_marker_id
        doc_marker_id = doc_marker_id if doc_marker_id is not None else default_doc_marker_id
        cls_token_id = cls_token_id if cls_token_id is not None else default_cls_token_id
        sep_token_id = sep_token_id if sep_token_id is not None else default_sep_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else default_pad_token_id
        mask_token_id = mask_token_id if mask_token_id is not None else default_mask_token_id

        # Determine attention mask query behaviour (attention mask is all 1s for Jina query)
        self.query_attn_mask_all_1s = is_jina

        # Precompute skip list (punctuation tokens) exactly matching fastembed
        skip_list = []
        try:
            import string
            vocab_tokens = locals().get("vocab_tokens")
            if vocab_tokens:
                for symbol in string.punctuation:
                    if symbol in vocab_tokens:
                        skip_list.append(vocab_tokens[symbol])
            else:
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(tokenizer_path)
                for symbol in string.punctuation:
                    ids = tok.encode(symbol, add_special_tokens=False).ids
                    if ids:
                        skip_list.append(ids[0])
        except Exception:
            pass

        # Initialize C++ core encoder
        self._encoder = CppLateEmbedder(
            model_path,
            tokenizer_path,
            do_lower_case,
            num_threads,
            query_marker_id,
            doc_marker_id,
            cls_token_id,
            sep_token_id,
            pad_token_id,
            mask_token_id,
            vocab_size,
            skip_list
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
        return self._encoder.encode_queries(queries, max_length, normalize, self.query_attn_mask_all_1s)

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

# Backwards compatibility alias
LateInteractionEncoder = LateEmbedder
