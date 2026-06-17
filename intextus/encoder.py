import os
from typing import List, Union, Dict, Any
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from intextus.utils import get_punctuation_token_ids

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
        Pure ONNX engine for generic ColBERT execution.
        
        Args:
            model_name_or_path: Local path to a directory, an ONNX file, or a Hugging Face Hub model ID/alias.
            tokenizer_path: Optional path to tokenizer.json. If None, it is resolved automatically.
            query_marker: Special marker string used to denote query sequence.
            doc_marker: Special marker string used to denote document sequence.
            provider: Execution provider for ONNX Runtime inference.
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
            
        # Initialize the ultra-fast Rust tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Initialize execution session
        self.session = ort.InferenceSession(model_path, providers=[provider])
        
        self.do_lower_case = do_lower_case

        # Dynamically discover graph inputs/outputs to remain generic
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        
        # Fetch token IDs for ColBERT context injection (handling trailing space variants)
        self.query_marker_id = self.tokenizer.token_to_id(query_marker)
        if self.query_marker_id is None:
            # Fallback for models (like PyLate/mxbai) where special tokens have trailing spaces
            self.query_marker_id = self.tokenizer.token_to_id(query_marker + " ")
            if self.query_marker_id is not None:
                query_marker = query_marker + " "
                
        self.doc_marker_id = self.tokenizer.token_to_id(doc_marker)
        if self.doc_marker_id is None:
            self.doc_marker_id = self.tokenizer.token_to_id(doc_marker + " ")
            if self.doc_marker_id is not None:
                doc_marker = doc_marker + " "
                
        if self.query_marker_id is None or self.doc_marker_id is None:
            print(f"[Warning] Custom markers '{query_marker.strip()}'/'{doc_marker.strip()}' not found in vocabulary. Defaulting to standard tokenization.")
            
        # Dynamically find all token IDs associated with string punctuation symbols
        # to construct the punctuation masking skiplist.
        skiplist_set = get_punctuation_token_ids(
            vocab=self.tokenizer.get_vocab(),
            query_marker=query_marker,
            doc_marker=doc_marker
        )
        # Pre-compile the skiplist to a NumPy array for fast vector-optimized masking
        self.skiplist_arr = np.array(list(skiplist_set), dtype=np.int64)

    def _prepare_inputs(self, texts: List[str], marker_id: int, max_length: int) -> Dict[str, np.ndarray]:
        # Lowercase texts if the model is case-insensitive
        if self.do_lower_case:
            texts = [t.lower() for t in texts]
            
        # Determine the target tokenization length prior to inserting the prefix token
        token_len = max_length - 1 if marker_id is not None else max_length
        
        self.tokenizer.enable_padding(style="max_length", length=token_len)
        self.tokenizer.enable_truncation(max_length=token_len)
        
        encodings = self.tokenizer.encode_batch(texts)
        
        input_ids = []
        attention_masks = []
        
        for enc in encodings:
            ids = list(enc.ids)
            mask = list(enc.attention_mask)
            
            # Insert the ColBERT interaction marker [Q] or [D] right after [CLS] (index 1)
            if marker_id is not None and len(ids) > 1:
                ids.insert(1, marker_id)
                ids = ids[:max_length]
                mask.insert(1, 1)
                mask = mask[:max_length]
                
            input_ids.append(ids)
            attention_masks.append(mask)
            
        inputs = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_masks, dtype=np.int64)
        }
        
        # Handle models exported with an optional token_type_ids layer
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
            
        return inputs

    def encode_queries(self, queries: Union[str, List[str]], max_length: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encodes query texts into multi-vector embeddings.
        
        Args:
            queries: A single query string or list of query strings.
            max_length: Maximum query sequence length (usually 32 for ColBERT).
            normalize: Whether to apply L2 normalization to the output vectors.
            
        Returns:
            A NumPy array of query embeddings with shape (Batch, Seq_Len, Dim).
        """
        if isinstance(queries, str):
            queries = [queries]
        onnx_inputs = self._prepare_inputs(queries, self.query_marker_id, max_length)
        embeddings = self.session.run([self.output_name], onnx_inputs)[0]
        
        if normalize:
            norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            # Optimize in-place division using where filter to avoid zero-division allocation
            np.divide(embeddings, norm, out=embeddings, where=norm != 0.0)
            
        return embeddings

    def encode_docs(self, docs: Union[str, List[str]], max_length: int = 256, normalize: bool = True) -> np.ndarray:
        """
        Encodes document texts into multi-vector embeddings, automatically zeroing out
        embeddings corresponding to punctuation tokens to reduce index footprint and search noise.
        
        Args:
            docs: A single document string or list of document strings.
            max_length: Maximum document sequence length (usually 256 for ColBERT).
            normalize: Whether to apply L2 normalization to the output vectors.
            
        Returns:
            A NumPy array of document embeddings with shape (Batch, Seq_Len, Dim).
        """
        if isinstance(docs, str):
            docs = [docs]
        onnx_inputs = self._prepare_inputs(docs, self.doc_marker_id, max_length)
        embeddings = self.session.run([self.output_name], onnx_inputs)[0]
        
        # Zero out embeddings for punctuation tokens in the document
        input_ids = onnx_inputs["input_ids"]
        # Optimized set membership check using pre-compiled NumPy array
        mask = np.isin(input_ids, self.skiplist_arr)
        
        # Apply the mask via element-wise multiplication (1.0 for words, 0.0 for punctuation)
        # This executes in-place using continuous memory strides, bypassing index copy overhead
        keep_mask = (~mask)[:, :, np.newaxis]
        embeddings *= keep_mask
        
        if normalize:
            norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            np.divide(embeddings, norm, out=embeddings, where=norm != 0.0)
            
        return embeddings
