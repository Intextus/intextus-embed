# intextus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**intextus** is an ultra-lightweight, 100% PyTorch-free, and production-grade Python library designed to encode late-interaction ColBERT multi-vectors. 

By replacing massive deep learning libraries with highly optimized, compiled C++/Rust backends, **intextus** delivers full ColBERT MaxSim embeddings in **under 65MB of RAM** with **zero PyTorch or Transformers dependencies**. It is optimized for edge devices, serverless functions (AWS Lambda, Cloudflare Workers), and resource-constrained environments.

---

## Installation

Install the library directly via pip:

```bash
pip install intextus-embed
```

> [!NOTE]
> `intextus` currently defaults to highly optimized CPU inference. Full hardware acceleration and GPU execution support are planned for a future release.


---

## Quick Start

Here is how to load a model, extract multi-vector embeddings, and compute late-interaction cross-similarity scores entirely in NumPy:

```python
from intextus import IntextusEncoder, compute_maxsim

# Initialize the encoder (defaults to intextus/mxbai-edge-colbert-v0-17m-onnx)
model = IntextusEncoder()

# Or initialize from a local directory containing 'model.onnx' and 'tokenizer.json'
# model = IntextusEncoder("./my_model_directory")

# Extract query and document embeddings (Batch_Size, Sequence_Length, Dimension)
query_embeddings = model.encode_queries("What is ultra-low latency?")
doc_embeddings = model.encode_docs("ONNX runtime bypasses the PyTorch layer completely.")

# Compute the cross-similarity score via NumPy (using the first item in the batch)
score = compute_maxsim(query_embeddings[0], doc_embeddings[0])
print(f"Relevance Score (MaxSim): {score:.4f}")
```

---

## Supported & Tested Models

`intextus` is designed for ultra-fast, edge-compatible ColBERT execution. The primary officially supported and fully validated models are:

- **`intextus/mxbai-edge-colbert-v0-17m-onnx`** (Alias: `mxbai-edge-colbert-v0-17m`) — A highly-optimized, single-file ONNX representation of ModernBERT-backed `mxbai-edge-colbert-v0-17m` (66 MB, 48-dimensional late-interaction embeddings). **(Default Model)**
- **`intextus/mxbai-edge-colbert-v0-32m-onnx`** (Alias: `mxbai-edge-colbert-v0-32m`) — A larger, higher-capacity ONNX representation of ModernBERT-backed `mxbai-edge-colbert-v0-32m` (124 MB, 64-dimensional late-interaction embeddings).
- **`intextus/lateon-onnx`** (Alias: `lateon`) — A high-capacity base ModernBERT-backed model (580 MB, 128-dimensional late-interaction embeddings). Note: LateOn is case-sensitive, so load it with `IntextusEncoder("lateon", do_lower_case=False)`.

> [!NOTE]
> Any ColBERT model exported via standard Hugging Face/PyLate workflows can be loaded locally by providing the path to its `model.onnx` and `tokenizer.json`.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
