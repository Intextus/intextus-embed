# intextus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

ColBERT embedding and MaxSim scoring without PyTorch. Uses a native C++ extension (ONNX Runtime + tokenizers-cpp) so you don't need to pull in 2 GB of deep learning dependencies just to encode some text.

## Install

```bash
pip install intextus-embed
```

Only runtime deps are `numpy` and `huggingface-hub`. The C++ bits (ONNX Runtime, tokenizer) are compiled into the wheel.

> [!NOTE]
> **Docker & Alpine Linux Compatibility**: Because the underlying precompiled ONNX Runtime library is linked against `glibc`, this package will not run out-of-the-box on Alpine Linux images (e.g., `python:3.10-alpine`).
>
> If deploying via Docker, it is highly recommended to use a Debian-based slim image:
> ```dockerfile
> FROM python:3.10-slim
> ```
> If you must use Alpine, you will need to install the compatibility layer: `apk add --no-cache gcompat`.

## Usage

```python
from intextus import LateInteractionEncoder, compute_maxsim

model = LateInteractionEncoder()  # downloads intextus/mxbai-edge-colbert-v0-17m-onnx

q = model.encode_queries("What is late interaction?")
d = model.encode_docs("ColBERT computes token-level similarity.")

score = compute_maxsim(q[0], d[0])
print(score)
```

You can also point it at a local directory with `model.onnx` and `tokenizer.json`:

```python
model = LateInteractionEncoder("./my-model/")
```

## Models

| Alias | Repo | Size | Dim | Notes |
|---|---|---|---|---|
| `mxbai-edge-colbert-v0-17m` | `intextus/mxbai-edge-colbert-v0-17m-onnx` | 66 MB | 48 | Default |
| `mxbai-edge-colbert-v0-32m` | `intextus/mxbai-edge-colbert-v0-32m-onnx` | 124 MB | 64 | |
| `lateon` | `intextus/lateon-onnx` | 580 MB | 128 | Case-sensitive: use `do_lower_case=False` |

Any ColBERT ONNX model should work if you put `model.onnx` and `tokenizer.json` in a folder and pass the path.

## How it works

- Tokenization and inference run in C++ via a nanobind extension
- GIL is released during encode and MaxSim calls, so you can run multiple threads
- Punctuation tokens are masked out of document embeddings (standard ColBERT behavior)
- Embeddings are L2-normalized by default
- CPU only for now

## License

MIT. See [LICENSE](LICENSE).
