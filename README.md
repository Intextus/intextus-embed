# intextus

[![PyPI Version](https://img.shields.io/pypi/v/intextus-embed.svg)](https://pypi.org/project/intextus-embed/)
[![CI/CD Status](https://github.com/intextus/intextus-embed/actions/workflows/publish.yml/badge.svg)](https://github.com/intextus/intextus-embed/actions/workflows/publish.yml)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/intextus-embed)](https://pypi.org/project/intextus-embed/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://pypi.org/project/intextus-embed/)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-blue.svg)](https://pypi.org/project/intextus-embed/)
[![Architectures](https://img.shields.io/badge/arch-x86__64%20%7C%20arm64%20%7C%20aarch64-lightgrey.svg)](https://pypi.org/project/intextus-embed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ColBERT embedding and MaxSim scoring without PyTorch. Uses a native C++ extension (ONNX Runtime + tokenizers-cpp) so you don't need to pull in 2 GB of deep learning dependencies just to encode some text.

## Install

```bash
pip install intextus-embed
```

Only runtime deps are `numpy` and `huggingface-hub`.

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

## Docker & Alpine Linux Compatibility

Because the underlying precompiled ONNX Runtime library is linked against `glibc`, this package will not run out-of-the-box on Alpine Linux images (e.g., `python:3.10-alpine`).

If deploying via Docker, it is highly recommended to use a Debian-based slim image:

```dockerfile
FROM python:3.10-slim
```

If you must use Alpine, you will need to install the compatibility layer: `apk add --no-cache gcompat`.

## Supported Platforms & Architectures

Precompiled wheels are published to PyPI for the following environments:

| Operating System | Architecture | Python Versions | Notes |
| --- | --- | --- | --- |
| **Linux** | `x86_64`, `aarch64` | 3.9, 3.10, 3.11, 3.12, 3.13, 3.14 | Built on `manylinux_2_28` (glibc-based) |
| **macOS** | `arm64` (Apple Silicon) | 3.9, 3.10, 3.11, 3.12, 3.13, 3.14 | SDK/deployment target macOS 13.3+ |
| **Windows** | `AMD64` (x86_64) | 3.9, 3.10, 3.11, 3.12, 3.13, 3.14 | |

> [!NOTE]
> Other platforms (such as Intel-based macOS or ARM-based Windows) will fall back to compilation from the source distribution (`sdist`). This requires a local C++ compiler (supporting C++17) and CMake.

## License

MIT. See [LICENSE](LICENSE).
