import os
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch
import pytest
from intextus import IntextusEncoder

@pytest.fixture
def mock_dependencies():
    # Setup mock files
    temp_dir = tempfile.TemporaryDirectory()
    model_path = os.path.join(temp_dir.name, "model.onnx")
    tokenizer_path = os.path.join(temp_dir.name, "tokenizer.json")
    
    with open(model_path, "wb") as f:
        f.write(b"mock_model_data")
    with open(tokenizer_path, "w") as f:
        f.write('{"vocab": {}}')
        
    yield model_path, tokenizer_path
    
    temp_dir.cleanup()

@patch("intextus.encoder.CppIntextusEncoder")
def test_encoder_init_and_encode(mock_cpp_encoder_cls, mock_dependencies):
    model_path, tokenizer_path = mock_dependencies
    
    # Configure mock C++ encoder
    mock_cpp_encoder = MagicMock()
    mock_cpp_encoder_cls.return_value = mock_cpp_encoder
    
    mock_cpp_encoder.query_marker_id = 1
    mock_cpp_encoder.doc_marker_id = 2
    mock_cpp_encoder.skiplist_arr = {10}
    
    # Mock return values for methods
    dummy_q_embs = np.array([[[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]]], dtype=np.float32)
    dummy_d_embs = np.array([[[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    
    mock_cpp_encoder.encode_queries.return_value = dummy_q_embs
    mock_cpp_encoder.encode_docs.return_value = dummy_d_embs
    
    # Create encoder
    encoder = IntextusEncoder(model_path, tokenizer_path)
    
    assert encoder.query_marker_id == 1
    assert encoder.doc_marker_id == 2
    assert 10 in encoder.skiplist_arr
    
    # Test encode_queries
    q_embs = encoder.encode_queries("test query", max_length=3, normalize=True)
    mock_cpp_encoder.encode_queries.assert_called_with(["test query"], 3, True)
    assert np.array_equal(q_embs, dummy_q_embs)
    
    # Test encode_docs
    d_embs = encoder.encode_docs("test doc", max_length=3, normalize=True)
    mock_cpp_encoder.encode_docs.assert_called_with(["test doc"], 3, True)
    assert np.array_equal(d_embs, dummy_d_embs)

@patch("intextus.encoder.CppIntextusEncoder")
def test_encoder_init_with_directory(mock_cpp_encoder_cls):
    temp_dir = tempfile.TemporaryDirectory()
    model_path = os.path.join(temp_dir.name, "model.onnx")
    tokenizer_path = os.path.join(temp_dir.name, "tokenizer.json")
    
    with open(model_path, "wb") as f:
        f.write(b"mock_model_data")
    with open(tokenizer_path, "w") as f:
        f.write('{"vocab": {}}')
        
    encoder = IntextusEncoder(temp_dir.name)
    
    mock_cpp_encoder_cls.assert_called_with(
        model_path,
        tokenizer_path,
        "[Q]",
        "[D]",
        True
    )
    
    temp_dir.cleanup()

@patch("intextus.encoder.CppIntextusEncoder")
@patch("intextus.encoder.os.path.exists")
@patch("huggingface_hub.hf_hub_download")
def test_encoder_init_with_hf_hub(mock_hf_download, mock_exists, mock_cpp_encoder_cls):
    def exists_side_effect(path):
        if path in ["mxbai-edge-colbert-v0-17m", "intextus/mxbai-edge-colbert-v0-17m-onnx"]:
            return False
        return True
    mock_exists.side_effect = exists_side_effect
    
    mock_hf_download.side_effect = lambda repo_id, filename: f"/mocked/path/{repo_id}/{filename}"
    
    encoder = IntextusEncoder("mxbai-edge-colbert-v0-17m")
    
    mock_hf_download.assert_any_call(repo_id="intextus/mxbai-edge-colbert-v0-17m-onnx", filename="model.onnx")
    mock_hf_download.assert_any_call(repo_id="intextus/mxbai-edge-colbert-v0-17m-onnx", filename="tokenizer.json")
    
    mock_cpp_encoder_cls.assert_called_with(
        "/mocked/path/intextus/mxbai-edge-colbert-v0-17m-onnx/model.onnx",
        "/mocked/path/intextus/mxbai-edge-colbert-v0-17m-onnx/tokenizer.json",
        "[Q]",
        "[D]",
        True
    )

def test_real_embedding_end_to_end():
    # End-to-end validation with the real default C++ engine and ONNX model
    print("\nRunning real end-to-end embedding test...")
    encoder = IntextusEncoder("mxbai-edge-colbert-v0-17m")
    
    # 1. Check metadata and properties
    assert encoder.query_marker_id >= 0
    assert encoder.doc_marker_id >= 0
    assert len(encoder.skiplist_arr) > 0
    
    # 2. Test query encoding
    queries = ["hello world", "this is an integration test query"]
    q_embs = encoder.encode_queries(queries, max_length=32, normalize=True)
    assert q_embs.shape == (2, 32, 48)  # mxbai-edge-colbert-v0-17m has a 48-dimensional embedding space
    
    # Check L2 normalization (sum of squares is close to 1)
    norm = np.linalg.norm(q_embs[0, 0])
    assert np.allclose(norm, 1.0, atol=1e-5)
    
    # 3. Test document encoding with punctuation masking
    docs = ["Hello world! This is a test document.", "Second document with some punct? Yes."]
    d_embs = encoder.encode_docs(docs, max_length=128, normalize=True)
    assert d_embs.shape == (2, 128, 48)
    
    # 4. Test MaxSim calculation (using the accelerated C++ version and python fallback)
    from intextus import compute_maxsim
    
    score = compute_maxsim(q_embs[0], d_embs[0])
    assert isinstance(score, float)
    assert score > 0.0
    
    print(f"Integration test score for 'hello world' similarity: {score:.4f}")
