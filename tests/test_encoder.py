import os
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch
import pytest
from intextus import LateInteractionEncoder

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
    encoder = LateInteractionEncoder(model_path, tokenizer_path)
    
    assert encoder.query_marker_id == 1
    assert encoder.doc_marker_id == 2
    assert 10 in encoder.skiplist_arr
    
    # Test encode_queries (asserting query_attn_mask_all_1s is passed as False by default)
    q_embs = encoder.encode_queries("test query", max_length=3, normalize=True)
    mock_cpp_encoder.encode_queries.assert_called_with(["test query"], 3, True, False)
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
        
    encoder = LateInteractionEncoder(temp_dir.name)
    
    mock_cpp_encoder_cls.assert_called_with(
        model_path,
        tokenizer_path,
        False,          # do_lower_case
        0,              # num_threads
        1,              # query_marker_id
        2,              # doc_marker_id
        101,            # cls_token_id
        102,            # sep_token_id
        0,              # pad_token_id
        103,            # mask_token_id
        250000,         # vocab_size
        []              # skip_list
    )
    
    temp_dir.cleanup()

@pytest.mark.parametrize("model_alias,expected_repo_id,expected_params", [
    (
        "mxbai-edge-colbert-v0-17m",
        "intextus/mxbai-edge-colbert-v0-17m-onnx",
        {"do_lower": False, "num_threads": 0, "query_m": 1, "doc_m": 2, "cls": 101, "sep": 102, "pad": 0, "mask": 103, "vocab": 250000}
    ),
    (
        "colbertv2.0",
        "intextus/colbertv2.0-onnx",
        {"do_lower": True, "num_threads": 0, "query_m": 1, "doc_m": 2, "cls": 101, "sep": 102, "pad": 0, "mask": 103, "vocab": 250000}
    ),
    (
        "colbertv2",
        "intextus/colbertv2.0-onnx",
        {"do_lower": True, "num_threads": 0, "query_m": 1, "doc_m": 2, "cls": 101, "sep": 102, "pad": 0, "mask": 103, "vocab": 250000}
    ),
    (
        "jina-colbert-v2",
        "intextus/jina-colbert-v2-onnx",
        {"do_lower": False, "num_threads": 0, "query_m": 250002, "doc_m": 250003, "cls": 0, "sep": 2, "pad": 1, "mask": 250001, "vocab": 250000}
    ),
    (
        "answerai-colbert-small-v1",
        "intextus/answerai-colbert-small-v1-onnx",
        {"do_lower": False, "num_threads": 0, "query_m": 1, "doc_m": 2, "cls": 101, "sep": 102, "pad": 0, "mask": 103, "vocab": 250000}
    )
])
@patch("intextus.encoder.CppIntextusEncoder")
@patch("intextus.encoder.os.path.exists")
@patch("huggingface_hub.hf_hub_download")
def test_encoder_init_with_hf_hub(mock_hf_download, mock_exists, mock_cpp_encoder_cls, model_alias, expected_repo_id, expected_params):
    def exists_side_effect(path):
        if path in [model_alias, expected_repo_id]:
            return False
        return True
    mock_exists.side_effect = exists_side_effect
    
    mock_hf_download.side_effect = lambda repo_id, filename: f"/mocked/path/{repo_id}/{filename}"
    
    encoder = LateInteractionEncoder(model_alias)
    
    mock_hf_download.assert_any_call(repo_id=expected_repo_id, filename="model.onnx")
    mock_hf_download.assert_any_call(repo_id=expected_repo_id, filename="tokenizer.json")
    
    mock_cpp_encoder_cls.assert_called_with(
        f"/mocked/path/{expected_repo_id}/model.onnx",
        f"/mocked/path/{expected_repo_id}/tokenizer.json",
        expected_params["do_lower"],
        expected_params["num_threads"],
        expected_params["query_m"],
        expected_params["doc_m"],
        expected_params["cls"],
        expected_params["sep"],
        expected_params["pad"],
        expected_params["mask"],
        expected_params["vocab"],
        []
    )

def test_real_embedding_end_to_end():
    # End-to-end validation with the real default C++ engine and ONNX model
    print("\nRunning real end-to-end embedding test...")
    encoder = LateInteractionEncoder("mxbai-edge-colbert-v0-17m")
    
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
    assert d_embs.shape[0] == 2
    assert d_embs.shape[2] == 48
    
    # 4. Test MaxSim calculation (using the accelerated C++ version and python fallback)
    from intextus import compute_maxsim
    
    score = compute_maxsim(q_embs[0], d_embs[0])
    assert isinstance(score, float)
    assert score > 0.0
    
    print(f"Integration test score for 'hello world' similarity: {score:.4f}")


def test_real_embedding_no_tokenizers_package():
    # Verify that the encoder successfully falls back to loading punctuation
    # even when the python tokenizers package is not available.
    import sys
    with patch.dict(sys.modules, {"tokenizers": None}):
        encoder = LateInteractionEncoder("mxbai-edge-colbert-v0-17m")
        assert len(encoder.skiplist_arr) > 0
        
        # Test basic encoding works
        q_embs = encoder.encode_queries(["hello"], max_length=16)
        assert q_embs.shape[0] == 1
