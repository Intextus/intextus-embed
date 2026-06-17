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

@patch("intextus.encoder.ort.InferenceSession")
@patch("intextus.encoder.Tokenizer")
def test_encoder_init_and_encode(mock_tokenizer_cls, mock_session_cls, mock_dependencies):
    model_path, tokenizer_path = mock_dependencies
    
    # Configure mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer_cls.from_file.return_value = mock_tokenizer
    mock_tokenizer.token_to_id.side_effect = lambda x: 1 if x == "[Q]" else (2 if x == "[D]" else None)
    mock_tokenizer.get_vocab.return_value = {
        "[CLS]": 101, "[SEP]": 102, "[Q]": 1, "[D]": 2, ".": 10, "hello": 20
    }
    
    # Configure mock ONNX session
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    
    mock_input_1 = MagicMock()
    mock_input_1.name = "input_ids"
    mock_input_2 = MagicMock()
    mock_input_2.name = "attention_mask"
    mock_session.get_inputs.return_value = [mock_input_1, mock_input_2]
    
    mock_output = MagicMock()
    mock_output.name = "embeddings"
    mock_session.get_outputs.return_value = [mock_output]
    
    # Mock model execution output (Batch=1, Seq=3, Dim=4)
    # Token 0: [CLS], Token 1: [Q]/[D], Token 2: "." (punctuation)
    dummy_output_embeddings = np.array([[[1.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 0.0]]], dtype=np.float32)
    mock_session.run.return_value = [dummy_output_embeddings]
    
    # Mock Tokenizer encodings (marker token will be inserted, making length 3)
    mock_encoding = MagicMock()
    mock_encoding.ids = [101, 10] # 101 is [CLS], 10 is punctuation
    mock_encoding.attention_mask = [1, 1]
    mock_tokenizer.encode_batch.return_value = [mock_encoding]
    
    # Create encoder
    encoder = IntextusEncoder(model_path, tokenizer_path)
    
    assert encoder.query_marker_id == 1
    assert encoder.doc_marker_id == 2
    assert 10 in encoder.skiplist_arr
    assert 20 not in encoder.skiplist_arr
    
    # Test encode_queries (no punctuation masking, but L2 normalized)
    q_embs = encoder.encode_queries("test query", max_length=3, normalize=True)
    mock_session.run.assert_called()
    assert q_embs.shape == (1, 3, 4)
    # Check L2 normalization: [1, 0, 0, 0] norm is 1
    assert np.allclose(q_embs[0, 0], [1.0, 0.0, 0.0, 0.0])
    
    # Reset mock for document test
    mock_session.run.reset_mock()
    
    # Test encode_docs (with punctuation masking - token at index 2 has ID 10 which is punctuation)
    d_embs = encoder.encode_docs("test doc", max_length=3, normalize=True)
    mock_session.run.assert_called()
    
    # Verify punctuation token is zeroed out
    assert np.allclose(d_embs[0, 2], [0.0, 0.0, 0.0, 0.0])
    # Other tokens are normalized
    assert np.allclose(d_embs[0, 0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(d_embs[0, 1], [0.0, 1.0, 0.0, 0.0])

@patch("intextus.encoder.ort.InferenceSession")
@patch("intextus.encoder.Tokenizer")
def test_encoder_init_with_directory(mock_tokenizer_cls, mock_session_cls):
    temp_dir = tempfile.TemporaryDirectory()
    model_path = os.path.join(temp_dir.name, "model.onnx")
    tokenizer_path = os.path.join(temp_dir.name, "tokenizer.json")
    
    with open(model_path, "wb") as f:
        f.write(b"mock_model_data")
    with open(tokenizer_path, "w") as f:
        f.write('{"vocab": {}}')
        
    mock_tokenizer = MagicMock()
    mock_tokenizer_cls.from_file.return_value = mock_tokenizer
    mock_tokenizer.get_vocab.return_value = {}
    
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.get_inputs.return_value = []
    
    mock_output = MagicMock()
    mock_output.name = "embeddings"
    mock_session.get_outputs.return_value = [mock_output]
    
    encoder = IntextusEncoder(temp_dir.name)
    
    mock_tokenizer_cls.from_file.assert_called_with(tokenizer_path)
    mock_session_cls.assert_called_with(model_path, providers=["CPUExecutionProvider"])
    
    temp_dir.cleanup()

@patch("intextus.encoder.ort.InferenceSession")
@patch("intextus.encoder.Tokenizer")
@patch("intextus.encoder.os.path.exists")
@patch("huggingface_hub.hf_hub_download")
def test_encoder_init_with_hf_hub(mock_hf_download, mock_exists, mock_tokenizer_cls, mock_session_cls):
    # Setup mock returns
    def exists_side_effect(path):
        # The model names are not local paths
        if path in ["mxbai-edge-colbert-v0-17m", "intextus/mxbai-edge-colbert-v0-17m-onnx"]:
            return False
        # The resolved downloaded paths exist
        return True
    mock_exists.side_effect = exists_side_effect
    
    mock_hf_download.side_effect = lambda repo_id, filename: f"/mocked/path/{repo_id}/{filename}"
    
    mock_tokenizer = MagicMock()
    mock_tokenizer_cls.from_file.return_value = mock_tokenizer
    mock_tokenizer.get_vocab.return_value = {}
    
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.get_inputs.return_value = []
    
    mock_output = MagicMock()
    mock_output.name = "embeddings"
    mock_session.get_outputs.return_value = [mock_output]
    
    # 1. Initialize using a model name alias
    encoder = IntextusEncoder("mxbai-edge-colbert-v0-17m")
    
    # Verify it maps alias to full repo and downloads it
    mock_hf_download.assert_any_call(repo_id="intextus/mxbai-edge-colbert-v0-17m-onnx", filename="model.onnx")
    mock_hf_download.assert_any_call(repo_id="intextus/mxbai-edge-colbert-v0-17m-onnx", filename="tokenizer.json")
    
    # Verify it loaded the downloaded files
    mock_tokenizer_cls.from_file.assert_called_with("/mocked/path/intextus/mxbai-edge-colbert-v0-17m-onnx/tokenizer.json")
    mock_session_cls.assert_called_with("/mocked/path/intextus/mxbai-edge-colbert-v0-17m-onnx/model.onnx", providers=["CPUExecutionProvider"])

    # 2. Reset mock calls to verify the default constructor
    mock_hf_download.reset_mock()
    mock_tokenizer_cls.from_file.reset_mock()
    mock_session_cls.reset_mock()
    
    encoder_default = IntextusEncoder()
    mock_hf_download.assert_any_call(repo_id="intextus/mxbai-edge-colbert-v0-17m-onnx", filename="model.onnx")
    mock_hf_download.assert_any_call(repo_id="intextus/mxbai-edge-colbert-v0-17m-onnx", filename="tokenizer.json")


