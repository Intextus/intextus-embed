import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Export a PyTorch ColBERT model to ONNX for intextus.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID or path to local PyTorch ColBERT model.")
    parser.add_argument("--output", type=str, default="model.onnx", help="Path to save the output ONNX model.")
    parser.add_argument("--tokenizer-output", type=str, default="tokenizer.json", help="Path to save the tokenizer.json file.")
    
    args = parser.parse_args()
    
    try:
        import torch
        import transformers
    except ImportError:
        print("Error: PyTorch and Transformers are required for the export utility.")
        print("Please install them using: pip install torch transformers")
        sys.exit(1)
        
    print(f"Loading ColBERT model from '{args.model}'...")
    
    class ColBERTWrapper(torch.nn.Module):
        def __init__(self, base_model, linear):
            super().__init__()
            self.base_model = base_model
            self.linear = linear
            
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            if token_type_ids is not None:
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Use last hidden state
            last_hidden_state = outputs.last_hidden_state
            
            # Apply the custom linear projection layer
            if self.linear is not None:
                embeddings = self.linear(last_hidden_state)
            else:
                embeddings = last_hidden_state
                
            return embeddings
            
    # Load model and tokenizer
    from transformers import AutoModel, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    
    # Check for linear projection layer
    linear = None
    if hasattr(model, "linear"):
        linear = model.linear
    elif hasattr(model, "projection"):
        linear = model.projection
    elif hasattr(model, "proj"):
        linear = model.proj
    elif hasattr(model, "pooler"):
        pass
        
    wrapper = ColBERTWrapper(model, linear)
    wrapper.eval()
    
    # Create dummy inputs
    dummy_input_ids = torch.ones(1, 32, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 32, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(1, 32, dtype=torch.long)
    
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "embeddings": {0: "batch_size", 1: "sequence_length"}
    }
    
    inputs = (dummy_input_ids, dummy_attention_mask)
    
    # Check if the base model accepts token_type_ids
    import inspect
    sig = inspect.signature(model.forward)
    if "token_type_ids" in sig.parameters:
        inputs = (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
        
    print("Exporting model to ONNX...")
    torch.onnx.export(
        wrapper,
        inputs,
        args.output,
        input_names=input_names,
        output_names=["embeddings"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True
    )
    
    print(f"ONNX model saved successfully to '{args.output}'")
    
    # Save tokenizer.json
    print(f"Saving tokenizer to '{args.tokenizer_output}'...")
    tokenizer._tokenizer.save(args.tokenizer_output)
    print("Done!")

if __name__ == "__main__":
    main()
