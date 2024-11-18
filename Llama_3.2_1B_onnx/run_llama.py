import os
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import sys

class LlamaOnnxModel:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        # Initialize ONNX Runtime session
        model_path = self.model_dir / "model_q4f16.onnx"
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Model configuration
        self.num_attention_heads = 8
        self.head_dim = 64
        self.num_layers = 16
        
    def prepare_inputs(self, input_ids, attention_mask):
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs (int64)
        position_ids = np.arange(seq_length, dtype=np.int64)
        position_ids = np.expand_dims(position_ids, 0)
        
        # Keep input_ids and attention_mask as int64
        input_ids = input_ids.astype(np.int64)
        attention_mask = attention_mask.astype(np.int64)
        
        # Initialize past key values with float16
        past_key_values = []
        
        # Create empty past key/value states for each layer using float16
        for _ in range(self.num_layers):
            past_key = np.zeros((batch_size, self.num_attention_heads, 0, self.head_dim), dtype=np.float16)
            past_value = np.zeros((batch_size, self.num_attention_heads, 0, self.head_dim), dtype=np.float16)
            past_key_values.extend([past_key, past_value])
        
        # Prepare inputs dictionary
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        
        # Add past key values to inputs
        for i in range(self.num_layers):
            inputs[f'past_key_values.{i}.key'] = past_key_values[i*2]
            inputs[f'past_key_values.{i}.value'] = past_key_values[i*2 + 1]
            
        return inputs

    def generate_streaming(self, prompt, max_length=100, temperature=0.7):
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        generated_tokens = input_ids[0].tolist()
        
        # Print the prompt first
        print(prompt, end='', flush=True)
        
        try:
            # Generate tokens
            for _ in range(max_length):
                # Prepare all required inputs
                ort_inputs = self.prepare_inputs(input_ids, attention_mask)
                
                # Run inference
                outputs = self.session.run(None, ort_inputs)
                next_token_logits = outputs[0][:, -1, :]
                
                # Convert logits to float32 for numerical stability in softmax
                next_token_logits = next_token_logits.astype(np.float32)
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Sample next token
                probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
                next_token = np.random.choice(len(probs[0]), p=probs[0])
                
                # Append next token
                generated_tokens.append(next_token)
                
                # Decode just the new token
                new_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
                print(new_text, end='', flush=True)
                
                # Update input_ids and attention_mask for next iteration
                input_ids = np.array([generated_tokens], dtype=np.int64)
                attention_mask = np.ones((1, len(generated_tokens)), dtype=np.int64)
                
                # Check for end of generation
                if next_token == self.tokenizer.eos_token_id:
                    break
                    
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
        
        print("\n")  # Add a newline at the end
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

def main():
    # Set model directory
    model_dir = os.path.join(os.path.expanduser("~"), "Desktop/PythonServer/Llama_3.2_1B_onnx/model")
    
    print(f"Loading model from: {model_dir}\n")
    
    try:
        # Initialize model
        model = LlamaOnnxModel(model_dir)
        
        # Example usage
        prompt = "Write a short story about a robot learning to paint: "
        model.generate_streaming(prompt)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Please ensure all required files are in: {model_dir}")

if __name__ == "__main__":
    main()