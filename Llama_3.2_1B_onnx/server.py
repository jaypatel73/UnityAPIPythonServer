# server.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import uvicorn
from typing import List, Optional
import asyncio
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

class LlamaOnnxModel:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model_path = self.model_dir / "model_q4f16.onnx"
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.num_attention_heads = 8
        self.head_dim = 64
        self.num_layers = 16
        
    def prepare_inputs(self, input_ids, attention_mask):
        batch_size, seq_length = input_ids.shape
        position_ids = np.arange(seq_length, dtype=np.int64)
        position_ids = np.expand_dims(position_ids, 0)
        input_ids = input_ids.astype(np.int64)
        attention_mask = attention_mask.astype(np.int64)
        past_key_values = []
        
        for _ in range(self.num_layers):
            past_key = np.zeros((batch_size, self.num_attention_heads, 0, self.head_dim), dtype=np.float16)
            past_value = np.zeros((batch_size, self.num_attention_heads, 0, self.head_dim), dtype=np.float16)
            past_key_values.extend([past_key, past_value])
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        
        for i in range(self.num_layers):
            inputs[f'past_key_values.{i}.key'] = past_key_values[i*2]
            inputs[f'past_key_values.{i}.value'] = past_key_values[i*2 + 1]
            
        return inputs

    async def generate_streaming(self, prompt, max_length=50, temperature=0.7):
        # Add instruction for concise response
        formatted_prompt = f"Answer the following question in 1-2 sentences: {prompt}"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="np")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        generated_tokens = input_ids[0].tolist()
        
        try:
            for _ in range(max_length):
                ort_inputs = self.prepare_inputs(input_ids, attention_mask)
                outputs = self.session.run(None, ort_inputs)
                next_token_logits = outputs[0][:, -1, :]
                next_token_logits = next_token_logits.astype(np.float32)
                next_token_logits = next_token_logits / temperature
                probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
                next_token = np.random.choice(len(probs[0]), p=probs[0])
                generated_tokens.append(next_token)
                new_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
                yield new_text
                input_ids = np.array([generated_tokens], dtype=np.int64)
                attention_mask = np.ones((1, len(generated_tokens)), dtype=np.int64)
                if next_token == self.tokenizer.eos_token_id:
                    break
                await asyncio.sleep(0)
                
        except Exception as e:
            print(f"Error during generation: {str(e)}")

# Initialize model globally
model_dir = os.path.join(os.path.expanduser("~"), "Desktop/PythonServer/Llama_3.2_1B_onnx/model")
model = LlamaOnnxModel(model_dir)

class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 50  # Reduced default max length
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        if request.stream:
            generator = model.generate_streaming(
                request.prompt,
                max_length=request.max_length,
                temperature=request.temperature
            )
            return EventSourceResponse(generate_events(generator))
        else:
            tokens = []
            async for token in model.generate_streaming(
                request.prompt,
                max_length=request.max_length,
                temperature=request.temperature
            ):
                tokens.append(token)
            return {"text": "".join(tokens)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_events(generator):
    try:
        async for token in generator:
            yield {"data": token}
    except Exception as e:
        yield {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)