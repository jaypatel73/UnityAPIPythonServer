
# LLaMA ONNX Model API Setup for Unity

This project sets up a Python server on your laptop that runs a LLaMA model using ONNX. You can make API calls from Unity to ask the LLaMA model questions and get responses.

## Project Structure

- `Model/`: Folder to store the ONNX model and tokenizer files.
  - `tokenizer.json`
  - `tokenizer_config.json`
- `run_llama.py`: Script to run the LLaMA model and generate answers.
- `server.py`: Python server script to handle API calls.
- `test_client.py`: Client script to test the API.
- `ModelRunner.cs`: Unity script that makes API calls to the server.
- **Dependencies**: Python scripts and ONNX model files.

## Setup Instructions

### Step 1: Download the ONNX Model
1. Go to [Hugging Face](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-onnx-web-gqa/tree/main/onnx).
2. Download the ONNX model files.
3. Place the downloaded files inside the `Model` folder.

### Step 2: Run the Python Server
1. Open a terminal on your laptop.
2. Navigate to the directory containing `server.py`.
3. Run the server using the command:
   ```bash
   python server.py
   ```
   
### Step 3: Setup Unity
1. Open your Unity project.
2. Attach `ModelRunner.cs` to a GameObject in your scene.
3. The script `ModelRunner.cs` contains a hardcoded question. When you run the Unity project, it will make an API call to the Python server and display the answer from the LLaMA model.

## How It Works
- **server.py**: Starts the server and waits for API calls from Unity.
- **run_llama.py**: Called by `server.py` to run the LLaMA ONNX model and return answers.
- **ModelRunner.cs**: Sends the question from Unity to the Python server and receives the model's response.

## Notes
- The API call in `ModelRunner.cs` is set up for a hardcoded question. You can modify it as needed.
