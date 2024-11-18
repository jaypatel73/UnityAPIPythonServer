# test_client.py
import requests
import sseclient

def test_regular_generation():
    url = "http://localhost:8000/generate"
    
    # Test questions
    questions = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is machine learning?",
        "Why is the sky blue?"
    ]
    
    for question in questions:
        data = {
            "prompt": question,
            "max_length": 50,
            "temperature": 0.7,
            "stream": False
        }
        
        print(f"\nQuestion: {question}")
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print("Answer:", result["text"].strip())
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    print("Testing with various questions:")
    test_regular_generation()