"""
Test script to verify Ollama connection
"""

import requests
import sys

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("✅ Ollama is running and accessible!")
            models = response.json().get('models', [])
            if models:
                print("Available models:")
                for model in models:
                    print(f"  - {model.get('name', 'Unknown')}")
            else:
                print("⚠️  No models found. You may need to pull a model with 'ollama pull <model_name>'")
            return True
        else:
            print(f"❌ Ollama responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Make sure Ollama is running with 'ollama serve'")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

if __name__ == "__main__":
    print("Testing Ollama connection...")
    success = test_ollama_connection()
    if not success:
        print("\nTo start Ollama:")
        print("1. Run 'ollama serve' in a terminal")
        print("2. In another terminal, run 'ollama pull llama3' (or another model)")
        sys.exit(1)