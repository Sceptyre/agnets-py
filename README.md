# agnets-py

A Python library for building flexible AI agent workflows with support for multiple backend providers (Ollama, OpenAI, etc.).

## What is an Agent?

The `Agent` class provides a unified interface for interacting with AI models through different backends. It handles:
- Message formatting
- Tool integration
- Response processing
- Backend configuration

## Getting Started

### Installation (Development Mode)
```bash
uv pip install -e .
```

### Basic Usage
```python
from agnets import Agnet, Config
from agnets.backends.openai import OpenAICompatibleBackend

# Create an agnet with OpenAI-compatible backend
agnet = Agnet(
    config=Config(model_name="gpt-3.5-turbo"),
    backend=OpenAICompatibleBackend(api_key="your-key")
)

# Simple chat
response = agnet.invoke("What is the capital of France?")
print(response)

# Agnet with Ollama backend
ollama_agnet = Agnet(
    config=Config(model_name="llama3"),
    backend=OpenAICompatibleBackend(base_url="http://localhost:11434/v1")
)
response = ollama_agnet.invoke("Explain quantum computing")
print(response)
```

## Backends Supported
- OpenAI (gpt-3.5, gpt-4, etc.)
- Ollama (local models like llama3, mistral)
- Add new backends by implementing the `Backend` interface
