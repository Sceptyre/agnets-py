# agnets-py

A Python library for building flexible AI agent workflows with support for multiple backend providers (Ollama, OpenAI, etc.).

## What is an Agnet?

An **agnet** is simply a common misspelling of the word "agent" - it has no special acronym or intentional meaning.

As an AI agent implementation, it provides:

Key characteristics include:

- **Backend Agnosticism**: Works seamlessly with OpenAI, Ollama, and custom backends through a unified API
- **Extensible Architecture**: Easily add support for new AI models by implementing the `Backend` interface
- **Workflow Standardization**: Provides consistent message formatting, tool integration, and response processing
- **Configurable Behavior**: Fine-tune model parameters through the `Config` class (temperature, max tokens, etc.)

The core components:
1. **Config**: Defines model behavior parameters (equivalent to LLM hyperparameters)
2. **Backend**: Implements provider-specific communication logic
3. **Agnet**: Orchestrates the interaction workflow between config and backend

This architecture enables developers to:
- Switch between AI models with minimal code changes
- Combine multiple models in complex workflows
- Extend functionality through custom backend implementations

## Getting Started

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
    config=Config(model_name="qwen3:1.7b"),
    backend=OllamaBackend()
)
response = ollama_agnet.invoke("Explain quantum computing")
print(response)
```

## Backends Supported
- OpenAI (gpt-3.5, gpt-4, etc.)
- Ollama (local models like llama3, mistral)
- Add new backends by implementing the `Backend` interface
