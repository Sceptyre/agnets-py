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
from agnets import Agent, OpenAIConfig

# Create an agent with OpenAI backend
agent = Agent(config=OpenAIConfig(api_key="your-key"))

# Simple chat
response = agent.chat("What is the capital of France?")
print(response)

# Agent with Ollama backend
ollama_agent = Agent(config=OllamaConfig(model="llama3"))
response = ollama_agent.chat("Explain quantum computing")
print(response)
```

## Backends Supported
- OpenAI (gpt-3.5, gpt-4, etc.)
- Ollama (local models like llama3, mistral)
- Add new backends by implementing the `Backend` interface
