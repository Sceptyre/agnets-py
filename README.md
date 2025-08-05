# agnets-py

## Overview

agnets-py is a Python framework for building AI agents with tool-calling capabilities and multi-agent coordination. It provides a flexible, type-safe foundation for creating sophisticated AI-powered applications.

## Key Features

- **AI Agent Framework**: Core Agent class with tool management
- **Multi-Backend Support**: Works with OpenAI-compatible and Ollama backends
- **Multi-Agent Coordination**: Fleet class for managing agent interactions
- **Type-Safe Configuration**: Pydantic models for configuration management
- **MCP Integration**: Standardized tool calling using Model Context Protocol
- **Extensible Architecture**: Designed for easy extension and customization

## Quick Start

### Installation

```bash
pip install agnets-py
```

### Basic Usage

```python
from agnets import Agent, Config
from agnets.backends.openai import OpenAICompatibleBackend

# Create an agent with OpenAI backend
agent = Agent(
    config=Config(model_name="gpt-4"),
    backend=OpenAICompatibleBackend()
)

# Add a tool to the agent
@agent.add_tool
def example_tool(input: str) -> str:
    return f"Processed: {input}"

# Invoke the agent
response = agent.invoke("Hello, world!")
print(response)
```

## Examples

See the `examples/` directory for practical usage demonstrations including:
- Single agent with tools
- Multi-agent coordination
- Different backend configurations

## Contributing

Contributions are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
