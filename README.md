# agnets

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://opensource.org/license/gpl-3-0)

## Overview

`agnets` is a modern, Python-first framework designed for building powerful AI agents and orchestrating them into collaborative, multi-agent systems—a "fleet." It provides a flexible, type-safe foundation for creating everything from simple automated assistants to complex, coordinated AI organizations that can delegate tasks and specialize.

Our vision is to empower developers to build sophisticated AI-powered applications by providing a framework that makes defining agents, their tools, and their inter-agent escalations intuitive and robust, allowing you to construct pseudo-organizations as collections of inference-powered Python objects.

## Key Features

- **👥 Multi-Agent Fleets**: The `Fleet` class is the core of multi-agent coordination, managing agent interactions, escalations, and communication pathways.
- **🤖 AI Agent Framework**: Core `Agent` class with integrated tool management, configuration, and lifecycle handling.
- **🔗 Agent Delegation & Escalation**: Define which agents can escalate tasks to others, enabling complex problem-solving and specialization within your fleet.
- **🛠️ Tool Integration**: Built-in tool management system using the Model Context Protocol (MCP) for standardized tool registration and execution via an intuitive decorator pattern.
- **🔌 Multi-Backend Support**: Pluggable architecture supporting OpenAI-compatible APIs and local Ollama models, allowing agents to use different AI providers.
- **🔒 Type-Safe Design**: Comprehensive type hints and Pydantic models ensure reliability and excellent IDE support.
- **📡 Message-Based Communication**: Structured, component-based message system for clear and extensible inter-agent and user-agent interactions.
- **⚙️ Extensible Architecture**: Designed for easy extension and customization of agents, backends, and fleet behaviors.

## Why `agnets`?

`agnets` stands out by treating multi-agent systems as a first-class citizen:

*   **Fleet-Centric Design**: Unlike frameworks where multi-agent is an add-on, `agnets` is built around the `Fleet` concept, making it natural to design organizations of AI agents.
*   **Structured Escalation Paths**: Easily define rules for how agents can delegate tasks or escalate queries to other specialized agents within the fleet, mimicking real-world organizational structures.
*   **Simplified Multi-Agent Logic**: The framework handles the complexities of inter-agent communication, allowing you to focus on the individual agent's expertise and the overall fleet structure.
*   **Backend Flexibility per Agent**: Different agents within the same fleet can use different AI backends (e.g., one agent uses a powerful cloud model for complex reasoning, another uses a local model for quick, specific tasks).
*   **Type Safety & Modern Python**: Leverages Python 3.12+ features and Pydantic for a development experience that is both productive and safe, catching errors early in complex multi-agent setups.

## Quick Start

### Prerequisites

- Python 3.12 or higher.
- An API key for an OpenAI-compatible service (like OpenAI itself) or Ollama running locally.
- The [UV](https://docs.astral.sh/uv/) package manager (recommended for faster installs).

### Installation

Install `agnets` with core dependencies:

```bash
pip install agnets
```

Or, using UV (recommended):

```bash
uv add agnets
```

To use with a specific backend, install the optional dependencies:

```bash
# For OpenAI-compatible backends
pip install "agnets[openai]"

# For Ollama
pip install "agnets[ollama]"
```

### Single Agent Example

Before diving into fleets, let's set up a single agent.

```python
import os
from agnets import Agent, Config
from agnets.backends.openai import OpenAICompatibleBackend

# os.environ["OPENAI_API_KEY"] = "your-api-key"

agent_config = Config(
    model_name="gpt-4o-mini",
    system_prompt="You are a helpful assistant."
)
backend = OpenAICompatibleBackend(config={"api_key": os.environ.get("OPENAI_API_KEY")})
agent = Agent(config=agent_config, backend=backend)

@agent.add_tool
def get_weather(location: str) -> str:
    """Gets the current weather for a given location."""
    return f"The weather in {location} is sunny and 25°C."

response = agent.invoke("What's the weather in Paris?")
print(response.content)
# Expected output: The weather in Paris is sunny and 25°C.
```

### Multi-Agent & Fleet System Example

This is where `agnets` truly shines. Let's create a small fleet of two agents: one for general queries and another specialized for calculations.

**1. Define your Specialist Agents (e.g., Calculator Agent)**
*(This could be in a separate file like `calculator.py`)*

```python
# calculator_agent.py
from agnets import Agent, Config
from agnets.backends.ollama import OllamaBackend # Or any backend

calc_config = Config(
    model_name="llama3.1:8b", # Or your preferred model
    system_prompt="You are a calculator agent. Only perform calculations."
)
calc_backend = OllamaBackend()
calculator_agent = Agent(config=calc_config, backend=calc_backend)

@calculator_agent.add_tool
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b

@calculator_agent.add_tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b
```

**2. Define your Main/Orchestrator Agent**

```python
# main_agent.py
from agnets import Agent, Config
from agnets.backends.openai import OpenAICompatibleBackend

main_config = Config(
    model_name="gpt-4o-mini",
    system_prompt="""
    You are a helpful assistant. If the user asks for a calculation,
    you MUST delegate it to the 'calculator_agent' using the available tools.
    When ready to reply to the user, use the 'respond_to_user' tool.
    """
)
main_backend = OpenAICompatibleBackend(config={"api_key": os.environ.get("OPENAI_API_KEY")})
main_agent = Agent(config=main_config, backend=main_backend)

# This tool is for the main agent to communicate its final answer back to the user/fleet orchestrator
@main_agent.add_tool
def respond_to_user(message: str) -> str:
    """Use this tool to provide the final response to the user."""
    return message
```

**3. Assemble and Manage the Fleet**

```python
# fleet_setup.py
from agnets.fleet import Fleet
from calculator_agent import calculator_agent # Assuming saved as above
from main_agent import main_agent # Assuming saved as above

# Initialize the fleet
my_fleet = Fleet()

# Add agents to the fleet, defining escalation paths
# 'main_agent' is allowed to escalate to 'calculator_agent'
my_fleet.add_agent('main_agent', main_agent, allowed_escalation_agent_names=['calculator_agent'])
my_fleet.add_agent('calculator_agent', calculator_agent) # No escalations needed for this specialist

# Now, interact with the fleet through the main agent
if __name__ == "__main__":
    user_input = "What is 15 times 32?"
    
    # The prompt can be structured; the agent will use its tools to interact.
    # The `stop_on` condition tells the fleet when the agent has finalized its response.
    response_messages = my_fleet.invoke_agent(
        "main_agent", 
        user_input, 
        stop_on=['respond_to_user']
    )
    
    # The final response from 'respond_to_user' will be in the last message's components
    final_response_component = response_messages[-1].components[-1]
    if hasattr(final_response_component, 'content') and hasattr(final_response_component.content, 'content'):
        print(f"Final Answer: {final_response_component.content.content[0].text}")
    else:
        print("No clear final response found.")

    # Expected Output:
    # Final Answer: 15 times 32 is 480.
```

## Core Concepts

*   **Agent**: An individual AI entity with a specific role, configuration (model, system prompt), backend, and a set of tools it can use.
*   **Backend**: An abstraction layer for communicating with different AI providers (e.g., OpenAI, Ollama). Allows agents to use various AI models seamlessly.
*   **Fleet**: The central orchestrator for multi-agent systems. It manages a collection of agents, defines who can talk to whom (escalation paths), and handles the routing of messages and tool calls between them.
*   **Tools**: Functions that agents can call to perform actions (e.g., calculate, fetch data, or in a fleet context, message another agent). Registered with `@agent.add_tool`.
*   **Messages & Components**: Structured data for communication. Messages can contain various components like text, agent thoughts, tool calls, and tool results, enabling rich interactions.
*   **Escalation Paths**: Defined when adding an agent to a fleet (`allowed_escalation_agent_names`). This dictates which other agents an agent is permitted to delegate tasks to or ask for help.

## Examples

The `examples/` directory provides practical demonstrations:

- **`calculator.py`**: A simple, single agent with basic math tools. Good for understanding individual agent setup.
- **`openai_example.py` & `ollama_example.py`**: Basic single-agent setups for specific backends.
- **`org_fleet_example.py`**: **(Highly Recommended)** A comprehensive example demonstrating how to build a mini-organization of agents. It shows how to set up a fleet, define escalation paths between agents (e.g., a generalist agent escalating to a specialist calculator agent), and manage interactions within the fleet. This example is the best illustration of the framework's multi-agent power.

We strongly recommend reviewing `org_fleet_example.py` to grasp the full potential of the `Fleet` system.

## Installation (Detailed)

For the most stable release, install from PyPI as shown in the Quick Start.

To install the latest development version from source:

```bash
git clone https://github.com/Sceptyre/agnets.git
cd agnets
pip install -e .
```

Or with UV:

```bash
git clone https://github.com/Sceptyre/agnets.git
cd agnets
uv pip install -e .
```

You can also install with specific backend dependencies from source:

```bash
pip install -e ".[openai,ollama]"
```

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get involved.

## License

This project is licensed under the GPLv3 License. See [LICENSE.md](LICENSE.md) for details.

## Contact & Support

For questions, bug reports, or feature requests, please open an issue on our [GitHub Issues page](https://github.com/Sceptyre/agnets/issues).
