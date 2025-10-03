"""
NOTE: Requires basic-memory to be install via
uv pip install basic-memory
"""

from agnets.agentv2 import Agent
from agnets.config import Config
from agnets.utils import new_simple_litellm_router
from agnets.fleet import Fleet
import logging

# logging.basicConfig(level=logging.DEBUG)


from basic_memory.mcp.server import mcp as basic_memory_mcp

import basic_memory.mcp.tools

# print(basic_memory_mcp._tool_manager._tools)

f = Fleet()

def new_simple_agent(agent_name: str, prompt: str) -> Agent:
    r = new_simple_litellm_router(
        provider_name='openai',
        model_name='openai/gpt-oss-120b'
    )

    c = Config(
        model_name='openai/gpt-oss-120b',
        system_prompt=f"""
You are {agent_name}

{prompt}
"""
    )

    a = Agent(router=r,config=c)

    return a


agents = [
    ('Chief Executive Officer', '', ['Chief Technology Officer', 'Chief Financial Officer', 'Chief Information Security Officer']),
    ('Chief Technology Officer', '', ['Director of IT']),
    ('Chief Financial Officer', '', []),
    ('Chief Information Security Officer', '', []),
    ('Director of IT', '', ['Network Manager', 'Endpoint Manager', 'Collaboration Manager', 'Telecom Manager']),
    ('Network Manager', '', []),
    ('Endpoint Manager', '', []),
    ('Collaboration Manager', '', []),
    ('Telecom Manager', '', []),
]


def main():
    for a in agents:
        agent = new_simple_agent(a[0], a[1])

        f.add_agent(a[0], agent, a[2])

    CEO_AGENT = f.agents.get('Chief Executive Officer')
    @CEO_AGENT.add_tool
    def respond_to_user(response: str):
        return response

    for k, v in basic_memory_mcp._tool_manager._tools.items():
        CEO_AGENT.set_tool(k, v)

    response = CEO_AGENT.invoke('We need to establish a security strategy for any new products we release. Use your memory', stop_on=['respond_to_user'])
    response_content = response[-1]['content']
    print(response_content)

if __name__ == '__main__':
    main()
