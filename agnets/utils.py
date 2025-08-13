from typing import Dict, Literal
from enum import Enum
from mcp.server.fastmcp.tools import Tool as FastMCPTool
import litellm

from .types import literals

def map_fastmcp_tool_to_openai_tool(tool: FastMCPTool) -> Dict:
    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description,
            'parameters': tool.parameters
        }
    }

def new_simple_litellm_router(provider_name: literals.LITELLM_PROVIDERS_TYPE, model_name: str, litellm_params: Dict = {}) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                'model_name': model_name,
                'litellm_params': {
                    **litellm_params,
                    'model': f"{provider_name}/{model_name}"
                }
            }
        ]
    )