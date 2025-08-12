from typing import Dict
from mcp.server.fastmcp.tools import Tool as FastMCPTool

def map_fastmcp_tool_to_openai_tool(tool: FastMCPTool) -> Dict:
    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description,
            'parameters': tool.parameters
        }
    }