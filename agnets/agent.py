from pydantic import BaseModel, Field
from mcp.server.fastmcp.tools import ToolManager
import mcp

from typing import Any, List

from .types.backend import Backend
from .types.message import Message
from .config import Config

import json

class Agent(BaseModel):
    config: Config = Field(default_factory=Config)
    backend: Backend

    __tool_manager: ToolManager

    def model_post_init(self, ctx):
        self.__tool_manager = ToolManager()

    @property
    def add_tool(self): 
        return self.__tool_manager.add_tool

    @property
    def list_tools(self): 
        return self.__tool_manager.list_tools

    def _call_tool(self, tool_name: str, *args, **kvargs) -> Any:
        tool = self.__tool_manager.get_tool(tool_name)
        
        return tool.fn(*args, **kvargs)

    def _workaround_system_prompt(self, tools: List[mcp.Tool]) -> str:
        system_prompt = self.config.system_prompt

        system_prompt += """# TOOL CALLING
TOOL CALLS ARE PERFORMED IN JSON OBJECTS THAT LOOK AS FOLLOWS:
{{
    "action": "tool_call",
    "tool_call": {{
        "tool_name": "<tool name>",
        "input": <input object/data>
    }}
}}
Responses that contain anything more WILL BE REJECTED.


# TOOLS
"""
        for tool in tools:
            system_prompt += f"""
tool_name: {tool.name}
{tool.description}

input_schema: 
{json.dumps(tool.parameters)}

outputs:
{tool.fn_metadata.output_schema}"""
        return system_prompt

    def invoke(self, user_message: str) -> Any:
        tools = self.list_tools()

        messages = [
            Message(
                role='user',
                message_type='message',
                message_content=user_message
            )
        ]
        
        if self.config.do_unsupported_model_workaround:
            system_prompt = self._workaround_system_prompt(tools)
            return self.backend.generate_response(messages, self.config, system_prompt_override=system_prompt)

        else:
            return self.backend.generate_response(messages, self.config, tools=tools)


# Alias
Agnet = Agent