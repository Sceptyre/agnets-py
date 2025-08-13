from pydantic import BaseModel, Field
from mcp.server.fastmcp.tools import ToolManager

from typing import Any, List, Dict

from .config import Config
from .utils import map_fastmcp_tool_to_openai_tool

import json

import litellm

import logging
logger = logging.getLogger(__name__)

class Agent(BaseModel):
    config: Config = Field(default_factory=Config)
    router: litellm.Router

    __tool_manager: ToolManager

    model_config = {
        'arbitrary_types_allowed': True
    }

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

        if not tool:
            logger.debug(f"Agent requested unknown tool {tool_name}")
            return f"ERROR: Unknown tool: '{tool_name}'"
        
        try:
            tool_result = tool.fn(*args, **kvargs)
        except Exception as err:
            logger.error(f"Exception occured while calling {tool_name}: {err}")
            return f"ERROR: Exception occured while calling {tool_name}: {err}"

        return tool_result
    

    def _invoke_completion(self, messages: List[litellm.Message], stop_on: List[str], force_tools: bool = True) -> List[litellm.Message]:
        if len(stop_on) == 0 and force_tools:
            raise Exception(f"Empty `stop_on` and `force_tools` = True not supported.")

        tools = self.list_tools()
        tools_mapped = [map_fastmcp_tool_to_openai_tool(t) for t in tools]

        messages = [
            {
                'role': 'system',
                'content': self.config.system_prompt
            },
            *messages
        ]

        while True:
            # GENERATE RESPONSE
            logger.debug(f"Generating response via litellm ({self.config.model_name})")
            [print(m) for m in messages]
            response = self.router.completion(
                model=self.config.model_name, 
                messages=messages, 
                tools=tools_mapped,
                stream=False
            )
            response_message = response.choices[0].message
            messages.append(response_message)
            logger.debug(f"Received response via litellm ({self.config.model_name}): {response}")

            has_tool_call = response_message.tool_calls and len(response_message.tool_calls) > 0

            # RETURN IF NO TOOL USE REQUIRED
            if not force_tools:
                return messages


            # TOOL USE ENFORCEMENT
            if force_tools and not has_tool_call:
                logger.debug(f"`force_tools` enabled and no tool response received. Re-prompting....")
                messages.append(
                    litellm.Message(
                        role='user',
                        content="ERROR: Calling a tool is REQUIRED"
                    )
                )
                continue

            # EXECUTE TOOLS
            for tool_call in response_message.tool_calls:
                tool_call_id = tool_call.id
                try:
                    tool_call_args = json.loads(tool_call.function.arguments)
                except Exception as err:
                    messages.append(
                        {
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': str(err)
                        }
                    )
                    continue

                logger.debug(f"Invoking tool_call({tool_call_id}) ({tool_call})")
                result = self._call_tool(tool_call.function.name, **tool_call_args)
                logger.debug(f"Result of tool_call({tool_call_id}) ({result})")

                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': str(result)
                    }
                )


                # TOOL STOP
                if tool_call.function.name in stop_on:
                    logger.debug(f"Stopping after tool_call({tool_call_id}) ({tool_call.function})")
                    return messages

    def invoke(self, user_message: str, stop_on: List[str] = [], force_tools: bool = True) -> List[litellm.Message]:
        messages = [
            {
                'role': 'user',
                'content': user_message
            }
        ]

        return self._invoke_completion(messages=messages, stop_on=stop_on, force_tools=force_tools)


# Alias
Agnet = Agent