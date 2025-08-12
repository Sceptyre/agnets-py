from pydantic import BaseModel, Field
from mcp.server.fastmcp.tools import ToolManager
import mcp

from typing import Any, List, Dict

from .types.backend import Backend
from .types.message import Message, MessageComponent, MessageToolResultComponent, MessageToolCallComponent
from .config import Config
from .utils import map_fastmcp_tool_to_openai_tool

import json

import logging
logger = logging.getLogger(__name__)

import litellm
import litellm.types.completion
import litellm.types.utils

class Agent(BaseModel):
    config: Config = Field(default_factory=Config)
    litellm_config: Dict = Field(default_factory=dict)

    __tool_manager: ToolManager

    def model_post_init(self, ctx):
        self.__tool_manager = ToolManager()

        # if self.backend is None:
        #     from .backends.litellm import LiteLLMBackend
        #     self.backend = LiteLLMBackend()

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

    def _workaround_map_message_to_tool_call(self, workaround_message: Message) -> Message:
        # Last message should always be a content block
        workaround_message_component = workaround_message.components[-1]

        if workaround_message_component.type != 'message':
            return workaround_message

        try:
            # Attempt to extract a json object
            json_content = workaround_message_component.content.removeprefix('```').removeprefix('json').removesuffix('```')

            tool_call_obj = json.loads(json_content)
        
            return Message(
                role = 'assistant',
                components  = [
                    MessageToolCallComponent(
                        type='tool_call',
                        content=mcp.types.CallToolRequest(
                            method = 'tools/call',
                            params = mcp.types.CallToolRequestParams(
                                name = tool_call_obj['tool_call']['tool_name'],
                                arguments = tool_call_obj['tool_call']['input']
                            )
                        )
                    )
                ]
            )
        except Exception as _:
            return workaround_message

    

    def _invoke_completion(self, messages: List[litellm.types.utils.Message], stop_on: List[str], force_tools: bool = True) -> List[Message]:
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

        # PREPARE LLM RESPONSE ARGS
        # _kvargs = {
        #     'agent_config': self.config,
        # }

        # if self.config.do_unsupported_model_workaround:
        #     system_prompt = self._workaround_system_prompt(tools)
        # else: 
        #     system_prompt = self.config.system_prompt

        while True:
            print(messages)
            # # PREPARE LLM RESPONSE ARGS
            # _kvargs['messages'] = messages


            # else:
            #     _kvargs['tools'] = tools
                
            # GENERATE RESPONSE
            logger.debug(f"Generating response via litellm ({self.config.model_name})")
            response = litellm.completion(
                model=self.config.model_name, 
                messages=messages, 
                tools=tools_mapped,
                **self.litellm_config,
                stream=False
            )
            response_message = response.choices[0].message
            messages.append(response_message)
            logger.debug(f"Received response via litellm ({self.config.model_name}): {response}")
            print(response_message)

            has_tool_call = response_message.tool_calls and len(response_message.tool_calls) > 0

            # RETURN IF NO TOOL USE REQUIRED
            if not force_tools:
                return messages

            # # HANDLE WORKAROUND
            # if force_tools and self.config.do_unsupported_model_workaround:
            #     response = self._workaround_map_message_to_tool_call(response)


            # TOOL USE ENFORCEMENT
            if force_tools and not has_tool_call:
                logger.debug(f"`force_tools` enabled and no tool response received. Re-prompting....")
                messages.append(
                    litellm.types.utils.Message(
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

#                 if self.config.do_unsupported_model_workaround:
#                     messages.append(
#                         {
#                             'tool_call_id': tool_call_id,
#                             'role': 'tool',
#                             'content': str(result)
#                         }
#                         lite
                        
#                         Message(
#                         role='system',
#                         components=[
#                             MessageComponent(
#                                 type='message',
#                                 content=f"""<tool_call_result>
# <{tool_call.content.params.name}>
# {result}
# </{tool_call.content.params.name}>
# </tool_call_result>""",
#                             )
#                         ]
#                     ))

#                 else: 
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

    def invoke(self, user_message: str, stop_on: List[str] = [], force_tools: bool = True) -> List[Message]:
        messages = [
            {
                'role': 'user',
                'content': user_message
            }
        ]

        return self._invoke_completion(messages=messages, stop_on=stop_on, force_tools=force_tools)


# Alias
Agnet = Agent