from pydantic import BaseModel, Field
from mcp.server.fastmcp.tools import ToolManager
import mcp

from typing import Any, List

from .types.backend import Backend
from .types.message import Message, MessageComponent, MessageToolResultComponent, MessageToolCallComponent
from .config import Config

import json

import logging
logger = logging.getLogger(__name__)

class Agent(BaseModel):
    config: Config = Field(default_factory=Config)
    backend: Backend | None

    __tool_manager: ToolManager

    def model_post_init(self, ctx):
        self.__tool_manager = ToolManager()

        if self.backend is None:
            from .backends.litellm import LiteLLMBackend
            self.backend = LiteLLMBackend()

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

    

    def _invoke_completion(self, messages: List[Message], stop_on: List[str], force_tools: bool = True) -> List[Message]:
        if len(stop_on) == 0 and force_tools:
            raise Exception(f"Empty `stop_on` and `force_tools` = True not supported.")

        tools = self.list_tools()

        # PREPARE LLM RESPONSE ARGS
        _kvargs = {
            'agent_config': self.config,
        }

        while True:
            # PREPARE LLM RESPONSE ARGS
            _kvargs['messages'] = messages

            if self.config.do_unsupported_model_workaround:
                _kvargs['system_prompt_override'] = self._workaround_system_prompt(tools)

            else:
                _kvargs['tools'] = tools
                
            # GENERATE RESPONSE
            logger.debug(f"Generating response from backend({self.backend.__class__.__qualname__}): {_kvargs}")
            response =  self.backend.generate_response(**_kvargs)
            messages.append(response)
            logger.debug(f"Received response from backend({self.backend.__class__.__qualname__}): {response}")

            # RETURN IF NO TOOL USE REQUIRED
            if not force_tools:
                return messages

            # HANDLE WORKAROUND
            if force_tools and self.config.do_unsupported_model_workaround:
                response = self._workaround_map_message_to_tool_call(response)


            # TOOL USE ENFORCEMENT
            if force_tools and response.components[-1].type != 'tool_call':
                logger.debug(f"`force_tools` enabled and no tool response received. Re-prompting....")
                messages.append(Message(role='user', components=[MessageComponent(type='message', content="ERROR: Calling a tool is REQUIRED")]))
                continue

            # EXECUTE TOOLS
            for tool_call in filter(lambda x: x.type == 'tool_call', response.components):
                tool_call_id = tool_call.meta.get('tool_call_id')

                logger.debug(f"Invoking tool_call({tool_call_id}) ({tool_call.content.params})")
                result = self._call_tool(tool_call.content.params.name, **tool_call.content.params.arguments)
                logger.debug(f"Result of tool_call({tool_call_id}) ({result})")

                if self.config.do_unsupported_model_workaround:
                    messages.append(Message(
                        role='system',
                        components=[
                            MessageComponent(
                                type='message',
                                content=f"""<tool_call_result>
<{tool_call.content.params.name}>
{result}
</{tool_call.content.params.name}>
</tool_call_result>""",
                            )
                        ]
                    ))

                else: 
                    messages.append(Message(
                        role='system',
                        components=[
                            MessageToolResultComponent(
                                meta={
                                    'tool_call_id': tool_call_id, 
                                    'tool_call_name': tool_call.content.params.name
                                },
                                type='tool_result',
                                content=mcp.types.CallToolResult(
                                    content=[mcp.types.TextContent(type='text', text=str(result))]
                                ),
                            )
                        ]
                    ))


                # TOOL STOP
                if tool_call.content.params.name in stop_on:
                    logger.debug(f"Stopping after tool_call({tool_call_id}) ({tool_call.content.params})")
                    return messages

    def invoke(self, user_message: str, stop_on: List[str] = [], force_tools: bool = True) -> List[Message]:
        messages = [
            Message(
                role='user',
                components=[MessageComponent(
                    type='message',
                    content=user_message
                )],
            )
        ]

        return self._invoke_completion(messages=messages, stop_on=stop_on, force_tools=force_tools)


# Alias
Agnet = Agent