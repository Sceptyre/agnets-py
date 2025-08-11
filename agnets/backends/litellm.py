# from openai import OpenAI
# from openai.types.chat.chat_completion_message import ChatCompletionMessage
# from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
# from openai.types.shared_params.function_definition import FunctionDefinition

from ..types.backend import Backend
from ..types.message import Message, MessageComponent, MessageThinkingComponent, MessageToolCallComponent

from typing import Dict

import mcp.server.fastmcp.tools

import json

from litellm import completion, LiteLLM
from litellm.types.completion import ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, ChatCompletionToolMessageParam, ChatCompletionMessageParam

def _map_to_litellm_message(message: Message) -> Dict:
    msg = {
        "role": message.role,
    }

    for component in message.components:
        if component.type == 'message':
            msg['content'] = component.content
            continue

        if component.type == 'thinking':
            msg['thinking'] = component.content
            continue

        if component.type == 'tool_call':
            if not msg.get('tool_calls'):
                msg['tool_calls'] = []

            msg['tool_calls'].append({
                'id': component.meta.get('tool_call_id'),
                'type': 'function',
                'function': {
                    'name': component.content.params.name,
                    'arguments': json.dumps(component.content.params.arguments)
                }
            })
            continue

        if component.type == 'tool_result':
            return {
                'role': 'tool',
                'tool_call_id': component.meta.get('tool_call_id'),
                'content': component.content.content[0].text
            }

    return msg

def _map_to_litellm_tool(tool: mcp.server.fastmcp.tools.Tool) -> ChatCompletionMessageParam:
    ChatCompletionToolMessageParam()
    return ChatCompletionToolParam(
        type='function',
        function=FunctionDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters
        )
    )

def _map_from_litellm_message(message: ChatCompletionMessageParam) -> Message:
    msg = Message(role=message.role, components=[])

    if message.content:
        msg.components.append(MessageComponent(
            type='message',
            content=message.content
        ))

    if message.reasoning:
        msg.components.append(MessageThinkingComponent(
            type='thinking',
            content=message.reasoning
        ))

    for tool_call in message.tool_calls or []:
        msg.components.append(MessageToolCallComponent(
            meta={'tool_call_id': tool_call.id},
            type='tool_call',
            content=mcp.types.CallToolRequest(
                method='tools/call',
                params=mcp.types.CallToolRequestParams(
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments)
                )
            )
        ))

    return msg

class LiteLLMBackend(Backend):
    def generate_response(self, messages, agent_config, tools = [], system_prompt_override: str = "", **kvargs):
        system_prompt = system_prompt_override or agent_config.system_prompt

        mapped_messages=[
            ChatCompletionSystemMessageParam(
                role    = 'system',
                content = system_prompt
            )
        ]

        for message in messages:
            mapped_messages.append(_map_to_litellm_message(message))

        tools_mapped = []
        for tool in tools:
            tools_mapped.append(_map_to_litellm_tool(tool))

        response = completion(
            model=agent_config.model_name,
            messages=mapped_messages,
            tools=tools_mapped
        )

        return _map_from_litellm_message(response.choices[0].message)
