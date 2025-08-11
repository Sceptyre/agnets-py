from ..types.backend import Backend
from ..types.message import Message, MessageComponent, MessageThinkingComponent, MessageToolCallComponent, MessageToolResultComponent

from typing import Dict

import mcp.server.fastmcp.tools

import json

import litellm
from litellm.types.completion import Function, ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, ChatCompletionToolMessageParam, ChatCompletionMessageToolCallParam, ChatCompletionMessageParam

def _map_to_litellm_system_message(message: Message) -> ChatCompletionSystemMessageParam:
    return ChatCompletionSystemMessageParam(
        role='system',
        content=message.components[0].content
    )

def _map_to_litellm_assistant_message(message: Message) -> ChatCompletionAssistantMessageParam | ChatCompletionToolMessageParam:
    msg = ChatCompletionAssistantMessageParam(
        role='assistant',
        content=None,
        tool_calls=[],
    )

    for component in message.components:
        if component.type == 'message':
            msg.content = component.content
            continue

        if component.type == 'tool_call':
            msg.tool_calls.append(ChatCompletionMessageToolCallParam(
                id = component.meta.get('tool_call_id'),
                type='function',
                function=Function(
                    name=component.content.params.name,
                    arguments=component.content.params.arguments
                )
            ))
            continue

        if component.type == 'tool_result':
            return ChatCompletionToolMessageParam(
                role = 'tool',
                tool_call_id=component.meta.get('tool_call_id'),
                content= component.content.content[0].text
            )

    return msg

def _map_to_litellm_user_message(message: Message) -> ChatCompletionUserMessageParam:
    return ChatCompletionUserMessageParam(
        role='user',
        content=message.components[0].content
    )

def _map_to_litellm_message(message: Message) -> ChatCompletionMessageParam:
    switch = {
        'system': _map_to_litellm_system_message,
        'assistant': _map_to_litellm_assistant_message,
        'user': _map_to_litellm_user_message
    }

    return switch[message.role](message)

def _map_to_litellm_tool(tool: mcp.server.fastmcp.tools.Tool) -> Dict:
    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description,
            'parameters': tool.parameters
        }
    }

def _map_from_litellm_message(message: litellm.Message) -> Message:
    if message.role == 'tool':
        return Message(role='user', components=[
            MessageToolResultComponent(
                type='tool_result',
                content=mcp.types.CallToolResult(
                    meta={
                        'tool_call_id': message.tool_call_id
                    },
                    content=mcp.types.TextContent(
                        text=message.content
                    )
                )
            )
        ])

    msg = Message(role=message.role, components=[])

    if message.content:
        msg.components.append(MessageComponent(
            type='message',
            content=message.content
        ))

    if message.provider_specific_fields.get('reasoning'):
        msg.components.append(MessageThinkingComponent(
            type='thinking',
            content=message.provider_specific_fields.get('reasoning')
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

        response = litellm.completion(
            model=agent_config.model_name,
            messages=mapped_messages,
            tools=tools_mapped,
            api_key=self.config.get('api_key', None),
            base_url=self.config.get('base_url', None),
        )

        return _map_from_litellm_message(response.choices[0].message)
