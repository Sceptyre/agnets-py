import ollama

from ..types.backend import Backend
from ..types.message import Message, MessageComponent, MessageThinkingComponent, MessageToolCallComponent

import mcp.types


# HELPERS
def _map_to_ollama_tool(tool) -> ollama.Tool:
    return ollama.Tool(
        type='function',
        function=ollama.Tool.Function(
            name=tool.name,
            description=tool.description,
            parameters=ollama.Tool.Function.Parameters.model_validate(tool.parameters)
        )
    )

def _map_to_ollama_message(message: Message) -> ollama.Message:
    msg_builder = {
        "role": message.role,
        "content": '',
        "thinking": None,
        "tool_calls": []
    }

    for component in message.components:
        if component.type == 'message':
            msg_builder['content'] = component.content
        
        if component.type == 'thinking':
            msg_builder['thinking'] = component.content
        
        if component.type == 'tool_call':
            msg_builder['tool_calls'].append(
                ollama.Message.ToolCall(
                    function=ollama.Message.ToolCall.Function(
                        name=component.content.params.name,
                        arguments=component.content.params.arguments
                    )
                )
            )
        
        if component.type == 'tool_result':
            msg_builder['content'] = f"<{component.meta.get('tool_call_name')}>{component.content.content[0]}</{component.meta.get('tool_call_name')}>"
        
        return ollama.Message.model_validate(msg_builder)


def _map_from_ollama_message(ollama_message: ollama.Message) -> Message:
    msg = Message(role=ollama_message.role, components=[])

    if ollama_message.thinking:
        msg.components.append(MessageThinkingComponent(
            type='thinking',
            content=ollama_message.thinking
        ))

    if ollama_message.content:
        msg.components.append(MessageComponent(
            type='message',
            content=ollama_message.content
        ))

    for tool_call in ollama_message.tool_calls or []:
        msg.components.append(MessageToolCallComponent(
            type='tool_call',
            content=mcp.types.CallToolRequest(
                method='tools/call',
                params=mcp.types.CallToolRequestParams(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments
                )
            )
        ))

    return msg



class OllamaBackend(Backend):
    def model_post_init(self, ctx):
        self._client = ollama.Client(
            host="127.0.0.1:11434"
        )

    def generate_response(self, messages, agent_config, tools = [], system_prompt_override: str = "", **kvargs):
        system_prompt = system_prompt_override or agent_config.system_prompt

        mapped_messages=[
            {
                'role': 'system',
                'content': system_prompt
            }
        ]

        for message in messages:
            mapped_messages.append(_map_to_ollama_message(message))

        tools_mapped = []
        for tool in tools:
            tools_mapped.append(_map_to_ollama_tool(tool))

        ollama_response = self._client.chat(
            model=agent_config.model_name,
            messages=mapped_messages,
            tools=tools_mapped,
            think=agent_config.do_thinking
        )

        return _map_from_ollama_message(ollama_response.message)
