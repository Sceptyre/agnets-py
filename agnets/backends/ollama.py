import ollama

from ..types.backend import Backend
from ..types.message import Message


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
            msg_builder['message'] = component.content
        
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
            msg_builder['content'] = f"<{component._meta.get('tool_name')}>{component.content.content[0]}</{component._meta.get('tool_name')}>"
        
        return ollama.Message.model_validate(msg_builder)


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
        print(ollama_response)

        return [Message(
            role='assistant',
            message_type='message',
            message_content=ollama_response['message']['content']
        )]
