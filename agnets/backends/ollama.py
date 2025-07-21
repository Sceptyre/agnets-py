from ollama import Client

from ..types.backend import Backend
from ..types.message import Message

class OllamaBackend(Backend):
    def model_post_init(self, ctx):
        self._client = Client(
            host="127.0.0.1:11434"
        )

    def generate_response(self, messages, agent_config, tools = [], system_prompt_override: str = ""):
        system_prompt = system_prompt_override or agent_config.system_prompt

        mapped_messages=[
            {
                'role': 'system',
                'content': system_prompt
            }
        ]

        for message in messages:
            mapped_messages.append({
                'role': message.role,
                'content': message.message_content
            })


        ollama_response = self._client.chat(
            model=agent_config.model_name,
            messages=mapped_messages,
            tools=tools,
            think=agent_config.do_thinking
        )

        return [Message(
            role='assistant',
            message_type='message',
            message_content=ollama_response['message']['content']
        )]
