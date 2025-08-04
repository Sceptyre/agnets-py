from openai import OpenAI
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

from ..types.backend import Backend
from ..types.message import Message

class OpenAICompatibleBackend(Backend):
    def model_post_init(self, ctx):
        self._client = OpenAI(
            api_key=self.config.get('OPENAI_API_KEY'),
            base_url=self.config.get('OPENAI_BASE_URL')
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
            mapped_messages.append({
                'role': message.role,
                'content': message.message_content
            })

        tools_mapped = []
        for tool in tools:
            print(tool)
            tools_mapped.append(
                ChatCompletionToolParam(
                    type='function',
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters
                    )
                )
            )
            

        response = self._client.chat.completions.create(
            model=agent_config.model_name,
            messages=mapped_messages,
            tools=tools_mapped
        )

        output = []
        
        if response.choices[0].message.reasoning:
            output.append(Message(
                role='assistant',
                message_type='thinking',
                message_content=response.choices[0].message.reasoning
            ))

        if response.choices[0].message.content:
            output.append(Message(
                role='assistant',
                message_type='message',
                message_content=response.choices[0].message.content
            ))

        for tool_call in response.choices[0].message.tool_calls:
            output.append(Message(
                id=tool_call.id,
                role='assistant',
                message_type='tool_call',
                message_content=tool_call
            ))


        return output
