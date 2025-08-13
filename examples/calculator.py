from agnets import Agnet, Config
from agnets.agentv2 import Agent
from agnets.backends.openai import OpenAICompatibleBackend
# from agnets.backends.ollama import OllamaBackend
from agnets.backends.litellm import LiteLLMBackend

agnet = Agent(
    config=Config(
        model_name="openai/openai/gpt-oss-20b",
        system_prompt="""
ALWAYS use tools available to respond to your fellow agent. 
Your response MUST be in a tool call.
Present to the user your answer with the `respond_to_user` tool.
"""
    ),
)

@agnet.add_tool
def add(a: int, b: int) -> int: 
    return a + b

@agnet.add_tool
def multiply(a: int, b: int) -> int: 
    return a * b

if __name__ == "__main__":
    @agnet.add_tool
    def respond_to_user(response: str):
        """
        Responds to user
        """
        print(response)

        return True

    user_input = input(">>> ")

    agnet.invoke(user_input, stop_on=['respond_to_user'])