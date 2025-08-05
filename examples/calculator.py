from agnets import Agnet, Config
from agnets.backends.openai import OpenAICompatibleBackend

agnet = Agnet(
    config=Config(
        model_name="z-ai/glm-4-32b",
        system_prompt="ALWAYS use tools available to respond to the user."
    ),
    backend=OpenAICompatibleBackend()
)

@agnet.add_tool
def add(a: int, b: int) -> int: 
    return a + b

@agnet.add_tool
def multiply(a: int, b: int) -> int: 
    return a * b

@agnet.add_tool
def respond_to_user(response: str):
    """
    Responds to user
    """
    print(response)

    return True

user_input = input(">>> ")

agnet.invoke(user_input, stop_on=['respond_to_user'])