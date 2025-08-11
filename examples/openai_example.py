from agnets import Agnet, Config
from typing import Literal
from agnets.backends.litellm import LiteLLMBackend

import os

agnet = Agnet(
    config=Config(
        model_name="openai/z-ai/glm-4-32b"
    ),
    backend=LiteLLMBackend()
)

@agnet.add_tool
def my_agnet_method(agnet: Literal['name', 'othername']) -> str: 
    print(agnet)

    return {}

@agnet.add_tool
def respond_to_user(message: str) -> str: 
    print(message)

    return {}


res = agnet.invoke("hello world", stop_on=['respond_to_user'])
print(res)
