from agnets import Agnet, Config
from typing import Literal
from agnets.backends.openai import OpenAICompatibleBackend

import os

agnet = Agnet(
    config=Config(
        model_name="qwen/qwen3-14b"
    ),
    backend=OpenAICompatibleBackend(
    )
)

@agnet.add_tool
def my_agnet_method(agnet: Literal['name', 'othername']) -> str: 
    print(agnet)

    return {}

@agnet.add_tool
def respond_to_user(message: str) -> str: 
    print(message)

    return {}


res = agnet.invoke("hello world")
print(res)
