from agnets import Agnet, Config
from typing import Literal
from agnets.backends.ollama import OllamaBackend

agnet = Agnet(
    config=Config(
        model_name="phi4-mini:latest",
        do_unsupported_model_workaround=True
    ),
    backend=OllamaBackend()
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