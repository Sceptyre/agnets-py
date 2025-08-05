from agnets.fleet import Fleet
from agnets.backends.openai import OpenAICompatibleBackend


example_fleet = Fleet()


def setup_ag1():
    from agnets import Agent, Config
    from agnets.backends.ollama import OllamaBackend

    ob1 = OpenAICompatibleBackend(config={})

    ag1 = Agent(
        config=Config(
            model_name="z-ai/glm-4-32b",
            system_prompt="Work with your agents to answer user questions. Do not ask the user permission to use a tool. The agents are your peers."
        ), 
        backend=ob1
    )

    @ag1.add_tool
    def respond_to_user(message: str):
        print(message)

        return "SUCCESS"
    
    return ag1

ag1 = setup_ag1()

example_fleet.add_agent('agent_one', ag1, allowed_escalation_agent_names=['calculator_agent'])

from calculator import agnet as calculator_agent

example_fleet.add_agent('calculator_agent', calculator_agent)

user_input = input(">>> ")

res = example_fleet.invoke_agent("agent_one", user_input, stop_on=['respond_to_user'])
print(res)