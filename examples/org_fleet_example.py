from agnets.fleet import Fleet
from agnets.backends.openai import OpenAICompatibleBackend


example_fleet = Fleet()


def setup_ag1():
    from agnets import Agent, Config
    from agnets.backends.ollama import OllamaBackend

    ob1 = OllamaBackend(config={})

    ag1 = Agent(
        config=Config(
            model_name="phi4-mini:3.8b",
            system_prompt="YOU MUST USE A TOOL"
        ), 
        backend=ob1
    )

    @ag1.add_tool
    def respond_to_user(message: str):
        print(message)

        return "SUCCESS"
    
    return ag1

ag1 = setup_ag1()

example_fleet.add_agent('agent_one', ag1)

example_fleet.invoke_agent("agent_one", "invoke the respond_to_user tool please")