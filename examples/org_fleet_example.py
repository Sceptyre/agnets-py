from agnets.fleet import Fleet
from agnets.backends.openai import OpenAICompatibleBackend


example_fleet = Fleet()


def setup_ag1():
    from agnets import Agent, Config
    from agnets.backends.ollama import OllamaBackend

    ob1 = OpenAICompatibleBackend(config={})

    ag1 = Agent(
        config=Config(
            model_name="google/gemini-2.5-flash",
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

res = example_fleet.invoke_agent("agent_one", "what functions are available to you?")
print(res)