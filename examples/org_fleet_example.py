from agnets.fleet import Fleet
from agnets.backends.openai import OpenAICompatibleBackend

import logging
logging.basicConfig(level=logging.DEBUG)


example_fleet = Fleet()


def setup_ag1():
    from agnets import Agent, Config
    from agnets.backends.ollama import OllamaBackend

    ob1 = OpenAICompatibleBackend(config={})

    ag1 = Agent(
        config=Config(
            model_name="z-ai/glm-4-32b",
            system_prompt="""
ALWAYS prefer an agent for answering a question before generating your own response. 
All responses must include a tool call. 
When ready to reply to the user, use the 'respond_to_user' tool with the response. 
The user can not see anything outside of what is in the tool call.
"""
        ), 
        backend=ob1
    )

    @ag1.add_tool
    def respond_to_user(message: str):
        return message
    
    return ag1

ag1 = setup_ag1()

example_fleet.add_agent('agent_one', ag1, allowed_escalation_agent_names=['calculator_agent'])

from calculator import agnet as calculator_agent

example_fleet.add_agent('calculator_agent', calculator_agent)

while True:
    message_history = []

    user_input = input(">>> ")
    if user_input.lower() == "exit":
        exit()

    message_history.append(f"User: {user_input}")

    prompt = "<chat_history>\n"
    for m in message_history:
        prompt += f"<message>\n{m}\n</message>\n"
    prompt += "</chat_history>\n"

    prompt += f"<newest_message>\n{user_input}\n</newest_message>"

    res = example_fleet.invoke_agent("agent_one", prompt, stop_on=['respond_to_user'])
    agent_response_message = res[-1].components[-1].content.content[0].text

    message_history.append(f"agent_one: {agent_response_message}")

    print(agent_response_message)