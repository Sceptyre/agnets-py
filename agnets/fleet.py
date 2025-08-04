from pydantic import BaseModel, Field
from .agent import Agent
from typing import Dict, List, Literal


class Fleet(BaseModel):
    agents: Dict[str, Agent] = Field(default_factory=dict)

    _relationships: Dict[str, List[str]]

    def model_post_init(self, context):
        self._relationships = {}
        return super().model_post_init(context)

    def add_agent(self, agent_name: str, agent: Agent, allowed_escalation_agent_names: List[str] =[]):
        self.agents[agent_name] = agent
        self._relationships[agent_name] = allowed_escalation_agent_names

        KNOWN_AGENTS_TYPE = Literal[allowed_escalation_agent_names]

        print(agent)

        @agent.add_tool
        def ask_agent(agent_name: KNOWN_AGENTS_TYPE, query: str, context: str):
            return self.agents.get(agent_name).invoke(f"Question: {query}\nContext: {context}")

    def invoke_agent(self, agent_name: str, query: str):
        return self.agents.get(agent_name).invoke(query)