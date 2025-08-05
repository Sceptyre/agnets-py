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

    
        @agent.add_tool
        def respond_to_agent(response: str) -> str:
            """
            Reply to an agent that prompted you
            """
            return response

        if len(allowed_escalation_agent_names) == 0:
            return

        KNOWN_AGENTS_TYPE = Literal[*allowed_escalation_agent_names]

        @agent.add_tool
        def ask_agent(agent_name: KNOWN_AGENTS_TYPE, query: str, context: str):
            """
            Ask a known agent a question
            """
            if agent_name not in allowed_escalation_agent_names:
                return f"ERROR: '{agent_name}' not in {allowed_escalation_agent_names}"

            return self.agents.get(agent_name).invoke(f"Question: {query}\nContext: {context}", stop_on=['respond_to_agent'], force_tools=True)

    def invoke_agent(self, agent_name: str, query: str, stop_on: List[str] = []):
        return self.agents.get(agent_name).invoke(query, stop_on = stop_on, force_tools=True)