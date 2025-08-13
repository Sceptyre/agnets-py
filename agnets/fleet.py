from pydantic import BaseModel, Field
from .agentv2 import Agent
from .types.message import Message, MessageComponent
from typing import Dict, List, Literal, Any, Annotated

import logging
logger = logging.getLogger(__name__)


class FleetDialogBranch(BaseModel):
    meta: Dict[str, Any]

    operating_agent: Agent

    messages: Annotated[List[Dict], Field(default_factory=list)]

    def invoke_agent(self, query: str) -> List[Dict]:
        self.messages.append(
            {
                'role': 'user',
                'content': f"<meta>\n{self.meta}\n</meta><query>{query}</query>"
            }
        )

        self.operating_agent._invoke_completion(self.messages, stop_on=['respond_to_agent'], force_tools=True)

        return self.messages



class Fleet(BaseModel):
    agents: Dict[str, Agent] = Field(default_factory=dict)

    _relationships: Dict[str, List[str]]

    _dialog_branches: Dict[str, FleetDialogBranch]

    def model_post_init(self, context):
        self._relationships = {}
        self._dialog_branches = {}
        return super().model_post_init(context)

    def add_agent(self, agent_name: str, agent: Agent, allowed_escalation_agent_names: List[str] =[]):
        self.agents[agent_name] = agent
        self._relationships[agent_name] = allowed_escalation_agent_names

        _agent_name = agent_name
    
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
            Ask a known specialized agent a question. Work with agents that have a more specific or applicable role than you may have
            """
            if agent_name not in allowed_escalation_agent_names:
                return f"ERROR: '{agent_name}' not in {allowed_escalation_agent_names}"
            
            logger.info(f"Escalating to {agent_name}: {query}, {context}")
            
            self._dialog_branches[f"{_agent_name}::{agent_name}"] = FleetDialogBranch(
                meta={
                    "asking_agent": _agent_name
                },
                operating_agent=self.agents.get(agent_name)
            )

            ask_response =  self._dialog_branches[f"{_agent_name}::{agent_name}"].invoke_agent(f"Question: {query}\nContext: {context}")
            print("\n\n")
            print(ask_response)

            return f"RESPONSE FROM {agent_name}:\n{ask_response[-1]['content']}"

    def invoke_agent(self, agent_name: str, query: str, stop_on: List[str] = []):
        return self.agents.get(agent_name).invoke(query, stop_on = stop_on, force_tools=True)