from pydantic import BaseModel, Field

from ..config import Config
from .message import Message

from typing import Dict, List
import abc

import mcp

class Backend(BaseModel, abc.ABC):
    config: Dict[str,str] = Field(default_factory=dict)

    @abc.abstractmethod
    def generate_response(self, messages: List[Message], agent_config: Config, tools: List[mcp.Tool] = [], system_prompt_override: str = "") -> List[Message]:
        return []
