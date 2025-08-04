from pydantic import BaseModel
import mcp.types

from typing import Literal, List, Dict, Any


class MessageComponent(BaseModel):
    type: Literal['message']
    content: str
    _meta: Dict[str, Any]

class MessageThinkingComponent(MessageComponent):
    type: Literal['thinking']
    content: str

class MessageToolCallComponent(MessageComponent):
    type: Literal['tool_call']
    content: mcp.types.CallToolRequest

class MessageToolResultComponent(MessageComponent):
    type: Literal['tool_result']
    content: mcp.types.CallToolResult



class Message(BaseModel):
    role: Literal['system', 'assistant', 'user']
    components: List[MessageComponent | MessageThinkingComponent | MessageToolCallComponent | MessageToolResultComponent]