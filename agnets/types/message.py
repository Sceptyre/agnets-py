from pydantic import BaseModel

from typing import Literal, Any

class Message(BaseModel):
    message_type: Literal['message', 'thinking', 'tool_call', 'tool_result']
    role: Literal['system', 'assistant', 'user']
    message_content: Any