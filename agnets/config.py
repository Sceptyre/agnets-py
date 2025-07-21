from pydantic import BaseModel, Field


class Config(BaseModel):
    model_name: str
    system_prompt: str = Field(default="")

    
    do_thinking: bool = Field(default=False)
    do_unsupported_model_workaround: bool = Field(default=False)