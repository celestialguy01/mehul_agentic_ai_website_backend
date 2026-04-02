from typing import TypedDict, Literal
from pydantic import BaseModel

class AgentState(TypedDict):
    user_input: str 
    intent: str 
    response: str

class IntentOutput(BaseModel):
    intent: Literal["faq", "explain", "contact", "irrelevant"]