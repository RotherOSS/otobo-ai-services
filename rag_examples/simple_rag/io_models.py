from typing import Optional
from pydantic import BaseModel


class RAGInput(BaseModel):
    question: str


class RAGOutput(BaseModel):
    question: str
    generation: Optional[str] = None
