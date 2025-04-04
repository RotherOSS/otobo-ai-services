from typing import Optional
from pydantic import BaseModel


class RAGInput(BaseModel):
    question: str
    do_scoring: Optional[bool] = False


class RAGOutput(BaseModel):
    question: str
    generation: Optional[str] = None
    score: Optional[float] = None
