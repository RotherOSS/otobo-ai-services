from typing import Optional
from pydantic import BaseModel


# Input model for API: just a question
class RAGInput(BaseModel):
    question: str
    lang: str | None


# Output model for API: includes original question and generated result
class RAGOutput(BaseModel):
    question: str
    generation: Optional[str] = None
