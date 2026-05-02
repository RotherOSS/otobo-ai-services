from typing import Optional
from pydantic import BaseModel


# Input format for the RAG request
class RAGInput(BaseModel):
    question: str
    do_scoring: Optional[bool] = False


# Output format for the RAG response
class RAGOutput(BaseModel):
    question: str
    generation: Optional[str] = None
    score: Optional[str] = None

