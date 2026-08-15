from typing import Optional, List
from pydantic import BaseModel


# Input format for the RAG request
class RAGInput(BaseModel):
    question: str
    do_scoring: Optional[bool] = False
    label: str


# Output format for the RAG response
class RAGOutput(BaseModel):
    question: str
    generation: Optional[str] = None
    score: Optional[str] = None
    source_ids: Optional[List[str]] = None

