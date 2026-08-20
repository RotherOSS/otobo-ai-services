from typing import Optional, List
from pydantic import BaseModel


# Overrides how many entries to retrieve for a given collection type.
class TypeN(BaseModel):
    type: str
    n: int


# Input format for the RAG request
class RAGInput(BaseModel):
    question: str
    do_scoring: Optional[bool] = False
    label: str
    types_n: Optional[List[TypeN]] = None  # Per-type overrides for how many context pieces to retrieve.


# Output format for the RAG response
class RAGOutput(BaseModel):
    question: str
    generation: Optional[str] = None
    score: Optional[str] = None
    source_ids: Optional[List[str]] = None


# Input format for the source_ids-only lookup
class GetSourceIdsInput(BaseModel):
    question: str
    label: str
    types_n: Optional[List[TypeN]] = None


# One entry of the source_ids-only response, per collection type
class SourceIdsByType(BaseModel):
    type: str
    source_ids: List[str]
