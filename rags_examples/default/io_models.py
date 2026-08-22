from typing import Optional, List
from pydantic import BaseModel


# Overrides how many entries to retrieve for a given collection type.
class TypeN(BaseModel):
    type: str
    n: int


# One retrieved entry's source_id together with its similarity score.
class SourceIdEntry(BaseModel):
    source_id: str
    score: float


# One collection type's retrieved entries, each with its similarity score.
class SourceIdsByType(BaseModel):
    type: str
    source_ids: List[SourceIdEntry]


# Input format for the RAG request
class RAGInput(BaseModel):
    question: str
    do_scoring: Optional[bool] = False
    label: str
    types_n: Optional[List[TypeN]] = None  # Per-type overrides for how many context pieces to retrieve.
    # Entries scoring below this are dropped: neither used as context nor returned as source_ids.
    similarity_threshold: Optional[float] = None


# Output format for the RAG response
class RAGOutput(BaseModel):
    question: str
    generation: Optional[str] = None
    score: Optional[str] = None
    source_ids: Optional[List[SourceIdsByType]] = None


# Input format for the source_ids-only lookup
class GetSourceIdsInput(BaseModel):
    question: str
    label: str
    types_n: Optional[List[TypeN]] = None
    # Entries scoring below this are dropped: neither used as context nor returned as source_ids.
    similarity_threshold: Optional[float] = None
