from typing import Union, Optional, List
from pydantic import BaseModel


# Represents a single piece of input content, e.g., a document or paragraph.
class ContentItem(BaseModel):
    type: str  # e.g., "text", "markdown", etc. Can be used to route content processing.
    text: str  # The actual content to ingest.

class ContentSet(BaseModel):
    source_id: str
    content_items = List[ContentItem]

# Model for a single ingestion payload.
class IngestInput(BaseModel):
    type: Optional[str] = None  # Optional name for the collection / data source.
    store_fulltext: bool = False  # If true, stores the raw input text in the database.
    fulltext_types: Optional[List[str]] = None  # Specifies which content types to store in full.
    embed_content_types: Optional[List[str]] = None  # Specifies which content types to embed for retrieval.
    source_id: str = None
    content: List[ContentItem]  # A list of content items to ingest.


# Batch version of IngestInput. Allows uploading multiple content lists in one request.
class IngestInputBatch(BaseModel):
    type: Optional[str] = None
    store_fulltext: bool = False
    fulltext_types: Optional[List[str]] = None
    embed_content_types: Optional[List[str]] = None
    content: List[ContentSet]  # List of lists of content items.
