from typing import Union, Optional, List
from pydantic import BaseModel


class ContentItem(BaseModel):
    type: str
    text: str


class IngestInput(BaseModel):
    type: Optional[str] = None  # name of collection and name of data source
    store_fulltext: bool = False
    fulltext_types: Optional[List[str]] = None
    embed_content_types: Optional[List[str]] = None
    content: List[ContentItem]


class IngestInputBatch(BaseModel):
    type: Optional[str] = None  # name of collection and name of data source
    store_fulltext: bool = False
    fulltext_types: Optional[List[str]] = None
    embed_content_types: Optional[List[str]] = None
    content: List[List[ContentItem]]