from typing import Union
from pydantic import BaseModel


# the base structure of a ticket, document must have a value, rest can be None but must exist
class Ticket(BaseModel):
    """Default Ticket structure, only document must have a value, rest can be None but must exist

    Args:
        BaseModel (_type_): Pydantic BaseModel
    """

    id: Union[str, None] = None
    process_id: Union[str, None] = None
    gdpr_id: Union[str, None] = None
    topic: Union[str, None] = None
    type: Union[str, None] = None  # "question", "answer", "additional", ...
    len: Union[int, None] = None
    document: str


# the base structure of a ticket, document must have a value, rest can be None but must exist
class UploadTicket(BaseModel):
    """Default Ticket structure, only document must have a value, rest can be None but must exist

    Args:
        BaseModel (_type_): Pydantic BaseModel
    """

    process_id: Union[str, None] = None
    gdpr_id: Union[str, None] = None
    topic: Union[str, None] = None
    type: Union[str, None] = None  # "question", "answer", "additional", ...
    len: Union[int, None] = None
    document: str


class ContentItem(BaseModel):
    type: Literal["title", "user", "agent"]
    message: str


class IngestInput(BaseModel):
    type: Optional[str] = None  # name of collection and name of data source
    store_fulltext: bool = False
    embed_content_type: Optional[List[str]] = None
    content: List[ContentItem]


class IngestInputBatch(BaseModel):
    type: Optional[str] = None  # name of collection and name of data source
    store_fulltext: bool = False
    embed_content_type: Optional[List[str]] = None
    content: List[List[ContentItem]]
