from typing import Union, Optional, List
from pydantic import BaseModel


# A single source ID targeted for deletion, with optional labels to remove.
class DeleteEntry(BaseModel):
    source_id: str
    labels: Optional[List[str]] = None  # Labels to remove; if omitted, the whole entry is deleted.


# Model for deleting ingested data.
class DeleteInput(BaseModel):
    type: Optional[str] = None  # Name of the collection to delete within.
    has_labels: bool = False  # Whether entries in source_ids carry labels to selectively remove.
    source_ids: List[DeleteEntry]  # The list of source ids (with optional labels) of the entries to delete.
