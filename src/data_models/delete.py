from typing import Union, Optional, List
from pydantic import BaseModel


# Model for deleting ingested data.
class DeleteInput(BaseModel):
    type: Optional[str] = None  # Name of the collection to delete within.
    source_ids: List[str]  # The list of source ids of the entries to delete.
