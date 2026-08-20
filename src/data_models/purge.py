from typing import List
from pydantic import BaseModel


# Model for purging an entire collection, optionally scoped to specific labels.
class PurgeCollectionInput(BaseModel):
    type: str  # Name of the collection to purge.
    labels: List[str]  # Labels to purge; each label's whole physical collection is dropped.
