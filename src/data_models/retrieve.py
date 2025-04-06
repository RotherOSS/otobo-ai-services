from typing import Union, Optional, List
from pydantic import BaseModel


# Model for querying ingested data.
class QueryInput(BaseModel):
    type: Optional[str] = None  # Name of the collection to query.
    query_text: str  # The actual user query (used to generate embeddings for retrieval).
    retrieve_fulltext: bool = False  # If true, returns the stored raw content alongside results.
    n_results: int = 10  # Number of top documents to retrieve.
