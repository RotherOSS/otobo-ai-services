from typing import Union, Optional, List
from pydantic import BaseModel


class QueryInput(BaseModel):
    type: Optional[str] = None  # name of collection and name of data source
    query_text: str
    retrieve_fulltext: bool = False
    n_results: int = 10
