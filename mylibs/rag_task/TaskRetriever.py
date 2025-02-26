from typing import List

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

# from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from elasticsearch import Elasticsearch
from mylibs.classes.AppSettings import AppSettings
from mylibs.embedding.embedding import get_vectorstore, query_embeddings

settings = AppSettings()

k_search = 20


class TaskRetriever(BaseRetriever):
    # vectorstore: BaseRetriever
    # search_kwargs: dict = Field(default_factory=dict)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieves all chunks of the 1-3 most relevant task.
        Determines the most relevant task by embedding the query and comparing it to the embedding of all tasks.
        Then retrieves all text chunks of the most relevant task.

        Args:
            query (str): the query given by the user

        Returns:
            List[Document]: list of all text chunks (maybe including overlapping)
        """

        get_task_result = query_embeddings(
            query_texts=[query],
            # where={"type": "question"},
            n_results=3,
        )
        ids = [doc.metadata["process_id"] for doc in get_task_result]
        ids = list(dict.fromkeys(ids))  # remove duplicates

        es = Elasticsearch(hosts=[settings.es_url])
        query = {"terms": {"metadata.process_id.keyword": ids}}
        results = es.search(index=settings.AI_VECTORSTORE_INDEX, query=query)

        docs: List[Document] = []
        for result in results["hits"]["hits"]:  # type: ignore
            doc = Document(
                page_content=result["_source"]["text"], metadata={"ids": ids}
            )
            docs.append(doc)

        return docs
