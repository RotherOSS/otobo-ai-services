from typing import List
from chromadb import Where
from elasticsearch import Elasticsearch
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.pydantic_v1 import Field
from langchain_core.documents import Document
from langchain.document_transformers.embeddings_redundant_filter import (
    EmbeddingsRedundantFilter,
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever

from mylibs.classes.AppSettings import AppSettings
from mylibs.embedding.embedding import (
    get_chroma_dbclient,
    get_model,
    embedding,
    embedding_function,
    get_vectorstore,
    query_embeddings,
)
from mylibs.utils.utils import get_content, get_prompt

settings = AppSettings()

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)


class SupportRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieves all chunks of the 1-3 most relevant task.
        Determines the most relevant task by embedding the query and comparing it to the embedding of all tasks.
        Then retrieves all text chunks of the most relevant task.

        Args:
            query (str): the query given by the user

        Returns:
            List[Document]: list of all text chunks (maybe including overlapping)
        """
        if settings.use_chromadb:
            get_task_result = query_embeddings(
                query_texts=[query],
                # where={"type": "question"},
                n_results=3,
                include=["metadatas"],
            )
            ids = [el["process_id"] for el in get_task_result["metadatas"][0]]  # type: ignore
            ids = list(dict.fromkeys(ids))  # remove duplicates
            chroma_client = get_chroma_dbclient()
            collection = chroma_client.get_collection(
                name=settings.AI_VECTORSTORE_INDEX,
                embedding_function=embedding_function(),
            )

            where: Where = {"process_id": {"$in": ids}}
            get_all_result = collection.get(where=where, include=["documents"])

            docs: List[Document] = []
            for result in get_all_result["documents"]:  # type: ignore
                doc = Document(page_content=result, metadata={"ids": ids})
                docs.append(doc)

            return docs
        else:
            get_task_result = await query_embeddings(
                query_texts=[query],
                # where={"type": "question"},
                n_results=3,
            )
            ids = [doc.metadata["process_id"].lower() for doc in get_task_result]  # type: ignore
            ids = list(dict.fromkeys(ids))  # remove duplicates

            es = Elasticsearch(settings.es_url)
            es_query = {"bool": {"filter": [{"terms": {"metadata.process_id": ids}}]}}
            embed = es.search(
                index=settings.AI_VECTORSTORE_INDEX,
                query=es_query,
                source_excludes="vector",
            )
            docs: List[Document] = []
            for result in embed.body["hits"]["hits"]:  # type: ignore
                doc = Document(
                    page_content=result["_source"]["text"], metadata={"ids": ids}
                )
                docs.append(doc)

            return docs


embedding = embedding()
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs=settings.rag_search_kwargs)
support_retriever = SupportRetriever(vectorstore=retriever)


sys_prompt = """Sie sind ein hilfsbereiter, respektvoller und ehrlicher Assistent. Antworten Sie immer auf Deutsch, so hilfreich wie möglich und verwenden Sie den vorgegebenen CONTEXT.
  Ihre Antworten sollten die Frage nur einmal beantworten und nach der Antwort keinen weiteren Text enthalten.
  Wenn eine Frage keinen Sinn ergibt oder sachlich nicht kohärent ist, erklären Sie bitte, warum, anstatt etwas Falsches zu antworten.
  Wenn Sie die Antwort auf eine Frage nicht wissen, oder der CONTEXT leer ist, sagen Sie 'Das kann ich nicht beantworten'. """
instruction = """CONTEXT:/n/n {context}/n
  Question: {question}"""
template = get_prompt(instruction, sys_prompt)
prompt_template = ChatPromptTemplate.from_template(template)


splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
relevant_filter = EmbeddingsFilter(
    embeddings=embedding, similarity_threshold=settings.SIMILARITY_THRESHOLD
)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=support_retriever
)

# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=pipeline_compressor, base_retriever=retriever
# )


# LLM
model = get_model()

# RAG chain. Hint: wrtitten in LangChain Expression Language (LCEL)
chain = (
    RunnableParallel(
        {
            "context": compression_retriever | RunnableLambda(get_content),
            "question": RunnablePassthrough(),
        }
    )
    | prompt_template
    | model
    | StrOutputParser()
)


# # Add typing for input
# class Question(BaseModel):
#     __root__: str


# chain = chain.with_types(input_type=Question)
