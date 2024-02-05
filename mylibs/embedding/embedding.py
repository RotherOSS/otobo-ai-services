import uuid
from typing import Dict, List, Optional
from fastapi import HTTPException
import chromadb
from chromadb import GetResult, QueryResult
from chromadb.api.types import ID, IDs, Include, OneOrMany, Where
from chromadb.config import Settings as ChromaDbSettings
from elasticsearch import Elasticsearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

from mylibs.classes.SefHostedEmbeddingFunction import (
    HuggingFaceEmbeddingFunction,
    OllamaEmbeddingFunction,
)
from mylibs.classes.AppSettings import AppSettings
from mylibs.classes.Ticket import Ticket

settings = AppSettings()


def embedding():
    if settings.use_localembedding:
        # Self hosted embedding model (+2GB ram)
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        return HuggingFaceBgeEmbeddings(model_name=settings.embedding_model_name)
    else:
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=settings.LLM_OLLAMA_URL, model=settings.LLM_OLLAMA_MODEL
        )


def embedding_function():
    """returns the embedding function depending on settings flag"""
    if settings.use_localembedding:
        return HuggingFaceEmbeddingFunction()
    else:
        return OllamaEmbeddingFunction()


def get_chroma_dbclient():
    """Helper function to get db client"""
    try:
        if settings.AI_VECTORDB_AUTH_TOKEN is None:
            return chromadb.HttpClient(
                host=settings.AI_VECTORDB_HOST,
                port=settings.AI_VECTORDB_PORT,
            )
        else:
            return chromadb.HttpClient(
                host=settings.AI_VECTORDB_HOST,
                port=settings.AI_VECTORDB_PORT,
                settings=ChromaDbSettings(
                    chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                    chroma_client_auth_credentials=settings.AI_VECTORDB_AUTH_TOKEN,
                ),
            )
    except Exception as e:
        print("error in get_dbclient:")
        print(e)
        raise e


def get_vectorstore():
    db_embedding = embedding()
    if settings.use_chromadb:
        client = get_chroma_dbclient()
        vectorstore = Chroma(
            collection_name=settings.AI_VECTORSTORE_INDEX, embedding_function=db_embedding, client=client  # type: ignore
        )
        return vectorstore
    else:
        from langchain_community.vectorstores.elasticsearch import ElasticsearchStore

        # print(f"{settings.AI_VECTORDB_HOST}:{settings.AI_VECTORDB_PORT}")
        return ElasticsearchStore(
            es_url=settings.es_url,
            index_name=settings.AI_VECTORSTORE_INDEX,
            embedding=db_embedding,
        )


def get_model():
    if settings.use_together:
        from langchain_community.llms.together import Together

        return Together(
            model=settings.TOGETHERAI_MODEL,  # type: ignore
            together_api_key=settings.TOGETHERAI_API_KEY,  # type: ignore
            max_tokens=2048,
            temperature=settings.LLM_TEMPERATURE,
        )
    else:
        from langchain_community.llms.ollama import Ollama

        return Ollama(
            base_url=settings.LLM_OLLAMA_URL,
            model=settings.LLM_OLLAMA_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            headers={"otobo-api-key": settings.LLM_OTOBO_API_KEY},
        )


def get_meta(ticket: Ticket):
    """returns the meta data for Chroma data set from Ticket structure.

    Args:
        ticket (Ticket): Ticket structure

    Returns:
        _type_: meta data to save in db
    """ """"""
    meta = {
        "process_id": ticket.process_id,
        "gdpr_id": ticket.gdpr_id,
        "topic": ticket.topic,
        "type": ticket.type,
        "len": ticket.len,
        # "document": ticket.document,
    }
    return meta


async def get_heartbeat():
    """Get the current time in nanoseconds since epoch. Used to check if the server is alive.

    Raises:
        HTTPException: 500

    Returns:
        int: The current time in nanoseconds since epoch
    """
    try:
        if settings.use_chromadb:
            return get_chroma_dbclient().heartbeat()
        else:

            client = Elasticsearch(settings.es_url)
            return client.info()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


async def get_embedding(id: str) -> GetResult:
    """returns the embedding with the given id.

    Args:
        id (str): id of embedding

    Raises:
        HTTPException: 500

    Returns:
        GetResult: A GetResult object containing the results.
    """ """"""
    try:
        if settings.use_chromadb:
            client = get_chroma_dbclient()
            collection = client.get_collection(name=settings.AI_VECTORSTORE_INDEX)
            embedding = collection.get(ids=id, include=["documents", "metadatas"])
            return embedding
        else:
            es = Elasticsearch(settings.es_url)
            embedding = es.get(index=settings.AI_VECTORSTORE_INDEX, id=id)
            return embedding
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


async def get_embeddings(
    ids: Optional[OneOrMany[ID]] = None,
    process_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    include: Include = ["metadatas", "documents"],
) -> GetResult:
    """returns the embeddings with the given search parameter.

    At least one of the optional parameters 'ids' and 'process_id' must be specified.


    Args:
        ids (Optional[OneOrMany[ID]], optional): list of ids. Defaults to None.
        process_id (Optional[str], optional): list of process ids. Defaults to None.
        limit (Optional[int], optional): max no of results. Defaults to None.
        offset (Optional[int], optional): offset. Defaults to None.
        include (Include, optional): part of result. Defaults to ["metadatas", "documents"].

    Raises:
        HTTPException: 500

    Returns:
        GetResult: A GetResult object containing the results.
    """ """"""
    try:
        if settings.use_chromadb:
            client = get_chroma_dbclient()
            collection = client.get_collection(name=settings.AI_VECTORSTORE_INDEX)
            where: Where | None
            if process_id is not None:
                where = {"process_id": process_id}
            else:
                where = None
            embed = collection.get(
                ids=ids,
                where=where,
                limit=limit,
                offset=offset,
                include=include,
            )
            return embed
        else:
            es = Elasticsearch(settings.es_url)
            query = {"bool": {"filter": []}}
            if ids:
                query["bool"]["filter"].append({"terms": {"_id": ids}})
            if process_id:
                query["bool"]["filter"].append(
                    {"match_phrase": {"metadata.process_id": process_id}}
                )
            # todo include ggf erweitern
            if "embeddings" not in include:
                source_excludes = "vector"
            else:
                source_excludes = None
            embed = es.search(
                index=settings.AI_VECTORSTORE_INDEX,
                query=query,
                from_=offset,
                size=limit,
                source_excludes=source_excludes,
            )
            return embed.body["hits"]["hits"]
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


async def query_embeddings(
    query_texts: List[str],
    where: Optional[Dict] = None,
    n_results: int = 10,
    include: Include = ["metadatas", "documents"],
) -> QueryResult:
    """returns embeddings queried by the given query text(s).

    This is a semantic search only. No LLM involved!
    Returns a QueryResult

    Args:
        query_texts (List[str]): query text(s)
        where (Optional[Dict], optional): where condition. Defaults to None.
        n_results (int, optional): max no of results. Defaults to 10.
        include (Include, optional): include. Defaults to ["metadatas", "documents"].

    Raises:
        HTTPException: _description_

    Returns:
        QueryResult: _description_
    """
    try:
        if settings.use_chromadb:
            client = get_chroma_dbclient()
            collection = client.get_collection(
                name=settings.AI_VECTORSTORE_INDEX,
                embedding_function=embedding_function(),
            )
            result = collection.query(
                query_embeddings=None,
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                include=include,
            )
            return result
        else:
            es: ElasticsearchStore = get_vectorstore()  # type: ignore
            # include parameter not supported with Elasticsearch
            result = await es.asimilarity_search(query=query_texts[0] if query_texts else "", filter=[where], k=n_results)  # type: ignore
            return result

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


async def put_embeddings(tickets: List[Ticket]):
    """Puts the given list of data sets in Ticket format into the Vector DB

    Splits the texts into chunks with chunk size Settings.embedding_chunk_size and overlap Settings.embedding_chunk_overlap
    Creates the embedding vector.
    Notice: maybe max 50 Tickets can be a good size, may take a few minutes

    Args:
        tickets (List[Ticket]): list of ticktets to be embedded

    Raises:
        HTTPException: 500
    Returns:
        List[str]: created ids
    """ """"""
    # Idea: add datetime

    all_ids = []
    try:
        if settings.use_chromadb:
            client = get_chroma_dbclient()
            chroma_db = client.get_or_create_collection(
                name=settings.AI_VECTORSTORE_INDEX,
                embedding_function=embedding_function(),  # type: ignore
            )
            for ticket in tickets:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.embedding_chunk_size,
                    chunk_overlap=settings.embedding_chunk_overlap,
                )
                all_splits = text_splitter.create_documents([ticket.document])

                docs = [item.page_content for item in all_splits]
                meta = get_meta(ticket)

                metas: List[chromadb.Metadata] = [
                    {**meta, "chunk_id": i, "chunks": len(docs)}
                    for i in range(len(docs))
                ]

                ids: chromadb.IDs = [str(uuid.uuid4()) for i in range(len(docs))]
                chroma_db.add(ids=ids, documents=docs, metadatas=metas)  # type: ignore
                all_ids = all_ids + ids
        else:
            elasticdb: ElasticsearchStore = get_vectorstore()  # type: ignore
            for ticket in tickets:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.embedding_chunk_size,
                    chunk_overlap=settings.embedding_chunk_overlap,
                )
                all_splits = text_splitter.create_documents([ticket.document])

                for i, doc in enumerate(all_splits):
                    doc.metadata = get_meta(ticket)
                    doc.metadata["chunk_id"] = i
                    doc.metadata["chunks"] = len(all_splits)

                ids = await elasticdb.aadd_documents(all_splits)

                all_ids = all_ids + ids
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

    return all_ids


async def delete_embedding(id: str):
    """delete a embedding by id

    Args:
        id (str): id of dataset to be deleted

    Raises:
        HTTPException: 500

    Returns:
        str: deleted id
    """
    try:
        if settings.use_chromadb:
            client = get_chroma_dbclient()
            collection = client.get_collection(name=settings.AI_VECTORSTORE_INDEX)
            collection.delete(ids=[id])
            return {"id": id}
        else:
            es = Elasticsearch(settings.es_url)
            response = es.delete(index=settings.AI_VECTORSTORE_INDEX, id=id)
            return response.body["_id"]
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


async def delete_embeddings(
    ids: Optional[IDs] = None,
    where: Optional[Dict] = None,
):
    """deletes the embeddings with the given parameters.

    You have to specify at least one parameter.


    Args:
        ids (Optional[IDs], optional): list of ids. Defaults to None.
        where (Optional[Dict], optional): where condition. Defaults to None.

    Raises:
        HTTPException: 500

    Returns:
        Dict: ids when Chroma, deleted when Elasticsearch
    """ """"""
    try:
        if settings.use_chromadb:
            client = get_chroma_dbclient()
            collection = client.get_collection(name=settings.AI_VECTORSTORE_INDEX)
            collection.delete(
                ids=ids,
                where=where,
            )
            return {"ids": ids}
        else:
            es = Elasticsearch(settings.es_url)

            body = {"query": {"bool": {"filter": []}}}
            filter = body["query"]["bool"]["filter"]
            if ids:
                filter.append({"terms": {"_id": ids}})
            if where:
                filter.append(where)

            response = es.delete_by_query(
                index=settings.AI_VECTORSTORE_INDEX, body=body
            )
            return {"deleted": response["deleted"]}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
