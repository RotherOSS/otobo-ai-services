import uuid
from typing import Dict, List, Optional, Union
from fastapi import HTTPException
import chromadb
from chromadb import GetResult, QueryResult
from chromadb.api.types import ID, IDs, Include, OneOrMany, Where
from chromadb.config import Settings as ChromaDbSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import OllamaEmbeddings

from mylibs.classes.SefHostedEmbeddingFunction import (
    HuggingFaceEmbeddingFunction,
    OllamaEmbeddingFunction,
)
from mylibs.classes.AppSettings import AppSettings
from mylibs.classes.Ticket import Ticket


settings = AppSettings()


def embedding():
    if settings.use_huggingface:
        from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

        # Self hosted embedding model (+2GB ram)
        return HuggingFaceBgeEmbeddings(model_name=settings.embedding_model_name)

        # embedding = HuggingFaceInferenceAPIEmbeddings(
        #     api_key=settings.HUGGINGFACEHUB_API_TOKEN,
        #     model_name=settings.embedding_model_name,
        # )
    else:
        return OllamaEmbeddings(
            base_url=settings.LLM_OLLAMA_URL, model=settings.LLM_OLLAMA_MODEL
        )


def embedding_function():
    """returns the embedding function depending on settings flag"""
    if settings.use_huggingface:
        return HuggingFaceEmbeddingFunction()
    else:
        return OllamaEmbeddingFunction()


def get_dbclient():
    """Helper function to always use the same Chroma client"""
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


def get_heartbeat():
    """Get the current time in nanoseconds since epoch. Used to check if the server is alive.

    Raises:
        HTTPException: 500

    Returns:
        int: The current time in nanoseconds since epoch
    """
    try:
        return get_dbclient().heartbeat()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


def get_embedding(id: str) -> GetResult:
    """returns the embedding with the given id.

    Args:
        id (str): id of embedding

    Raises:
        HTTPException: 500

    Returns:
        GetResult: A GetResult object containing the results.
    """ """"""
    try:
        client = get_dbclient()
        collection = client.get_collection(name=settings.AI_VECTORSTORE_INDEX)
        embedding = collection.get(ids=id, include=["documents", "metadatas"])
        return embedding
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


def get_embeddings(
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
        client = get_dbclient()
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
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


def query_embeddings(
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
        client = get_dbclient()
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
        client = get_dbclient()
        collection = client.get_or_create_collection(
            name=settings.AI_VECTORSTORE_INDEX,
            embedding_function=embedding_function(),
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.embedding_chunk_size,
            chunk_overlap=settings.embedding_chunk_overlap,
        )
    except Exception as e:
        print("Error while getting client: ", e)
        raise HTTPException(status_code=500, detail=str(e))

    for ticket in tickets:
        try:
            all_splits = text_splitter.create_documents([ticket.document])

            docs = [item.page_content for item in all_splits]
            meta = get_meta(ticket)

            metas: List[chromadb.Metadata] = [
                {**meta, "chunk_id": i, "chunks": len(docs)} for i in range(len(docs))
            ]
            ids: chromadb.IDs = [str(uuid.uuid4()) for i in range(len(docs))]

            collection.add(ids=ids, documents=docs, metadatas=metas)
            all_ids = all_ids + ids
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))
    return all_ids


def delete_embedding(id: str):
    """delete a embedding by id

    Args:
        id (str): id of dataset to be deleted

    Raises:
        HTTPException: 500

    Returns:
        str: deleted id
    """
    try:
        client = get_dbclient()
        collection = client.get_collection(name=settings.AI_VECTORSTORE_INDEX)
        collection.delete(ids=[id])
        return {"id": id}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


def delete_embeddings(
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
        List[str] | None: given list of ids
    """ """"""
    try:
        client = get_dbclient()
        collection = client.get_collection(name=settings.AI_VECTORSTORE_INDEX)
        collection.delete(
            ids=ids,
            where=where,
        )
        return {"ids": ids}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
