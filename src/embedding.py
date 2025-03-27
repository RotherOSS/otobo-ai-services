from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger

from src.settings import AppSettings
from src.data_models.ticket import Ticket, UploadTicket

settings = AppSettings()


@logger.catch(reraise=True)
def get_embeddingsmodel():
    return OllamaEmbeddings(
        base_url=settings.OTOBO_AI_LLM_HOST, model=settings.OTOBO_AI_EMBEDDING_MODEL
    )


@logger.catch(reraise=True)
def get_vectorstore(with_embedding: bool = True):
    db_embedding = get_embeddingsmodel() if with_embedding else None

    return Chroma(
        collection_name=settings.OTOBO_AI_CHROMA_COLLECTION,
        embedding_function=db_embedding,
        persist_directory=settings.OTOBO_AI_CHROMA_DIR  # You can define this in your .env or settings
    )


@logger.catch(reraise=True)
def get_model(use_ollama_json_format: bool = False):
    if use_ollama_json_format:
        return ChatOllama(
            base_url=settings.OTOBO_AI_LLM_HOST,
            model=settings.OTOBO_AI_LLM_MODEL,
            temperature=settings.OTOBO_AI_LLM_TEMPERATURE,
            headers={"otobo-api-key": settings.OTOBO_AI_LLM_API_KEY},
            format="json",
        )
    else:
        return ChatOllama(
            base_url=settings.OTOBO_AI_LLM_HOST,
            model=settings.OTOBO_AI_LLM_MODEL,
            temperature=settings.OTOBO_AI_LLM_TEMPERATURE,
            headers={"otobo-api-key": settings.OTOBO_AI_LLM_API_KEY},
        )


@logger.catch(reraise=True)
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


@logger.catch(reraise=True)
async def get_heartbeat():
    """Get the vectorestore client info.

    Raises:
        HTTPException: 500

    Returns:
        some info
    """
    try:
        client = get_vectorstore(with_embedding=False)
        return client.client.info()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def get_embedding(id: str):
    """Returns the embedding with the given id.

    Args:
        id (str): id of embedding

    Raises:
        HTTPException: 500

    Returns:
        dict: The retrieved embedding metadata and vector.
    """
    try:
        client = PersistentClient(path=settings.OTOBO_AI_CHROMA_DIR)
        collection = client.get_collection(settings.OTOBO_AI_CHROMA_COLLECTION)
        embedding = collection.get(ids=[id])
        return embedding
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
def query_embeddings(
    query_texts: Optional[List[str]] = None,
    where_filter: Optional[List[Dict]] = None,
    n_results: int = 10,
    # include: Include = ["metadatas", "documents"],
):
    """returns embeddings queried by the given query text(s).

    This is a semantic search only. No LLM involved!
    Returns a QueryResult

    Args:
        query_texts (List[str]): query text(s)
        where_filter (Optional[List[Dict], optional): where condition. Defaults to None. I.e. [{"match_phrase": {"metadata.process_id": "Slice_0_100"}}]
        n_results (int, optional): max no of results. Defaults to 10.
        include (Include, optional): include. Defaults to ["metadatas", "documents"]. Only ChromaDB!

    Raises:
        HTTPException: _description_

    Returns:
        QueryResult: _description_
    """
    try:
        vector_store = get_vectorstore()
        where_list = where_filter if where_filter and where_filter[0] else None
        result = vector_store.similarity_search(query=query_texts[0] if query_texts else "", filter=where_list, k=n_results)  # type: ignore
        return result

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def aquery_embeddings(
    query_texts: Optional[List[str]] = None,
    where_filter: Optional[List[Dict]] = None,
    n_results: int = 10,
):
    """returns embeddings queried by the given query text(s).

    This is a semantic search only. No LLM involved!
    Returns a QueryResult

    Args:
        query_texts (List[str]): query text(s)
        where_filter (Optional[List[Dict], optional): where condition. Defaults to None. I.e. [{"match_phrase": {"metadata.process_id": "Slice_0_100"}}]
        n_results (int, optional): max no of results. Defaults to 10.
        include (Include, optional): include. Defaults to ["metadatas", "documents"]. Only ChromaDB!

    Raises:
        HTTPException: _description_

    Returns:
        QueryResult: _description_
    """
    try:
        vector_store = get_vectorstore()
        where_list = where_filter if where_filter and where_filter[0] else None
        result = await vector_store.asimilarity_search(query=query_texts[0] if query_texts else "", filter=where_list, k=n_results)  # type: ignore
        return result

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def put_embeddings(tickets: List[UploadTicket]):
    """Inserts a list of datasets in Ticket format into the Vector DB.

    Splits the texts into chunks with specified chunk size and overlap,
    then creates embedding vectors.

    Args:
        tickets (List[UploadTicket]): List of tickets to be embedded.

    Raises:
        HTTPException: 500 if an error occurs.

    Returns:
        List[str]: List of created IDs.
    """
    all_ids = []
    try:
        vector_store = get_vectorstore()
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

            ids = await vector_store.aadd_documents(all_splits)
            all_ids.extend(ids)
    except Exception as e:
        logger.error(f"Error inserting embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return all_ids


@logger.catch(reraise=True)
async def delete_embedding(id: str):
    """Delete an embedding by id.

    Args:
        id (str): id of embedding to be deleted

    Raises:
        HTTPException: 500

    Returns:
        str: deleted id
    """
    try:
        client = PersistentClient(path=settings.OTOBO_AI_CHROMA_DIR)
        collection = client.get_collection(settings.OTOBO_AI_CHROMA_COLLECTION)
        collection.delete(ids=[id])
        return id
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def delete_embeddings(
        ids: Any | None = None,
        where: Optional[Dict] = None,
):
    """Deletes the embeddings with the given parameters.

    You have to specify at least one parameter.

    Args:
        ids (Optional[Any], optional): list of ids. Defaults to None.
        where (Optional[Dict], optional): metadata filter condition. Defaults to None.

    Raises:
        HTTPException: 500

    Returns:
        Dict: ids of deleted embeddings.
    """
    if not ids and not where:
        raise HTTPException(status_code=400, detail="Either ids or where condition must be provided")

    try:
        client = PersistentClient(path=settings.OTOBO_AI_CHROMA_DIR)
        collection = client.get_collection(settings.OTOBO_AI_CHROMA_COLLECTION)

        if ids:
            collection.delete(ids=ids)
        if where:
            collection.delete(where=where)

        return {"deleted_ids": ids if ids else "Deleted by filter"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
