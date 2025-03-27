from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
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
    meta = {
        "process_id": ticket.process_id,
        "gdpr_id": ticket.gdpr_id,
        "topic": ticket.topic,
        "type": ticket.type,
        "len": ticket.len,
    }
    return meta


@logger.catch(reraise=True)
async def get_heartbeat():
    try:
        client = get_vectorstore(with_embedding=False)
        client.get()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error getting heartbeat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
def query_embeddings(
        query_texts: Optional[List[str]] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        n_results: int = 10,
):
    try:
        vector_store = get_vectorstore()
        where_clause = where_filter if where_filter else None
        result = vector_store.similarity_search(query=query_texts[0] if query_texts else "", filter=where_clause,
                                                k=n_results)
        return result

    except Exception as e:
        logger.error(f"Error querying embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def aquery_embeddings(
        query_texts: Optional[List[str]] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        n_results: int = 10,
):
    try:
        vector_store = get_vectorstore()
        where_clause = where_filter if where_filter else None
        result = await vector_store.asimilarity_search(query=query_texts[0] if query_texts else "", filter=where_clause,
                                                       k=n_results)
        return result

    except Exception as e:
        logger.error(f"Error asynch. querying embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def put_embeddings(tickets: List[UploadTicket]):
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
    try:
        vector_store = get_vectorstore(with_embedding=False)
        vector_store.delete(ids=[id])
        return id
    except Exception as e:
        logger.error(f"Error deleting embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def delete_embeddings(
    ids: Any | None = None,
    where: Optional[Dict] = None,
):
    if not ids and not where:
        raise HTTPException(status_code=400, detail="Either ids or where condition must be provided")

    try:
        vector_store = get_vectorstore(with_embedding=False)

        delete_kwargs = {}
        if ids:
            delete_kwargs["ids"] = ids
        if where:
            delete_kwargs["where"] = where

        vector_store.delete(**delete_kwargs)

        return {"deleted": delete_kwargs}
    except Exception as e:
        logger.error(f"Error deleting embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
