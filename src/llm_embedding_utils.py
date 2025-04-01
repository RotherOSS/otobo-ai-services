from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import Document
from loguru import logger

from src.settings import AppSettings
from src.db import get_pg_connection
from src.data_models.ticket import Ticket, UploadTicket, IngestInput, IngestInputBatch

settings = AppSettings()


@logger.catch(reraise=True)
def get_embeddingsmodel():
    return OllamaEmbeddings(
        base_url=settings.OTOBO_AI_LLM_HOST, model=settings.OTOBO_AI_EMBEDDING_MODEL
    )


@logger.catch(reraise=True)
def get_vectorstore(with_embedding: bool = True, collection_name: str = settings.OTOBO_AI_CHROMA_COLLECTION):
    db_embedding = get_embeddingsmodel() if with_embedding else None

    return Chroma(
        collection_name=collection_name,
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
async def put_embeddings(insert_input: IngestInput):
    try:
        collection_name = insert_input.type or settings.OTOBO_AI_CHROMA_COLLECTION
        fulltext_id = None

        if insert_input.store_fulltext:
            fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in insert_input.content])
            conn = get_pg_connection()
            row = await conn.fetchrow(
                "INSERT INTO fulltext (text) VALUES ($1) RETURNING id",
                fulltext
            )
            fulltext_id = row["id"]

        if insert_input.embed_content_type:
            selected = [item.text for item in insert_input.content if item.type in insert_input.embed_content_type]
        else:
            selected = [item.text for item in insert_input.content]

        if selected:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.embedding_chunk_size,
                chunk_overlap=settings.embedding_chunk_overlap,
            )
            all_splits = text_splitter.create_documents(selected)

            if fulltext_id is not None:
                for doc in all_splits:
                    doc.metadata["fulltext_id"] = fulltext_id

            embed_store = get_vectorstore(with_embedding=True, collection_name=collection_name)
            await embed_store.aadd_documents(all_splits)

    except Exception as e:
        logger.error(f"Error inserting embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def put_embeddings_batch(batch_input: IngestInputBatch):
    collection_name = batch_input.type or settings.OTOBO_AI_CHROMA_COLLECTION
    fulltext_ids = []

    if batch_input.store_fulltext:
        fulltext_texts = []
        for content_list in batch_input.content:
            fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in content_list])
            fulltext_texts.append(fulltext)

        conn = get_pg_connection()
        rows = await conn.fetch(
            "INSERT INTO fulltext (text) SELECT x FROM unnest($1::text[]) x RETURNING id",
            fulltext_texts
        )
        fulltext_ids = [row["id"] for row in rows]

    embed_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.embedding_chunk_size,
        chunk_overlap=settings.embedding_chunk_overlap,
    )

    for idx, content_list in enumerate(batch_input.content):
        if batch_input.embed_content_type:
            selected = [item.text for item in content_list if item.type in batch_input.embed_content_type]
        else:
            selected = [item.text for item in content_list]

        if not selected:
            continue

        splits = text_splitter.create_documents(selected)

        if batch_input.store_fulltext and idx < len(fulltext_ids):
            for doc in splits:
                doc.metadata["fulltext_id"] = fulltext_ids[idx]

        embed_docs.extend(splits)

    embed_store = get_vectorstore(with_embedding=True, collection_name=collection_name)
    await embed_store.aadd_documents(embed_docs)


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
