from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger

# Local imports from the project
from src.settings import AppSettings
from src.db import get_pg_pool
from src.data_models.ingest import IngestInput, IngestInputBatch
from src.data_models.retrieve import QueryInput

settings = AppSettings()


@logger.catch(reraise=True)
def get_embeddingsmodel():
    # Returns an embedding model instance using Ollama with config values
    return OllamaEmbeddings(
        base_url=settings.OTOBO_AI_LLM_HOST,
        model=settings.OTOBO_AI_EMBEDDING_MODEL
    )


@logger.catch(reraise=True)
def get_vectorstore(with_embedding: bool = True, collection_name: str = settings.OTOBO_AI_CHROMA_DEF_COL_NAME):
    # Returns a Chroma vector store instance, optionally attaching an embedding function
    db_embedding = get_embeddingsmodel() if with_embedding else None

    return Chroma(
        collection_name=collection_name,
        embedding_function=db_embedding,
        persist_directory=settings.OTOBO_AI_CHROMA_DIR  # Local dir for vector DB persistence
    )

@logger.catch(reraise=True)
async def purge_collection(with_embedding: bool = True, collection_name: str = settings.OTOBO_AI_CHROMA_DEF_COL_NAME):
    # Returns a Chroma vector store instance, optionally attaching an embedding function
    db_embedding = get_embeddingsmodel() if with_embedding else None

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=db_embedding,
        persist_directory=settings.OTOBO_AI_CHROMA_DIR  # Local dir for vector DB persistence
    )

    logger.error(f"Purge: {collection_name}")
    
    vector_store._client.delete_collection(collection_name)

    return { "success": True  }

@logger.catch(reraise=True)
async def purge_vectorstore(with_embedding: bool = True):
    # Returns a Chroma vector store instance, optionally attaching an embedding function
    db_embedding = get_embeddingsmodel() if with_embedding else None

    collections = [ "faq", "ticket_pairs", "ticket_chunks", "doc"  ]
    for collection in collections:

        await purge_collection( with_embedding=with_embedding, collection_name=collection )

    # postgres
    
    pool = get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM fulltext")

    return { "success": True  }

    

@logger.catch(reraise=True)
def get_model(use_ollama_json_format: bool = False):
    # Instantiates a chat model via Ollama, optionally using JSON output format
    return ChatOllama(
        base_url=settings.OTOBO_AI_LLM_HOST,
        model=settings.OTOBO_AI_LLM_MODEL,
        temperature=settings.OTOBO_AI_LLM_TEMPERATURE,
        headers={"otobo-api-key": settings.OTOBO_AI_LLM_API_KEY},
        format="json" if use_ollama_json_format else None,
    )


@logger.catch(reraise=True)
async def get_heartbeat():
    # Health check endpoint: verifies if the vector store backend is responsive
    try:
        client = get_vectorstore(with_embedding=False)
        client.get()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error getting heartbeat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def query_embeddings(retrieve: QueryInput):
    # Main query endpoint: retrieves most similar documents from the vector store
    try:
        collection_name = retrieve.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME
        vector_store = get_vectorstore(with_embedding=True, collection_name=collection_name)
        results = await vector_store.asimilarity_search(query=retrieve.query_text, k=retrieve.n_results)

        # Optionally enrich results with full text from the SQL database
        if retrieve.retrieve_fulltext:
            fulltext_ids = {doc.metadata.get("fulltext_id") for doc in results if doc.metadata.get("fulltext_id")}
            if fulltext_ids:
                pool = get_pg_pool()
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        f"SELECT id, text FROM fulltext WHERE id = ANY($1::int[]);",
                        list(fulltext_ids)
                    )
                id_to_text = {row["id"]: row["text"] for row in rows}

                for doc in results:
                    ft_id = doc.metadata.get("fulltext_id")
                    if ft_id in id_to_text:
                        doc.metadata["fulltext"] = id_to_text[ft_id]

        return results

    except Exception as e:
        logger.error(f"Error asynch. querying embeddings: {e}")
        return {"success": False, "error": str(e)}


@logger.catch(reraise=True)
async def put_embeddings(insert_input: IngestInput):
    # Ingests a single item into the vector store, optionally storing raw text in SQL
    try:
        collection_name = insert_input.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME
        fulltext_id = None

        # Optional: store fulltext in relational DB for later retrieval
        if insert_input.store_fulltext:
            if insert_input.fulltext_types:
                fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in insert_input.content if
                                        item.type in insert_input.fulltext_types])
            else:
                fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in insert_input.content])

            pool = get_pg_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "INSERT INTO fulltext (text) VALUES ($1) RETURNING id",
                    fulltext
                )
            fulltext_id = row["id"]

        # Select content types to embed (configurable)
        if insert_input.embed_content_types:
            selected = [item.text for item in insert_input.content if item.type in insert_input.embed_content_types]
        else:
            selected = [item.text for item in insert_input.content]

        # Split into chunks and embed
        if selected:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.embedding_chunk_size,
                chunk_overlap=settings.embedding_chunk_overlap,
            )
            all_splits = text_splitter.create_documents(selected)

            # Add fulltext reference to metadata if available
            if fulltext_id is not None:
                for doc in all_splits:
                    doc.metadata["fulltext_id"] = fulltext_id

            embed_store = get_vectorstore(with_embedding=True, collection_name=collection_name)
            await embed_store.aadd_documents(all_splits)
        return {"success": True}

    except Exception as e:
        logger.error(f"Error inserting embeddings: {e}")
        return {"success": False, "error": str(e)}


@logger.catch(reraise=True)
async def put_embeddings_batch(batch_input: IngestInputBatch):
    # Ingests multiple items in a batch; supports storing fulltext and embedding selected fields
    try:
        collection_name = batch_input.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME
        fulltext_ids = []

        # Optional fulltext storage
        if batch_input.store_fulltext:
            fulltext_texts = []
            if batch_input.fulltext_types:
                for content_list in batch_input.content:
                    fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in content_list if
                                            item.type in batch_input.fulltext_types])
                    fulltext_texts.append(fulltext)
            else:
                for content_list in batch_input.content:
                    fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in content_list])
                    fulltext_texts.append(fulltext)

            pool = get_pg_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "INSERT INTO fulltext (text) SELECT x FROM unnest($1::text[]) x RETURNING id",
                    fulltext_texts
                )
            fulltext_ids = [row["id"] for row in rows]

        # Prepare documents for embedding
        embed_docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.embedding_chunk_size,
            chunk_overlap=settings.embedding_chunk_overlap,
        )

        for idx, content_list in enumerate(batch_input.content):
            if batch_input.embed_content_types:
                selected = [item.text for item in content_list if item.type in batch_input.embed_content_types]
            else:
                selected = [item.text for item in content_list]

            if not selected:
                continue

            splits = text_splitter.create_documents(selected)

            # Attach correct fulltext ID to chunks
            if batch_input.store_fulltext and idx < len(fulltext_ids):
                for doc in splits:
                    doc.metadata["fulltext_id"] = fulltext_ids[idx]

            embed_docs.extend(splits)

        embed_store = get_vectorstore(with_embedding=True, collection_name=collection_name)
        await embed_store.aadd_documents(embed_docs)
        return {"success": True}

    except Exception as e:
        logger.error(f"Error inserting embeddings: {e}")
        return {"success": False, "error": str(e)}
