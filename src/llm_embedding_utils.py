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
from src.data_models.delete import DeleteInput

settings = AppSettings()


@logger.catch(reraise=True)
def get_embeddingsmodel():
    # Returns an embedding model instance using Ollama with config values
    return OllamaEmbeddings(
        base_url=settings.OTOBO_AI_LLM_HOST,
        model=settings.OTOBO_AI_EMBEDDING_MODEL,
        client_kwargs= {
            "headers" : {"Authorization": "Bearer " + settings.OTOBO_AI_LLM_API_KEY, "Content-Type" : "apllication/json"}
        }
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

    logger.info(f"Purge: {collection_name}")
    
    vector_store._client.delete_collection(collection_name)

    pool = get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM source_vector_index_map WHERE collection_name = $1 ", collection_name);

    return { "success": True  }

@logger.catch(reraise=True)
async def purge_vectorstore(with_embedding: bool = True):
    # Returns a Chroma vector store instance, optionally attaching an embedding function
    db_embedding = get_embeddingsmodel() if with_embedding else None

    collections = [ "faqs", "ticket_pairs", "ticket_chunks", "docs"  ]
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
        client_kwargs= {
            "headers" : {"Authorization": "Bearer " + settings.OTOBO_AI_LLM_API_KEY, "Content-Type" : "apllication/json"}
        },
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
        
        logger.info( f"query_embeddings from {collection_name}" )
        vector_store = get_vectorstore(with_embedding=True, collection_name=collection_name)
        results = await vector_store.asimilarity_search(query=retrieve.query_text, k=retrieve.n_results)

        # Optionally enrich results with full text from the SQL database
        if retrieve.retrieve_fulltext:
            fulltext_ids = {doc.metadata.get("fulltext_source_id") for doc in results if doc.metadata.get("fulltext_source_id")}
            if fulltext_ids:
                pool = get_pg_pool()
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        f"SELECT source_id, text FROM fulltext WHERE collection_name = $1 "
                        f"AND source_id = ANY($2::text[]);",
                        collection_name,
                        list(fulltext_ids)
                    )
                id_to_text = {row["source_id"]: row["text"] for row in rows}

                for doc in results:
                    ft_id = doc.metadata.get("fulltext_source_id")
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

        pool = get_pg_pool()
        async with pool.acquire() as conn:
            if insert_input.store_fulltext:
                if insert_input.fulltext_types:
                    fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in insert_input.content if
                                            item.type in insert_input.fulltext_types])
                else:
                    fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in insert_input.content])

                await conn.fetchrow(
                    "INSERT INTO fulltext (collection_name, source_id, text) VALUES ($1, $2, $3)",
                    collection_name,
                    insert_input.source_id,
                    fulltext
                )
                fulltext_id = insert_input.source_id

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
                        doc.metadata["fulltext_source_id"] = fulltext_id

                embed_store = get_vectorstore(with_embedding=True, collection_name=collection_name)
                vec_ids = await embed_store.aadd_documents(all_splits)
                await conn.executemany(
                    "INSERT INTO source_vector_index_map (collection_name, source_id, vector_id) VALUES ($1, $2, $3)",
                    [(insert_input.type, insert_input.source_id, vid) for vid in vec_ids]
                )
                logger.debug(f"wrote to index map: {insert_input.source_id}, {[vid for vid in vec_ids]}")
        return {"success": True}

    except Exception as e:
        logger.error(f"Error inserting embeddings: {e}")
        return {"success": False, "error": str(e)}


@logger.catch(reraise=True)
async def put_embeddings_batch(batch_input: IngestInputBatch):
    # Ingests multiple items in a batch; supports storing fulltext and embedding selected fields
    try:
        collection_name = batch_input.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME
        
        logger.info( f"ingest into collection {collection_name}" );
        fulltext_ids = []

        pool = get_pg_pool()
        async with pool.acquire() as conn:
        # Optional fulltext storage
            if batch_input.store_fulltext:
                fulltext_texts = []
                fulltext_ids = []
                if batch_input.fulltext_types:
                    for content_set in batch_input.content:
                        fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in content_set.content_items if
                                                item.type in batch_input.fulltext_types])
                        fulltext_texts.append(fulltext)
                        fulltext_ids.append(content_set.source_id)
                else:
                    for content_set in batch_input.content:
                        fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in content_set.content_items])
                        fulltext_texts.append(fulltext)
                        fulltext_ids.append(content_set.source_id)

                await conn.fetch(
                    """
                    INSERT INTO fulltext (collection_name, source_id, text)
                    SELECT
                        $1,
                        s.source_id,
                        s.text
                    FROM unnest($2::text[], $3::text[]) AS s(source_id, text)
                    """,
                    collection_name,
                    fulltext_ids,
                    fulltext_texts,
                )

            # Prepare documents for embedding
            embed_docs = []
            source_ids = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.embedding_chunk_size,
                chunk_overlap=settings.embedding_chunk_overlap,
            )

            for idx, content_set in enumerate(batch_input.content):
                if batch_input.embed_content_types:
                    selected = [item.text for item in content_set.content_items if item.type in batch_input.embed_content_types]
                else:
                    selected = [item.text for item in content_set.content_items]

                if not selected:
                    continue

                splits = text_splitter.create_documents(selected)

                # Attach correct fulltext ID to chunks
                if batch_input.store_fulltext and idx < len(fulltext_ids):
                    for doc in splits:
                        doc.metadata["fulltext_source_id"] = fulltext_ids[idx]

                embed_docs.extend(splits)
                source_ids.extend([content_set.source_id] * len(splits))

            embed_store = get_vectorstore(with_embedding=True, collection_name=collection_name)
            logger.info( f"embedding into {collection_name} : {embed_docs}" )
            vec_ids = await embed_store.aadd_documents(embed_docs)
            await conn.executemany(
                "INSERT INTO source_vector_index_map (collection_name, source_id, vector_id) VALUES ($1, $2, $3)",
                [(batch_input.type, sid, vid) for sid, vid in zip(source_ids, vec_ids)]
            )

        return {"success": True}

    except Exception as e:
        logger.error(f"Error inserting embeddings: {e}")
        return {"success": False, "error": str(e)}

#
# @logger.catch(reraise=True)
# async def delete_embeddings_by_id(delete: DeleteInput):
#     """
#     Delete a single embedding entry by ID from a Chroma collection.
#     """
#     try:
#         collection_name = delete.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME
#         entry_id = delete.id
#
#         logger.info(f"delete_embedding id={entry_id} from {collection_name}")
#
#         # Embedding function not required for deletion
#         vector_store = get_vectorstore(
#             with_embedding=False,
#             collection_name=collection_name,
#         )
#
#         # Recommended Chroma/LangChain delete strategy: delete by ID
#         vector_store.delete(ids=[entry_id])
#
#         return {
#             "success": True,
#             "collection": collection_name,
#             "deleted_id": entry_id,
#         }
#
#     except Exception as e:
#         logger.error(f"Error deleting embedding id={delete.id}: {e}")
#         return {
#             "success": False,
#             "error": str(e),
#         }

@logger.catch(reraise=True)
async def delete_embeddings_by_id(delete: DeleteInput):
    """
    Delete embedding entries by source IDs:
    1) look up vector IDs in source_vector_index_map
    2) delete those vectors from Chroma
    3) delete the mapping rows from Postgres
    """
    try:
        collection_name = delete.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME
        source_ids = list(dict.fromkeys(delete.source_ids or []))  # de-dupe, keep order

        if not source_ids:
            return {"success": False, "error": "No source IDs found"}

        pool = get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT vector_id
                FROM source_vector_index_map
                WHERE collection_name = $1
                  AND source_id = ANY($2::text[])
                """,
                collection_name,
                source_ids,
            )
            vec_ids = [r["vector_id"] for r in rows]

            if not vec_ids:
                return {
                    "success": False,
                    "error": "No vector IDs found",
                }

            vector_store = get_vectorstore(with_embedding=False, collection_name=collection_name)
            vector_store.delete(ids=vec_ids)

            await conn.execute(
                """
                DELETE FROM source_vector_index_map
                WHERE collection_name = $1
                  AND source_id = ANY($2::text[])
                """,
                collection_name,
                source_ids,
            )
            logger.debug(f"deleted in source_vector_index_map: {source_ids}, {vec_ids}")

            await conn.execute(
                """
                DELETE FROM fulltext
                WHERE collection_name = $1
                  AND source_id = ANY($2::text[])
                """,
                collection_name,
                source_ids,
            )
            logger.debug(f"deleted in fulltext: {source_ids}, {vec_ids}")
        return {
            "success": True
        }

    except Exception as e:
        logger.error(f"Error deleting embeddings for source_ids={getattr(delete, 'source_ids', None)}: {e}")
        return {"success": False, "error": str(e)}
