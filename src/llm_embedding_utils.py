from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
import chromadb
import json
import re
from uuid import uuid4

# Local imports from the project
from src.settings import AppSettings
from src.db import get_db_pool
from src.data_models.ingest import IngestInput, IngestInputBatch
from src.data_models.retrieve import QueryInput
from src.data_models.delete import DeleteInput

from langchain.callbacks.base import BaseCallbackHandler

from typing import Sequence


class DebugHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n--- PROMPT ---")
        for p in prompts:
            print(p)

    def on_llm_end(self, response, **kwargs):
        print("\n--- RESPONSE ---")
        print(response)


settings = AppSettings()


_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]")


def _get_chroma_client():
    # Raw chromadb client for operations spanning/managing whole collections (list,
    # delete, low-level add). Deliberately not going through langchain's Chroma
    # wrapper here, since its constructor always get-or-creates a collection as a
    # side effect, which we don't want for read/list/delete-only operations.
    return chromadb.PersistentClient(path=settings.OTOBO_AI_CHROMA_DIR)


def _physical_collection_name(type_name: str, label: str) -> str:
    # Chroma collection names must be 3-63 chars, start/end alphanumeric, and use only
    # [a-zA-Z0-9._-], so the label is sanitized to fit. Note: labels that sanitize to
    # the same string would collide onto the same physical collection.
    raw = f"{type_name}__{label}"
    safe = _SAFE_NAME_RE.sub("_", raw).strip("_.-")
    if len(safe) < 3:
        safe = (safe + "col")[:3]
    return safe[:63]


def _list_label_collection_names(type_name: str) -> list:
    # Enumerates every physical collection currently backing a given type
    prefix = f"{type_name}__"
    names = []
    for c in _get_chroma_client().list_collections():
        name = getattr(c, "name", c)
        if name.startswith(prefix):
            names.append(name)
    return names


def _delete_physical_collection(physical_name: str) -> None:
    try:
        _get_chroma_client().delete_collection(physical_name)
    except Exception as e:
        logger.debug(f"Collection {physical_name} not deleted (likely already absent): {e}")


def _delete_ids_from_physical_collection(physical_name: str, vec_ids: list) -> None:
    if not vec_ids:
        return
    try:
        collection = _get_chroma_client().get_collection(physical_name)
    except Exception:
        return  # Nothing to delete if the collection never existed.
    collection.delete(ids=vec_ids)


async def _embed_texts(texts: list) -> list:
    if not texts:
        return []
    embedding_model = get_embeddingsmodel()
    return await embedding_model.aembed_documents(texts)


async def _store_chunks(type_name: str, chunk_texts: list, chunk_metadatas: list, chunk_labels: list) -> list:
    """
    Embeds chunk_texts exactly once, then writes each chunk's (id, vector, text, metadata)
    into every one of its labels' physical Chroma collections. Returns the generated ids,
    one per chunk, shared across all of that chunk's label collections.
    """
    if not chunk_texts:
        return []

    chunk_ids = [str(uuid4()) for _ in chunk_texts]
    vectors = await _embed_texts(chunk_texts)

    indices_by_label = {}
    for i, labels in enumerate(chunk_labels):
        for lbl in labels:
            indices_by_label.setdefault(lbl, []).append(i)

    client = _get_chroma_client()
    for lbl, idxs in indices_by_label.items():
        physical_name = _physical_collection_name(type_name, lbl)
        collection = client.get_or_create_collection(name=physical_name)
        collection.add(
            ids=[chunk_ids[i] for i in idxs],
            embeddings=[vectors[i] for i in idxs],
            documents=[chunk_texts[i] for i in idxs],
            metadatas=[chunk_metadatas[i] for i in idxs],
        )

    return chunk_ids


def _merge_labels(existing: Sequence[str] | None, new: Sequence[str] | None) -> list:
    # Union of existing and new labels, preserving order, existing labels first.
    merged = list(existing or [])
    for lbl in (new or []):
        if lbl not in merged:
            merged.append(lbl)
    return merged


@logger.catch(reraise=True)
async def _get_existing_labels_batch(conn, collection_name: str, source_ids: Sequence[str]) -> dict:
    # Reads the current label list per source_id from whichever table already has it.
    source_ids = list(dict.fromkeys(sid for sid in source_ids if sid))
    if not source_ids:
        return {}

    labels_by_source = {sid: set() for sid in source_ids}

    rows = await conn.fetch(
        "SELECT source_id, labels FROM source_vector_index_map WHERE collection_name = %s AND source_id IN %s",
        collection_name,
        tuple(source_ids),
    )
    for row in rows:
        if row["labels"]:
            labels_by_source[row["source_id"]].update(json.loads(row["labels"]))

    rows = await conn.fetch(
        "SELECT source_id, labels FROM fulltext_documents WHERE collection_name = %s AND source_id IN %s",
        collection_name,
        tuple(source_ids),
    )
    for row in rows:
        if row["labels"]:
            labels_by_source[row["source_id"]].update(json.loads(row["labels"]))

    return {sid: list(lbls) for sid, lbls in labels_by_source.items()}


async def _get_existing_labels(conn, collection_name: str, source_id: str) -> list:
    if not source_id:
        return []
    return (await _get_existing_labels_batch(conn, collection_name, [source_id])).get(source_id, [])


@logger.catch(reraise=True)
def get_embeddingsmodel():
    # Returns an embedding model instance using Ollama with config values
    return OpenAIEmbeddings(
        model=settings.OTOBO_AI_EMBEDDING_MODEL,
        base_url=f"{settings.OTOBO_AI_LLM_HOST}/v1",
        encoding_format= None,
        api_key=settings.OTOBO_AI_LLM_API_KEY or "ollama",
        check_embedding_ctx_length=False,
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
async def purge_collection(
        collection_name: str = settings.OTOBO_AI_CHROMA_DEF_COL_NAME,
        labels: Sequence[str] | None = None,
):
    logger.info(f"Purge: {collection_name}, labels={labels}")

    pool = get_db_pool()
    async with pool.acquire() as conn:
        if labels:
            labels_list = list(labels)

            # Dropping a label means deleting its whole physical collection outright;
            # each entry's OTHER labels live in separate collections and stay untouched.
            for lbl in labels_list:
                _delete_physical_collection(_physical_collection_name(collection_name, lbl))

            labels_json = json.dumps(labels_list)
            rows = await conn.fetch(
                """
                SELECT source_id, labels
                FROM source_vector_index_map
                WHERE collection_name = %s
                  AND JSON_OVERLAPS(labels, %s)
                """,
                collection_name,
                labels_json,
            )
            labels_by_source = {row["source_id"]: (json.loads(row["labels"]) if row["labels"] else []) for row in rows}

            rows = await conn.fetch(
                """
                SELECT source_id, labels
                FROM fulltext_documents
                WHERE collection_name = %s
                  AND JSON_OVERLAPS(labels, %s)
                """,
                collection_name,
                labels_json,
            )
            for row in rows:
                labels_by_source.setdefault(row["source_id"], json.loads(row["labels"]) if row["labels"] else [])

            to_remove = set(labels_list)
            fully_removed = []
            for source_id, current_labels in labels_by_source.items():
                remaining = [lbl for lbl in current_labels if lbl not in to_remove]
                if not remaining:
                    fully_removed.append(source_id)
                else:
                    remaining_json = json.dumps(remaining)
                    await conn.execute(
                        "UPDATE source_vector_index_map SET labels = %s WHERE collection_name = %s AND source_id = %s",
                        remaining_json, collection_name, source_id,
                    )
                    await conn.execute(
                        "UPDATE fulltext_documents SET labels = %s WHERE collection_name = %s AND source_id = %s",
                        remaining_json, collection_name, source_id,
                    )

            if fully_removed:
                await conn.execute(
                    "DELETE FROM source_vector_index_map WHERE collection_name = %s AND source_id IN %s",
                    collection_name, tuple(fully_removed),
                )
                await conn.execute(
                    "DELETE FROM fulltext_documents WHERE collection_name = %s AND source_id IN %s",
                    collection_name, tuple(fully_removed),
                )
        else:
            for physical_name in _list_label_collection_names(collection_name):
                _delete_physical_collection(physical_name)
            await conn.execute("DELETE FROM source_vector_index_map WHERE collection_name = %s", collection_name)
            await conn.execute("DELETE FROM fulltext_documents WHERE collection_name = %s", collection_name)

    return { "success": True  }

@logger.catch(reraise=True)
async def purge_vectorstore(with_embedding: bool = True):
    collections = [ "faqs", "ticket_pairs", "ticket_chunks", "docs"  ]
    for collection in collections:

        await purge_collection(with_embedding=with_embedding, collection_name=collection )

    return { "success": True  }


@logger.catch(reraise=True)
def get_model(use_ollama_json_format: bool = False, eval: bool = False):
    # Instantiates a chat model via Ollama
    return ChatOpenAI(
        model=settings.OTOBO_AI_LLM_EVAL_MODEL if eval else settings.OTOBO_AI_LLM_MODEL,
        base_url=f"{settings.OTOBO_AI_LLM_HOST}/v1",
        api_key=settings.OTOBO_AI_LLM_API_KEY or "ollama",  # must be non-empty
        temperature=settings.OTOBO_AI_LLM_EVAL_TEMPERATURE if eval else settings.OTOBO_AI_LLM_TEMPERATURE,
        use_responses_api=False,  # important for Ollama compatibility
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
    # Main query endpoint: retrieves most similar documents for a single label's collection
    try:
        collection_name = retrieve.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME
        physical_name = _physical_collection_name(collection_name, retrieve.label)

        logger.info( f"query_embeddings from {collection_name} (label={retrieve.label})" )
        vector_store = get_vectorstore(with_embedding=True, collection_name=physical_name)
        results = await vector_store.asimilarity_search(query=retrieve.query_text, k=retrieve.n_results)

        # Optionally enrich results with full text from the SQL database
        if retrieve.retrieve_fulltext:
            source_ids = {doc.metadata.get("source_id") for doc in results if doc.metadata.get("source_id")}
            if source_ids:
                pool = get_db_pool()
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT source_id, text FROM fulltext_documents WHERE collection_name = %s "
                        "AND source_id IN %s",
                        collection_name,
                        tuple(source_ids)
                    )
                id_to_text = {row["source_id"]: row["text"] for row in rows}

                for doc in results:
                    ft_id = doc.metadata.get("source_id")
                    if ft_id in id_to_text:
                        doc.metadata["fulltext"] = id_to_text[ft_id]

        return results

    except Exception as e:
        logger.error(f"Error asynch. querying embeddings: {e}")
        return {"success": False, "error": str(e)}


@logger.catch(reraise=True)
async def _purge_source_ids_from_labels_batch(conn, collection_name: str, labels_by_source: dict):
    """
    Deletes each source_id's vectors from the given label collections and
    its rows in MariaDB. labels_by_source maps source_id -> the labels
    whose physical collections currently hold that source_id's vectors.
    """
    source_ids = [sid for sid in labels_by_source if sid]
    if not source_ids:
        return

    rows = await conn.fetch(
        "SELECT source_id, vector_id FROM source_vector_index_map WHERE collection_name = %s AND source_id IN %s",
        collection_name,
        tuple(source_ids),
    )
    vec_ids_by_source = {}
    for r in rows:
        vec_ids_by_source.setdefault(r["source_id"], []).append(r["vector_id"])

    # Group deletions by physical collection so each collection is touched at most once.
    ids_by_physical = {}
    for sid, vec_ids in vec_ids_by_source.items():
        for lbl in labels_by_source.get(sid, []):
            physical_name = _physical_collection_name(collection_name, lbl)
            ids_by_physical.setdefault(physical_name, set()).update(vec_ids)

    for physical_name, ids in ids_by_physical.items():
        _delete_ids_from_physical_collection(physical_name, list(ids))

    await conn.execute(
        "DELETE FROM source_vector_index_map WHERE collection_name = %s AND source_id IN %s",
        collection_name,
        tuple(source_ids),
    )
    await conn.execute(
        "DELETE FROM fulltext_documents WHERE collection_name = %s AND source_id IN %s",
        collection_name,
        tuple(source_ids),
    )


@logger.catch(reraise=True)
async def _remove_labels_from_source(conn, collection_name: str, source_id: str, labels_to_remove: Sequence[str] | None):
    """
    Removes labels_to_remove from a source_id's stored label list.
    If labels_to_remove is falsy, the entry is deleted entirely.
    If removing the labels empties the label list, the entry is deleted entirely.
    Otherwise, the entry is kept: just the removed labels' vector copies are dropped from
    their physical collections, and the DB rows' label lists are updated in place.
    """
    if not labels_to_remove:
        current_labels = await _get_existing_labels(conn, collection_name, source_id)
        await _purge_source_ids_from_labels_batch(conn, collection_name, {source_id: current_labels})
        return

    to_remove = set(labels_to_remove)

    rows = await conn.fetch(
        "SELECT vector_id, labels FROM source_vector_index_map WHERE collection_name = %s AND source_id = %s",
        collection_name,
        source_id,
    )

    if rows:
        current_labels = json.loads(rows[0]["labels"]) if rows[0]["labels"] else []
        remaining = [lbl for lbl in current_labels if lbl not in to_remove]
        vec_ids = [r["vector_id"] for r in rows]

        for lbl in current_labels:
            if lbl in to_remove:
                _delete_ids_from_physical_collection(_physical_collection_name(collection_name, lbl), vec_ids)

        if not remaining:
            await conn.execute(
                "DELETE FROM source_vector_index_map WHERE collection_name = %s AND source_id = %s",
                collection_name, source_id,
            )
            await conn.execute(
                "DELETE FROM fulltext_documents WHERE collection_name = %s AND source_id = %s",
                collection_name, source_id,
            )
        else:
            remaining_json = json.dumps(remaining)
            await conn.execute(
                "UPDATE source_vector_index_map SET labels = %s WHERE collection_name = %s AND source_id = %s",
                remaining_json, collection_name, source_id,
            )
            await conn.execute(
                "UPDATE fulltext_documents SET labels = %s WHERE collection_name = %s AND source_id = %s",
                remaining_json, collection_name, source_id,
            )
        return

    # No embedded vectors for this source_id -- it may still have a fulltext-only row.
    row = await conn.fetchrow(
        "SELECT labels FROM fulltext_documents WHERE collection_name = %s AND source_id = %s",
        collection_name,
        source_id,
    )
    if not row:
        return

    current_labels = json.loads(row["labels"]) if row["labels"] else []
    remaining = [lbl for lbl in current_labels if lbl not in to_remove]

    if not remaining:
        await conn.execute(
            "DELETE FROM fulltext_documents WHERE collection_name = %s AND source_id = %s",
            collection_name, source_id,
        )
    else:
        await conn.execute(
            "UPDATE fulltext_documents SET labels = %s WHERE collection_name = %s AND source_id = %s",
            json.dumps(remaining), collection_name, source_id,
        )


@logger.catch(reraise=True)
async def put_embeddings(insert_input: IngestInput):
    # Ingests a single item into the vector store, optionally storing raw text in SQL
    try:
        collection_name = insert_input.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME
        source_id = insert_input.source_id

        pool = get_db_pool()
        async with pool.acquire() as conn:
            # Add new labels to existing ones
            existing_labels = await _get_existing_labels(conn, collection_name, source_id)
            merged_labels = _merge_labels(existing_labels, insert_input.labels)

            # Re-ingest overwrites: drop the old vectors (from whichever label collections
            # currently hold them) and the old DB rows before writing the fresh entry.
            await _purge_source_ids_from_labels_batch(conn, collection_name, {source_id: existing_labels})

            if insert_input.store_fulltext:
                if insert_input.fulltext_types:
                    fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in insert_input.content if
                                            item.type in insert_input.fulltext_types])
                else:
                    fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in insert_input.content])

                await conn.execute(
                    "INSERT INTO fulltext_documents (collection_name, source_id, text, labels) VALUES (%s, %s, %s, %s)",
                    collection_name,
                    source_id,
                    fulltext,
                    json.dumps(merged_labels)
                )

            # Select content types to embed (configurable)
            if insert_input.embed_content_types:
                selected = [item.text for item in insert_input.content if item.type in insert_input.embed_content_types]
            else:
                selected = [item.text for item in insert_input.content]

            # Split into chunks and embed
            if selected:
                if not merged_labels:
                    logger.warning(
                        f"source_id={source_id} has no labels; content stored but not embedded "
                        f"(not retrievable via vector search)"
                    )
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=settings.embedding_chunk_size,
                        chunk_overlap=settings.embedding_chunk_overlap,
                    )
                    all_splits = text_splitter.create_documents(selected)

                    chunk_texts = [doc.page_content for doc in all_splits]
                    chunk_metadatas = []
                    for doc in all_splits:
                        meta = dict(doc.metadata)
                        if source_id:
                            meta["source_id"] = source_id
                        chunk_metadatas.append(meta)

                    chunk_ids = await _store_chunks(
                        collection_name, chunk_texts, chunk_metadatas, [merged_labels] * len(chunk_texts)
                    )
                    await conn.executemany(
                        "INSERT INTO source_vector_index_map (collection_name, source_id, vector_id, labels) VALUES (%s, %s, %s, %s)",
                        [(collection_name, source_id, cid, json.dumps(merged_labels)) for cid in chunk_ids]
                    )
                    logger.debug(f"wrote to index map: {source_id}, {chunk_ids}")
        return {"success": True}

    except Exception as e:
        logger.error(f"Error inserting embeddings: {e}")
        return {"success": False, "error": str(e)}


@logger.catch(reraise=True)
async def put_embeddings_batch(batch_input: IngestInputBatch):
    # Ingests multiple items in a batch; supports storing fulltext and embedding selected fields
    try:
        collection_name = batch_input.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME

        logger.info( f"ingest into collection {collection_name}" )

        pool = get_db_pool()
        async with pool.acquire() as conn:
            batch_source_ids = [content_set.source_id for content_set in batch_input.content]

            # Merge new labels to existing labels, then purge and insert.
            existing_labels_by_source = await _get_existing_labels_batch(conn, collection_name, batch_source_ids)
            merged_labels_by_source = {
                content_set.source_id: _merge_labels(
                    existing_labels_by_source.get(content_set.source_id, []),
                    content_set.labels if batch_input.has_labels else None,
                )
                for content_set in batch_input.content
            }

            await _purge_source_ids_from_labels_batch(conn, collection_name, existing_labels_by_source)

            # Optional fulltext storage
            if batch_input.store_fulltext:
                fulltext_texts = []
                source_ids = []
                for content_set in batch_input.content:
                    if batch_input.fulltext_types:
                        fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in content_set.content_items if
                                                item.type in batch_input.fulltext_types])
                    else:
                        fulltext = "\n\n".join([f"{item.type}: {item.text}" for item in content_set.content_items])
                    fulltext_texts.append(fulltext)
                    source_ids.append(content_set.source_id)

                labels_json = [json.dumps(merged_labels_by_source[sid]) for sid in source_ids]
                await conn.executemany(
                    "INSERT INTO fulltext_documents (collection_name, source_id, text, labels) VALUES (%s, %s, %s, %s)",
                    [(collection_name, sid, text, lbl) for sid, text, lbl in zip(source_ids, fulltext_texts, labels_json)]
                )

            # Prepare chunks for embedding (embedded once, written into each chunk's label collections)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.embedding_chunk_size,
                chunk_overlap=settings.embedding_chunk_overlap,
            )

            chunk_texts = []
            chunk_metadatas = []
            chunk_source_ids = []
            chunk_labels = []

            for content_set in batch_input.content:
                if batch_input.embed_content_types:
                    selected = [item.text for item in content_set.content_items if item.type in batch_input.embed_content_types]
                else:
                    selected = [item.text for item in content_set.content_items]

                if not selected:
                    continue

                merged_labels = merged_labels_by_source[content_set.source_id]
                if not merged_labels:
                    logger.warning(
                        f"source_id={content_set.source_id} has no labels; content stored but not embedded "
                        f"(not retrievable via vector search)"
                    )
                    continue

                splits = text_splitter.create_documents(selected)
                for doc in splits:
                    meta = dict(doc.metadata)
                    if content_set.source_id:
                        meta["source_id"] = content_set.source_id
                    chunk_texts.append(doc.page_content)
                    chunk_metadatas.append(meta)
                    chunk_source_ids.append(content_set.source_id)
                    chunk_labels.append(merged_labels)

            logger.info( f"embedding into {collection_name} : {len(chunk_texts)} chunks" )
            chunk_ids = await _store_chunks(collection_name, chunk_texts, chunk_metadatas, chunk_labels)
            if chunk_ids:
                await conn.executemany(
                    "INSERT INTO source_vector_index_map (collection_name, source_id, vector_id, labels) VALUES (%s, %s, %s, %s)",
                    [(collection_name, sid, cid, json.dumps(lbls)) for sid, cid, lbls in zip(chunk_source_ids, chunk_ids, chunk_labels)]
                )

        return {"success": True}

    except Exception as e:
        logger.exception(f"Error inserting embeddings: {e}")
        return {"success": False, "error": str(e)}


@logger.catch(reraise=True)
async def delete_embeddings_by_id(delete: DeleteInput):
    """
    Delete (or partially un-label) embedding entries by source ID.

    For each entry: if `delete.has_labels` is set and the entry carries labels, only
    those labels are removed (the entry is deleted entirely if that empties its label
    list). Otherwise, the entry is deleted entirely, regardless of any labels present.
    """
    try:
        collection_name = delete.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME

        # De-dupe by source_id, keeping the last entry's labels for a given id.
        entries = {}
        for entry in delete.source_ids or []:
            if entry.source_id:
                entries[entry.source_id] = entry.labels

        if not entries:
            return {"success": False, "error": "No source IDs found"}

        pool = get_db_pool()
        async with pool.acquire() as conn:
            for source_id, labels in entries.items():
                labels_to_remove = labels if delete.has_labels else None
                await _remove_labels_from_source(conn, collection_name, source_id, labels_to_remove)

            logger.debug(f"processed delete for source_ids={list(entries.keys())}")

        return {"success": True}

    except Exception as e:
        logger.error(f"Error deleting embeddings for source_ids={getattr(delete, 'source_ids', None)}: {e}")
        return {"success": False, "error": str(e)}
