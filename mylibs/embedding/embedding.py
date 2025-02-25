from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger

from mylibs.classes.AppSettings import AppSettings
from mylibs.classes.Ticket import Ticket, UploadTicket

settings = AppSettings()


@logger.catch(reraise=True)
def embedding():
    return OllamaEmbeddings(
        base_url=settings.LLM_OLLAMA_URL, model=settings.LLM_OLLAMA_EMBEDDING_MODEL
    )


@logger.catch(reraise=True)
def get_vectorstore():
    db_embedding = embedding()
    return ElasticsearchStore(
        es_url=settings.es_url,
        index_name=settings.AI_VECTORSTORE_INDEX,
        embedding=db_embedding,
    )


@logger.catch(reraise=True)
def get_model(use_ollama_json_format: bool = False):
    if use_ollama_json_format:
        return ChatOllama(
            base_url=settings.LLM_OLLAMA_URL,
            model=settings.LLM_OLLAMA_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            headers={"otobo-api-key": settings.LLM_OTOBO_API_KEY},
            format="json",
        )
    else:
        return ChatOllama(
            base_url=settings.LLM_OLLAMA_URL,
            model=settings.LLM_OLLAMA_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            headers={"otobo-api-key": settings.LLM_OTOBO_API_KEY},
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
    """Get the current time in nanoseconds since epoch. Used to check if the server is alive.

    Raises:
        HTTPException: 500

    Returns:
        int: The current time in nanoseconds since epoch
    """
    try:
        client = ElasticsearchStore(
            es_url=settings.es_url, index_name=settings.AI_VECTORSTORE_INDEX
        )
        return client.client.info()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def get_embedding(id: str):
    """returns the embedding with the given id.

    Args:
        id (str): id of embedding

    Raises:
        HTTPException: 500

    Returns:
        GetResult: A GetResult object containing the results.
    """ """"""
    try:
        es = ElasticsearchStore(
            es_url=settings.es_url, index_name=settings.AI_VECTORSTORE_INDEX
        )
        embedding = es.get(index=settings.AI_VECTORSTORE_INDEX, id=id)
        return embedding
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def get_embeddings(
    ids: Any | None = None,
    process_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    # include: Include = ["metadatas", "documents"],
):
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
        es = ElasticsearchStore(
            es_url=settings.es_url, index_name=settings.AI_VECTORSTORE_INDEX
        )
        query = {"bool": {"filter": []}}
        if ids:
            query["bool"]["filter"].append({"terms": {"_id": ids}})
        if process_id:
            query["bool"]["filter"].append(
                {"match": {"metadata.process_id": process_id}}
            )
        embed = es.search(
            index=settings.AI_VECTORSTORE_INDEX,
            query=query,
            from_=offset,
            size=limit,
            # source_excludes=source_excludes,
        )
        return embed.body["hits"]["hits"]
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def query_embeddings(
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
        es: ElasticsearchStore = get_vectorstore()  # type: ignore
        # include parameter not supported with Elasticsearch
        where_list = where_filter if where_filter else None
        result = await es.asimilarity_search(query=query_texts[0] if query_texts else "", filter=where_list, k=n_results)  # type: ignore
        return result

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def put_embeddings(tickets: List[UploadTicket]):
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


@logger.catch(reraise=True)
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
        es = ElasticsearchStore(
            es_url=settings.es_url, index_name=settings.AI_VECTORSTORE_INDEX
        )
        response = es.delete(index=settings.AI_VECTORSTORE_INDEX, id=id)
        return response.body["_id"]
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def delete_embeddings(
    ids: Any | None = None,
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
        es = ElasticsearchStore(
            es_url=settings.es_url, index_name=settings.AI_VECTORSTORE_INDEX
        )

        body = {"query": {"bool": {"filter": []}}}
        filter = body["query"]["bool"]["filter"]
        if ids:
            filter.append({"terms": {"_id": ids}})
        if where:
            filter.append(where)

        response = es.delete_by_query(index=settings.AI_VECTORSTORE_INDEX, body=body)
        return {"deleted": response["deleted"]}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
