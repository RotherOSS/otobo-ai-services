from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger
from pydantic import BaseModel

from mylibs.classes.AppSettings import AppSettings
from mylibs.classes.Ticket import Ticket, UploadTicket
from typing import Optional

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
)
from langchain_community.query_constructors.elasticsearch import ElasticsearchTranslator

settings = AppSettings()


@logger.catch(reraise=True)
def get_embeddingsmodel():
    return OllamaEmbeddings(
        base_url=settings.LLM_OLLAMA_URL, model=settings.LLM_OLLAMA_EMBEDDING_MODEL
    )


@logger.catch(reraise=True)
def get_vectorstore(with_embedding: bool = True):
    if with_embedding:
        db_embedding = get_embeddingsmodel()
        return ElasticsearchStore(
            es_url=settings.es_url,
            index_name=settings.AI_VECTORSTORE_INDEX,
            embedding=db_embedding,
        )
    else:
        return ElasticsearchStore(
            es_url=settings.es_url, index_name=settings.AI_VECTORSTORE_INDEX
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
    """returns the embedding with the given id.

    Args:
        id (str): id of embedding

    Raises:
        HTTPException: 500

    Returns:
        GetResult: A GetResult object containing the results.
    """ """"""
    try:
        es = Elasticsearch(hosts=[settings.es_url])
        embedding = es.get(index=settings.AI_VECTORSTORE_INDEX, id=id)
        return embedding
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


class Search(BaseModel):
    ids: Optional[List[str]] = None
    process_id: Optional[str] = None
    gdpr_id: Optional[str] = None
    type: Optional[str] = None


def construct_comparisons(query: Search):
    comparisons = []
    # actually not working with List[str]
    # if query.ids:
    #     comparisons.append(
    #         Comparison(
    #             comparator=Comparator.EQ,
    #             attribute="metadata.id",
    #             value=query.ids,
    #         )
    #     )
    if query.process_id:
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="process_id",
                value=query.process_id,
            )
        )
    if query.gdpr_id:
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="gdpr_id",
                value=query.gdpr_id,
            )
        )
    if query.type:
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="type",
                value=query.type,
            )
        )
    return comparisons


@logger.catch(reraise=True)
async def search_embeddings(
    ids: Optional[str] = None,
    process_id: Optional[str] = None,
    gdpr_id: Optional[str] = None,
    type: Optional[str] = None,
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
        search_query = Search(
            ids=[id for id in ids.split(",")] if ids else None,
            process_id=process_id,
            gdpr_id=gdpr_id,
            type=type,
        )
        comparisons = construct_comparisons(search_query)
        _filter = Operation(operator=Operator.AND, arguments=comparisons)
        query = ElasticsearchTranslator().visit_operation(_filter)

        if search_query.ids:
            query["bool"]["must"].append({"terms": {"_id": search_query.ids}})

        logger.debug(f"Query: {query}")

        # id_list = [id for id in ids.split(",")]
        # if not id_list:
        #     raise HTTPException(status_code=400, detail="No ids given")

        es = Elasticsearch(hosts=[settings.es_url])
        # query = {"terms": {"metadata.process_id.keyword": id_list}}
        results = es.search(
            index=settings.AI_VECTORSTORE_INDEX,
            query=query,
            from_=offset,
            size=limit,
            _source={"excludes": ["vector"]},
        )

        return results["hits"]["hits"]
    except Exception as e:
        # print(e)
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
        es = get_vectorstore()
        # include parameter not supported with Elasticsearch
        where_list = where_filter if where_filter and where_filter[0] else None
        result = es.similarity_search(query=query_texts[0] if query_texts else "", filter=where_list, k=n_results)  # type: ignore
        return result

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@logger.catch(reraise=True)
async def aquery_embeddings(
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
        es = get_vectorstore()
        # include parameter not supported with Elasticsearch
        where_list = where_filter if where_filter and where_filter[0] else None
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
        es = get_vectorstore(with_embedding=False)
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
        es = get_vectorstore(with_embedding=False)

        body = {"query": {"bool": {"filter": []}}}
        filter = body["query"]["bool"]["filter"]
        if ids:
            filter.append({"terms": {"_id": ids}})
        if where:
            filter.append(where)
        es = Elasticsearch(hosts=[settings.es_url])

        response = es.delete_by_query(index=settings.AI_VECTORSTORE_INDEX, body=body)
        return {"deleted": response["deleted"]}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
