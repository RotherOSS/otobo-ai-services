import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, NotRequired, Optional, TypedDict

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langserve import add_routes
from loguru import logger
from pydantic import BaseModel

from mylibs.auth.auth import get_api_key
from mylibs.classes.AppSettings import AppSettings
from mylibs.embedding.embedding import (
    UploadTicket,
    delete_embedding,
    delete_embeddings,
    get_embedding,
    get_embeddings,
    get_heartbeat,
    put_embeddings,
    query_embeddings,
)
from mylibs.rag.graph import graph as rag_graph
from mylibs.rag_compression.chain import chain as rag_compression_chain
from mylibs.rag_task.chain import chain as rag_task_chain

settings = AppSettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ###  before the application starts ###
    os.makedirs("./data/log", exist_ok=True)

    logger.add(
        settings.LOG_FILE,
        colorize=False,
        enqueue=True,
        level=os.getenv("LOGLEVEL", default="DEBUG"),
        rotation="1 MB",
    )
    logger.success(f"Starting server with loglevel: {settings.LOG_LEVEL}")

    ### after the application has finished ###
    yield
    # watchfolder.stop()
    logger.success("Server has shut down gracefully.")


app = FastAPI(
    title=settings.fastapi_title,
    version=settings.fastapi_version,
    description=settings.fastapi_description,
    lifespan=lifespan,
)

config = None
if settings.LANGFUSE_SECRET_KEY:
    from langfuse.callback import CallbackHandler

    langfuse_handler = CallbackHandler()
    try:
        # Todo: This method is blocking. It is discouraged to use it in production code.
        langfuse_handler.auth_check()
        logger.success("Verbindung mit Langfuse hergestellt.")
        config = RunnableConfig(callbacks=[langfuse_handler])
    except Exception as e:
        logger.error(
            "Die Authentifizierung mit Langfuse ist fehlgeschlagen. Sind die env-Variablen LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY und LANGFUSE_HOST gesetzt?"
        )
        logger.error(e)
else:
    logger.info("Langfuse ist deaktiviert.")


###################
### open routes ###
###################
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.get(
    "/ai/db/heartbeat",
    description="Get the current time in nanoseconds since epoch. Used to check if the database server is alive.",
)
async def heartbeat():
    return await get_heartbeat()


#####################
### secure routes ###
#####################
class Question(BaseModel):
    question: str


# @app.post(
#     "/ai/tas/create-answer",
#     name="Invoke LLM",
#     description="Answers the submitted question using the stored data records.",
#     dependencies=[Depends(get_api_key)],
# )
# async def rag(body: Question):
#     try:
#         return await rag_chain.ainvoke(body.question)
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail=str(e))
class InputDict(TypedDict):
    question: str
    generation: NotRequired[str]
    documents: NotRequired[List[Document]]
    collection_name: NotRequired[str]


class OutputDict(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    collection_name: NotRequired[str]


add_routes(
    app,
    rag_graph.with_config(config),
    input_type=InputDict,
    output_type=OutputDict,
    path="/ai/tas/create-answer",
)


@app.post(
    "/ai/tas/compression",
    name="Contextual compression",
    description="Answers the submitted question using the stored data records and contextual compression",
    dependencies=[Depends(get_api_key)],
)
async def rag(body: Question):
    try:
        response = await rag_compression_chain.ainvoke(body.question)
        return response
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ai/tas/task",
    name="Task compression",
    description="Answers the submitted question using the complete task.",
    dependencies=[Depends(get_api_key)],
)
async def rag(body: Question):
    try:
        response = await rag_task_chain.ainvoke(body.question)
        return response
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ai/embedding/filter/",
    name="Invoke many",
    description="""Get embedded data by search.\n
    At least one of the optional parameters 'ids' and 'process_id' must be specified.\n
    Include - ChromaDB: A list of what to include in the results. Can contain "embeddings", "metadatas", "documents", "distances".\n
    Include - Elasticserach: use "embeddings" to include vectors in result.
    """,
    dependencies=[Depends(get_api_key)],
)
async def filter(
    ids: Any | None = None,
    process_id: Optional[str] | None = Body(default=None),
    limit: Optional[int] | None = Body(default=None),
    offset: Optional[int] | None = Body(default=None),
    # include: Include = ["metadatas", "documents"],
):
    return await get_embeddings(
        ids=ids,
        process_id=process_id,
        limit=limit,
        offset=offset,
        # include=include,
    )


@app.post(
    "/ai/embedding/query/",
    name="Query Embedding",
    description="""Get embedded data by text-search.\n
    query_texts - The document texts to get the closes neighbors of. Optional. \n
    n_results - The number of neighbors to return for each query_embedding or query_texts. Optional.
    include - A list of what to include in the results. Can contain "embeddings", "metadatas", "documents", "distances".
    Ids are always included. Defaults to ["metadatas", "documents"]. Optional.""",
    dependencies=[Depends(get_api_key)],
)
async def post_query(
    query_texts: Optional[List[str]] = None,
    where: Optional[List[Dict]] = None,
    n_results: int = Body(default=10),
    # include: Include = ["metadatas", "documents"],
):
    return await query_embeddings(
        query_texts=query_texts,
        where_filter=where,
        n_results=n_results,
        # include=include,
    )


@app.get(
    "/ai/embedding/invoke/{id}",
    name="Get",
    description="Get embedded data by id.",
    dependencies=[Depends(get_api_key)],
)
async def get(id: str):
    return await get_embedding(id)


@app.put(
    "/ai/embedding/insert/",
    name="Insert",
    description="Put list of embed datas into db, chunks and creates vectors.",
    dependencies=[Depends(get_api_key)],
)
async def put(embeds: List[UploadTicket]):
    return await put_embeddings(embeds)


@app.delete(
    "/ai/embedding/delete/{id}",
    name="Delete",
    description="Delete embedded data by id.",
    dependencies=[Depends(get_api_key)],
)
async def delete(id: str):
    return await delete_embedding(id)


@app.post(
    "/ai/embedding/delete-many/",
    name="Delete Many",
    description="Delete embedded data by filter.\nAt least one of the optional parameters 'ids', 'where' must be specified.",
)
async def delete_many(
    ids: Any | None = None,
    where: Optional[Dict] = None,
):
    return await delete_embeddings(
        ids=ids,
        where=where,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.AI_API_SERVER_HOST, port=settings.AI_API_SERVER_PORT)
