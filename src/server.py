import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, NotRequired, Optional, Sequence, TypedDict

from fastapi import Body, Depends, FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langserve import add_routes
from loguru import logger
from pydantic import BaseModel

from src.auth import get_api_key
from src.settings import AppSettings
from src.llm_embedding_utils import (
    UploadTicket,
    aquery_embeddings,
    delete_embedding,
    delete_embeddings,
    get_heartbeat,
    put_embeddings,
)
import importlib
import pkgutil
from fastapi.routing import APIRouter

settings = AppSettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ###  before the application starts ###
    os.makedirs("./data/log", exist_ok=True)

    logger.add(
        settings.OTOBO_AI_LOG_FILE,
        colorize=False,
        enqueue=True,
        level=os.getenv("LOGLEVEL", default="DEBUG"),
        rotation="1 MB",
    )
    logger.success(f"Starting server with loglevel: {settings.OTOBO_AI_LOG_LEVEL}")

    ### after the application has finished ###
    yield
    # watchfolder.stop()
    logger.success("Server has shut down gracefully.")


def register_rags(app: FastAPI):
    class InputDict(TypedDict):
        question: str
        generation: NotRequired[str]
        messages: NotRequired[Sequence[BaseMessage]]

    class OutputDict(TypedDict):
        question: str
        generation: NotRequired[str]
        messages: NotRequired[Sequence[BaseMessage]]
        documents: NotRequired[List[Document]]

    logger.info(f"blablarabl")
    base_dir = os.path.join(os.path.dirname(__file__), "rags")
    logger.info(base_dir)

    for entry in os.listdir(base_dir):
        logger.info(entry)
        try:
            subdir_path = os.path.join(base_dir, entry)
            if os.path.isdir(subdir_path):
                graph_path = os.path.join(subdir_path, "graph.py")
                if os.path.isfile(graph_path):
                    spec = importlib.util.spec_from_file_location(f"rags.{entry}.graph", graph_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    add_routes(
                        app,
                        module.graph.with_config(config),
                        input_type=InputDict,
                        output_type=OutputDict,
                        path=f"/otobo-ai/{entry}",
                        dependencies=[Depends(get_api_key)],
                    )
        except (ImportError, AttributeError) as e:
            # Log or handle as needed
            logger.error(f"Failed to load graph: {e}")


app = FastAPI(
    title=settings.fastapi_title,
    version=settings.fastapi_version,
    description=settings.fastapi_description,
    lifespan=lifespan,
)

config = None
if settings.OTOBO_AI_LANGFUSE_SK:
    from langfuse.callback import CallbackHandler

    langfuse_handler = CallbackHandler()
    try:
        langfuse_handler.auth_check()
        logger.success("Verbindung mit Langfuse hergestellt.")
        config = RunnableConfig(callbacks=[langfuse_handler])
    except Exception as e:
        logger.error(
            "Die Authentifizierung mit Langfuse ist fehlgeschlagen. Sind die env-Variablen OTOBO_AI_LANGFUSE_PK, OTOBO_AI_LANGFUSE_SK und OTOBO_AI_LANGFUSE_HOST gesetzt?"
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
    "/otobo-ai/db/heartbeat",
    description="Checks whether the database is reachable.",
)
async def heartbeat():
    return await get_heartbeat()


#####################
### secure routes ###
#####################
register_rags(app)


@app.post(
    "/otobo-ai/embedding/query/",
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
    where: Optional[Dict[str, Any]] = None,
    n_results: int = Body(default=10),
):
    return await aquery_embeddings(
        query_texts=query_texts,
        where_filter=where,
        n_results=n_results,
    )


@app.put(
    "/otobo-ai/embedding/insert/",
    name="Insert",
    description="Put list of embed datas into db, chunks and creates vectors.",
    dependencies=[Depends(get_api_key)],
)
async def put(embeds: List[UploadTicket]):
    return await put_embeddings(embeds)


@app.delete(
    "/otobo-ai/embedding/delete/{id}",
    name="Delete",
    description="Delete embedded data by id.",
    dependencies=[Depends(get_api_key)],
)
async def delete(id: str):
    return await delete_embedding(id)


@app.post(
    "/otobo-ai/embedding/delete-many/",
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

    uvicorn.run(app, host=settings.OTOBO_AI_HOST, port=settings.OTOBO_AI_PORT)
