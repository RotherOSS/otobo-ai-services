import os
from contextlib import asynccontextmanager
from typing import Any, List

from fastapi import Depends, FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langserve import add_routes
from loguru import logger

from src.auth import get_api_key
from src.settings import AppSettings
from src.llm_embedding_utils import (
    query_embeddings,
    get_heartbeat,
    put_embeddings,
    put_embeddings_batch,
    purge_vectorstore,
)
from src.data_models.ingest import IngestInput, IngestInputBatch
from src.data_models.retrieve import QueryInput
import importlib
from src.db import init_pg_pool, close_pg_pool

# Load app settings from environment or config
settings = AppSettings()


# Define app lifecycle behavior: setup logging and DB connection at startup,
# and ensure graceful shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("./data/log", exist_ok=True)

    logger.add(
        settings.OTOBO_AI_LOG_FILE,
        colorize=False,
        enqueue=True,
        level=os.getenv("LOGLEVEL", default="DEBUG"),
        rotation="1 MB",
    )
    logger.success(f"Starting server with loglevel: {settings.OTOBO_AI_LOG_LEVEL}")
    await init_pg_pool(settings.OTOBO_AI_PG_DSN)

    yield  # Main app runs here

    await close_pg_pool()
    logger.success("Server has shut down gracefully.")


# Dynamically load all RAG definitions from the `rags/` folder.
# For each valid RAG submodule (must contain graph.py and io_models.py),
# routes are added automatically under `/otobo-ai/{rag_name}`
def register_rags(app: FastAPI):
    base_dir = os.path.join(os.path.dirname(__file__), "rags")

    for entry in os.listdir(base_dir):
        try:
            subdir_path = os.path.join(base_dir, entry)
            if not os.path.isdir(subdir_path):
                continue

            graph_path = os.path.join(subdir_path, "graph.py")
            io_path = os.path.join(subdir_path, "io_models.py")

            if not (os.path.isfile(graph_path) and os.path.isfile(io_path)):
                logger.warning(f"Missing graph.py or io_models.py in {entry}")
                continue

            # Dynamically import graph.py
            graph_spec = importlib.util.spec_from_file_location(f"rags.{entry}.graph", graph_path)
            graph_module = importlib.util.module_from_spec(graph_spec)
            graph_spec.loader.exec_module(graph_module)

            # Dynamically import io_models.py
            io_spec = importlib.util.spec_from_file_location(f"rags.{entry}.io_models", io_path)
            io_module = importlib.util.module_from_spec(io_spec)
            io_spec.loader.exec_module(io_module)

            # Ensure required types exist
            if not all(hasattr(io_module, attr) for attr in ("RAGInput", "RAGOutput")):
                logger.warning(f"io_models.py in {entry} missing RAGInput or RAGOutput")
                continue

            # Add the RAG route to the FastAPI app
            add_routes(
                app,
                graph_module.graph.with_config(config),
                input_type=io_module.RAGInput,
                output_type=io_module.RAGOutput,
                path=f"/otobo-ai/{entry}",
                dependencies=[Depends(get_api_key)],
            )
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load graph: {e}")


# FastAPI app with configured metadata and lifecycle
app = FastAPI(
    title=settings.fastapi_title,
    version=settings.fastapi_version,
    description=settings.fastapi_description,
    lifespan=lifespan,
)

# Langfuse callback setup: if keys are configured, use it for observability.
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

# Redirect root path to the interactive docs
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Simple DB heartbeat check
@app.get(
    "/otobo-ai/db/heartbeat",
    description="Checks whether the database is reachable.",
)
async def heartbeat():
    return await get_heartbeat()


#####################
### secure routes ###
#####################

# Registers RAGs as secure endpoints
register_rags(app)


# Embedding query endpoint — performs similarity search
@app.post(
    "/otobo-ai/embedding/query",
    name="Query Embedding",
    description="Get embedded data by text-search.",
    dependencies=[Depends(get_api_key)],
)
async def post_query(retrieve: QueryInput):
    return await query_embeddings(retrieve)


# Ingest a single item for embedding
@app.put(
    "/otobo-ai/embedding/ingest",
    name="Ingest",
    description="Ingest an item of data for embedding.",
    dependencies=[Depends(get_api_key)],
)
async def put(embeds: IngestInput):
    return await put_embeddings(embeds)

# purge the vector store
@app.delete(
    "/otobo-ai/embedding/ingest",
    name="Ingest Delete",
    description="Purge the vector store.",
    dependencies=[Depends(get_api_key)],
)
async def put(embeds: IngestInput):
    collection_name = embeds.type or settings.OTOBO_AI_CHROMA_DEF_COL_NAME
    logger.error(f"purge {collection_name}");
    return await purge_vectorstore(True,collection_name)


# Ingest a batch of items for embedding
@app.put(
    "/otobo-ai/embedding/ingest-many",
    name="Ingest Many",
    description="Ingest a list of items of data for embedding.",
    dependencies=[Depends(get_api_key)],
)
async def put(embeds: IngestInputBatch):    
    return await put_embeddings_batch(embeds)

# Entry point for running locally without docker
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.OTOBO_AI_HOST, port=settings.OTOBO_AI_PORT)
