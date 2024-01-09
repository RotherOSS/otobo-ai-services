from typing import Dict, List, Optional
from chromadb.api.types import OneOrMany, ID, IDs, Include
from fastapi import HTTPException, FastAPI, Depends, Body
from fastapi.responses import RedirectResponse
from langchain.pydantic_v1 import BaseModel
from mylibs.auth.auth import get_api_key
from mylibs.rag_chroma.chain import chain as rag_chroma_chain
from mylibs.classes.AppSettings import AppSettings
from mylibs.embedding.embedding import (
    Ticket,
    delete_embedding,
    delete_embeddings,
    get_embedding,
    get_embeddings,
    get_heartbeat,
    put_embeddings,
    query_embeddings,
)


settings = AppSettings()

app = FastAPI(
    title=settings.fastapi_title,
    version=settings.fastapi_version,
    description=settings.fastapi_description,
)


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
def heartbeat():
    return get_heartbeat()


#####################
### secure routes ###
#####################
class Question(BaseModel):
    question: str


@app.post(
    "/ai/tas/create-answer",
    name="Invoke LLM",
    description="Answers the submitted question using the stored data records.",
    dependencies=[Depends(get_api_key)],
)
def rag(body: Question):
    try:
        return rag_chroma_chain.invoke(body.question)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ai/embedding/filter/",
    name="Invoke many",
    description="""Get embedded data by search.\n
    At least one of the optional parameters 'ids' and 'process_id' must be specified.\n
    """,
    dependencies=[Depends(get_api_key)],
)
def filter(
    ids: Optional[OneOrMany[ID]] = None,
    process_id: Optional[str] | None = Body(default=None),
    limit: Optional[int] | None = Body(default=None),
    offset: Optional[int] | None = Body(default=None),
    include: Include = ["metadatas", "documents"],
):
    return get_embeddings(
        ids=ids,
        process_id=process_id,
        limit=limit,
        offset=offset,
        include=include,
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
def post_query(
    query_texts: List[str],
    where: Optional[Dict] = None,
    n_results: int = Body(default=10),
    include: Include = ["metadatas", "documents"],
):
    return query_embeddings(
        query_texts=query_texts, where=where, n_results=n_results, include=include
    )


@app.get(
    "/ai/embedding/invoke/{id}",
    name="Get",
    description="Get embedded data by id.",
    dependencies=[Depends(get_api_key)],
)
def get(id: str):
    return get_embedding(id)


@app.put(
    "/ai/embedding/insert/",
    name="Insert",
    description="Put list of embed datas into db, chunks and creates vectors.",
    dependencies=[Depends(get_api_key)],
)
async def put(embeds: List[Ticket]):
    return await put_embeddings(embeds)


@app.delete(
    "/ai/embedding/delete/{id}",
    name="Delete",
    description="Delete embedded data by id.",
    dependencies=[Depends(get_api_key)],
)
def delete(id: str):
    return delete_embedding(id)


@app.post(
    "/ai/embedding/delete-many/",
    name="Delete Many",
    description="Delete embedded data by filter.\nAt least one of the optional parameters 'ids', 'where' must be specified.",
)
def delete_many(
    ids: Optional[IDs] = None,
    where: Optional[Dict] = None,
):
    return delete_embeddings(
        ids=ids,
        where=where,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT)
