from typing import List

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from typing_extensions import TypedDict

from src.settings import AppSettings
from src.rags.simple_rag.chains import rag_chain
from src.llm_embedding_utils import query_embeddings
from src.data_models.retrieve import QueryInput

settings = AppSettings()


class GraphState(TypedDict):
    question: str
    generation: str | None
    docs: List[Document] | None
    collection_name: str | None


@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def retrieve(state: GraphState):
    """
    Retrieve documents from vectorstore
    """
    query_input = QueryInput(
        query_text="",
        type="documentation",
        retrieve_fulltext=False,
        n_results=6
    )
    logger.info(f"---Retrieving from {query_input.type}---")
    query_input.query_text = state["question"]
    results = await query_embeddings(query_input)
    results = [result.page_content for result in results]

    return {"docs": results}  # needs to match the name in GraphState


@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def generate(state: GraphState):
    """
    Generate answer using RAG on retrieved documents
    """
    logger.info("---Generating---")

    generation = rag_chain.invoke(state)
    return {
        "generation": generation,
    }


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()
