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


# This defines the state passed between graph nodes. Fields map to RAG steps.
class GraphState(TypedDict):
    question: str
    generation: str | None
    docs: List[Document] | None
    collection_name: str | None
    lang: str | None


@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def retrieve(state: GraphState):
    """
    Query the vector store using the user's question.
    This pulls relevant documents for RAG input.
    Retries up to 3 times on failure.
    """

    query_input = QueryInput(
        query_text="",
        type="documentation",  # hardcoded collection type
        retrieve_fulltext=False,
        n_results=6
    )

    lang = state["lang"]

    logger.info(f"---Retrieving from {query_input.type}---")
    logger.info(f"---Lang {lang}---")

    query_input.query_text = state["question"]
    results = await query_embeddings(query_input)

    # Only return document contents as strings
    results = [result.page_content for result in results]

    return {"docs": results}  # needs to match the name in GraphState


@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def generate(state: GraphState):
    """
    Uses the RAG chain to generate an answer based on the question and retrieved docs.
    Retries on failure.
    """
    logger.info("---Generating---")
    generation = rag_chain.invoke(state)
    return {"generation": generation}


# Define the RAG processing flow using LangGraph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)     # Step 1: Get documents
workflow.add_node("generate", generate)     # Step 2: Generate answer

workflow.set_entry_point("retrieve")        # Start from document retrieval
workflow.add_edge("retrieve", "generate")   # Move to generation
workflow.add_edge("generate", END)          # End workflow

graph = workflow.compile()                  # Compile to a runnable graph
