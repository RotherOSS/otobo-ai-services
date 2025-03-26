from typing import List

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from typing_extensions import TypedDict

from src.settings import AppSettings
from src.chains.simple_rag_chain import rag_chain
from src.embedding import get_vectorstore

settings = AppSettings()


class GraphState(TypedDict):
    """
    Status des Graphen.
    """

    question: str
    generation: str | None
    documents: List[Document] | None
    collection_name: str | None


@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def retrieve(state: GraphState):
    """
    Retrieve documents from vectorstore
    """
    logger.info("---ABRUFEN---")
    question = state["question"]

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs=settings.rag_search_kwargs)

    logger.debug(f"question: {question}")
    documents = retriever.invoke(
        question,
    )
    return {"documents": documents}


@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def generate(state: GraphState):
    """
    Generate answer using RAG on retrieved documents
    """
    logger.info("---GENERIEREN---")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})
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
