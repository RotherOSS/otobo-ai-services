from typing import List

from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from typing_extensions import TypedDict

from src.settings import AppSettings
from src.rags.tfc_rag1.chains import rag_chain
from src.llm_embedding_utils import get_vectorstore, query_embeddings
from src.data_models.retrieve import QueryInput

settings = AppSettings()


class GraphState(TypedDict):
    question: str
    generation: str | None
    faqs: List[Document] | None
    docs: List[Document] | None
    ticket_chunks: List[Document] | None
    ticket_pairs: List[Document] | None


def retrieve_function_generator(query_input: QueryInput, output: str):
    @logger.catch(reraise=True)
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    async def retrieve(state: GraphState):
        logger.info(f"---Retrieving from {query_input.type}---")
        query_input.query_text = state["question"]
        results = await query_embeddings(query_input)
        # logger.debug(f"Retrieved data: {results}")
        if query_input.retrieve_fulltext:
            results = [result.metadata["fulltext"] for result in results]
        else:
            results = [result.page_content for result in results]
        return {output: results}

    return retrieve


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

retrieve_input = QueryInput(
    query_text="",
    type="faq",
    retrieve_fulltext=True,
    n_results=3
)
workflow.add_node("retrieve_faq", retrieve_function_generator(retrieve_input, "faqs"))

retrieve_input = QueryInput(
    query_text="",
    type="documentation",
    retrieve_fulltext=False,
    n_results=3
)
workflow.add_node("retrieve_documentation", retrieve_function_generator(retrieve_input, "docs"))

retrieve_input = QueryInput(
    query_text="",
    type="ticket_chunks",
    retrieve_fulltext=False,
    n_results=3
)
workflow.add_node("retrieve_full_ticket_chunks", retrieve_function_generator(retrieve_input, "ticket_chunks"))

retrieve_input = QueryInput(
    query_text="",
    type="ticket_pairs",
    retrieve_fulltext=True,
    n_results=2
)
workflow.add_node("retrieve_ticket_pairs", retrieve_function_generator(retrieve_input, "ticket_pairs"))
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve_faq")
workflow.add_edge(START, "retrieve_documentation")
workflow.add_edge(START, "retrieve_full_ticket_chunks")
workflow.add_edge(START, "retrieve_ticket_pairs")
workflow.add_edge("retrieve_faq", "generate")
workflow.add_edge("retrieve_documentation", "generate")
workflow.add_edge("retrieve_full_ticket_chunks", "generate")
workflow.add_edge("retrieve_ticket_pairs", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()
