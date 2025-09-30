from typing import List

from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from typing_extensions import TypedDict

from src.settings import AppSettings
from src.rags.tfd_rag1.chains import rag_chain, eval_chain
from src.llm_embedding_utils import query_embeddings
from src.data_models.retrieve import QueryInput

settings = AppSettings()


# Shared state format passed between steps in the workflow
class GraphState(TypedDict):
    do_scoring: bool | None
    question: str
    generation: str | None
    faqs: List[Document] | None
    docs: List[Document] | None
    ticket_chunks: List[Document] | None
    ticket_pairs: List[Document] | None
    score: float | None


# Creates a retrieval function for the given input source and maps results to output key
def retrieve_function_generator(query_input: QueryInput, output: str):
    @logger.catch(reraise=True)
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    async def retrieve(state: GraphState):
        logger.info(f"---Retrieving from {query_input.type}---")
        query_input.query_text = state["question"]
        results = await query_embeddings(query_input)

        # Decide what to return: full text or just page content
        if query_input.retrieve_fulltext:
            results = [result.metadata["fulltext"] for result in results]
        else:
            results = [result.page_content for result in results]

        logger.info(results)    
        return {output: results}

    return retrieve


# Generates a response from retrieved content
@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def generate(state: GraphState):
    logger.info("---Generating---")
    generation = rag_chain.invoke(state)
    return {"generation": generation}


# Scores the generated output (if requested)
@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def evaluate(state: GraphState):
    if "do_scoring" in state and state["do_scoring"]:
        logger.info("---Evaluating---")
        score = eval_chain.invoke(state)
        return {"score": score}


# Define the full graph-based workflow
workflow = StateGraph(GraphState)

# Define multiple retrieval steps for different sources
workflow.add_node("retrieve_faq", retrieve_function_generator(
    QueryInput(query_text="", type="faq", retrieve_fulltext=True, n_results=3), "faqs"))

workflow.add_node("retrieve_documentation", retrieve_function_generator(
    QueryInput(query_text="", type="documentation", retrieve_fulltext=False, n_results=3), "docs"))

workflow.add_node("retrieve_full_ticket_chunks", retrieve_function_generator(
    QueryInput(query_text="", type="ticket_chunks", retrieve_fulltext=False, n_results=3), "ticket_chunks"))

workflow.add_node("retrieve_ticket_pairs", retrieve_function_generator(
    QueryInput(query_text="", type="ticket_pairs", retrieve_fulltext=True, n_results=2), "ticket_pairs"))

# Generation and optional evaluation step
workflow.add_node("generate", generate)
workflow.add_node("evaluate", evaluate)

# Define edges (execution order)
workflow.add_edge(START, "retrieve_faq")
workflow.add_edge(START, "retrieve_documentation")
workflow.add_edge(START, "retrieve_full_ticket_chunks")
workflow.add_edge(START, "retrieve_ticket_pairs")
workflow.add_edge("retrieve_faq", "generate")
workflow.add_edge("retrieve_documentation", "generate")
workflow.add_edge("retrieve_full_ticket_chunks", "generate")
workflow.add_edge("retrieve_ticket_pairs", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_edge("generate", END)
workflow.add_edge("evaluate", END)

# Compile into executable graph
graph = workflow.compile()
