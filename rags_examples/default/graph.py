#  Default RAG Version 1

from typing import List, Annotated
import operator

from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from typing_extensions import TypedDict

from src.settings import AppSettings
from src.llm_embedding_utils import query_embeddings
from src.data_models.retrieve import QueryInput
from src.utils import relative_import


rag_chain = relative_import("chains", file=__file__).rag_chain
eval_chain = relative_import("chains", file=__file__).eval_chain


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
    types_n: list | None
    source_ids_raw: Annotated[list[str], operator.add]
    source_ids: list[str] | None
    source_ids_by_type_raw: Annotated[list[dict], operator.add]
    source_ids_by_type: list[dict] | None
    score: str | None
    label: str | None


# Fallback label used when the caller doesn't supply one.
DEFAULT_LABEL = "default"


# Looks up a per-type override for n_results in the (optional) types_n list.
# Entries may arrive as TypeN models or plain dicts, depending on the caller.
def _resolve_n_results(types_n, type_name: str, default_n: int) -> int:
    if not types_n:
        return default_n
    for entry in types_n:
        entry_type = entry.get("type") if isinstance(entry, dict) else getattr(entry, "type", None)
        entry_n = entry.get("n") if isinstance(entry, dict) else getattr(entry, "n", None)
        if entry_type == type_name and entry_n is not None:
            return entry_n
    return default_n


# Creates a retrieval function for the given input source and maps results to output key
def retrieve_function_generator(query_input: QueryInput, output: str):
    default_n_results = query_input.n_results

    @logger.catch(reraise=True)
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    async def retrieve(state: GraphState):
        logger.info(f"---Retrieving from {query_input.type}---")
        n_results = _resolve_n_results(state.get("types_n"), query_input.type, default_n_results)

        results = await query_embeddings(QueryInput(
            type=query_input.type,
            query_text=state["question"],
            retrieve_fulltext=query_input.retrieve_fulltext,
            n_results=n_results,
            label=state.get("label") or DEFAULT_LABEL,
        ))

        source_ids = [
            result.metadata["source_id"]
            for result in results
            if result.metadata.get("source_id")
        ]

        # Decide what to return: full text or just page content
        if query_input.retrieve_fulltext:
            results = [result.metadata["fulltext"] for result in results]
        else:
            results = [result.page_content for result in results]

        logger.info(results)

        return {
            output: results,
            "source_ids_raw": source_ids,
            "source_ids_by_type_raw": [{"type": query_input.type, "source_ids": source_ids}],
        }

    return retrieve


# Generates a response from retrieved content
@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def generate(state: GraphState):
    logger.info("---Generating---")
    generation = rag_chain.invoke(state)
    return {
        "generation": generation,
        "source_ids": list(dict.fromkeys(state["source_ids_raw"]))
    }


# Scores the generated output (if requested)
@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def evaluate(state: GraphState):
    if "do_scoring" in state and state["do_scoring"]:
        logger.info("---Evaluating---")
        logger.info(state)
        score = eval_chain.invoke(state)
        return {"score": score}


# Groups the per-node source_ids into one entry per collection type, deduplicated.
@logger.catch(reraise=True)
def collect_source_ids(state: GraphState):
    logger.info("---Collecting source ids by type---")
    grouped: dict[str, list[str]] = {}
    for entry in state.get("source_ids_by_type_raw", []):
        grouped.setdefault(entry["type"], [])
        grouped[entry["type"]] += entry["source_ids"]

    return {
        "source_ids_by_type": [
            {"type": type_name, "source_ids": list(dict.fromkeys(ids))}
            for type_name, ids in grouped.items()
        ]
    }


# Retrieval steps shared by both the full RAG graph and the source_ids-only graph
# Do not set n_results to 0!
RETRIEVAL_NODES = [
    ("retrieve_faq", QueryInput(query_text="", type="faqs", retrieve_fulltext=True, n_results=3, label=DEFAULT_LABEL), "faqs"),
    ("retrieve_documentation", QueryInput(query_text="", type="docs", retrieve_fulltext=False, n_results=3, label=DEFAULT_LABEL), "docs"),
    ("retrieve_full_ticket_chunks", QueryInput(query_text="", type="ticket_chunks", retrieve_fulltext=False, n_results=3, label=DEFAULT_LABEL), "ticket_chunks"),
    ("retrieve_ticket_pairs", QueryInput(query_text="", type="ticket_pairs", retrieve_fulltext=True, n_results=2, label=DEFAULT_LABEL), "ticket_pairs"),
]


# Define the full graph-based workflow
workflow = StateGraph(GraphState)

for node_name, query_input, output_key in RETRIEVAL_NODES:
    workflow.add_node(node_name, retrieve_function_generator(query_input, output_key))

# Generation and optional evaluation step
workflow.add_node("generate", generate)
workflow.add_node("evaluate", evaluate)

# Define edges (execution order)
for node_name, _, _ in RETRIEVAL_NODES:
    workflow.add_edge(START, node_name)
    workflow.add_edge(node_name, "generate")

workflow.add_edge("generate", "evaluate")
workflow.add_edge("generate", END)
workflow.add_edge("evaluate", END)

# Compile into executable graph
graph = workflow.compile()


# Retrieval-only workflow: reuses the same retrieval nodes as `graph` but skips
# generation/evaluation, returning source_ids grouped by collection type instead.
source_ids_workflow = StateGraph(GraphState)

for node_name, query_input, output_key in RETRIEVAL_NODES:
    source_ids_workflow.add_node(node_name, retrieve_function_generator(query_input, output_key))

source_ids_workflow.add_node("collect_source_ids", collect_source_ids)

for node_name, _, _ in RETRIEVAL_NODES:
    source_ids_workflow.add_edge(START, node_name)
    source_ids_workflow.add_edge(node_name, "collect_source_ids")

source_ids_workflow.add_edge("collect_source_ids", END)

source_ids_graph = source_ids_workflow.compile()
