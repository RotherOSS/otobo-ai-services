from typing import List

from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from typing_extensions import Literal, TypedDict
from mylibs.embedding.embedding import get_vectorstore
from mylibs.core.chains import rag_chain

# from kilibrary.chains.core.chains import rag_chain
# from kilibrary.utils.retrieverutils import get_retriever
from mylibs.classes.AppSettings import AppSettings

settings = AppSettings()


### State
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

    # retriever = get_retriever(collection_name=state.get("collection_name", None))
    # documents = retriever.invoke(question)
    documents = retriever.invoke(
        question, config={"configurable": {"search_kwargs": {"k": 8}}}
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

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        # "documents": documents,
        "generation": generation,
    }


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()
"""this is the normal RAG graph
"""


@logger.catch(reraise=True, exclude=GraphInterrupt)
def human_review_node(state: GraphState) -> Command[Literal["generate", "retrieve", END]]:  # type: ignore
    """
    Human review node to decide whether to call LLM or stop
    """
    logger.info("---HUMAN REVIEW---")

    documents = state["documents"]
    human_review = interrupt(
        {
            # "question": "Passen die Dokumente? Antworte mit 'ok', 'update' oder 'stop'.",
            "question": "Passen die Dokumente? Antworte mit 'ok', 'stop' oder gib die Dokumente an, die du aus der Liste entfernen möchtest (z.B. '1, 3, 4').",
            # "documents": list(
            #     set([doc.metadata.get("filename", "") for doc in documents])
            # ),
            "documents": list(
                f"{doc.metadata.get('filename', '')}{", " + str(doc.metadata.get('page')) if doc.metadata.get('page') else ''}"
                for doc in documents
            ),
        }
    )

    review_action = human_review["action"].lower()
    review_data = human_review.get("data", {})

    if review_action == "ok":
        return Command(goto="generate")
    elif review_action == "remove_docs" and review_data:
        try:
            inputs = review_data.split(",")
            indices = [int(i.strip()) - 1 for i in inputs]
            documents = [doc for i, doc in enumerate(documents) if i not in indices]

            if not documents:
                return Command(goto=END)

            return Command(goto="generate", update={"documents": documents})
        except Exception as e:
            return Command(goto=END)
    elif review_action == "stop":
        return Command(goto=END)
    else:
        return Command(goto=END)
        # raise ValueError(f"Unknown action: {review_action}")


from langgraph.checkpoint.memory import MemorySaver

hre_workflow = StateGraph(GraphState)

hre_workflow.add_node("retrieve", retrieve)
hre_workflow.add_node("human_review", human_review_node)
hre_workflow.add_node("generate", generate)

hre_workflow.set_entry_point("retrieve")
hre_workflow.add_edge("retrieve", "human_review")
# now dynamically created: add_edge("human_review", "generate")
hre_workflow.add_edge("generate", END)

hre_graph = hre_workflow.compile(checkpointer=MemorySaver())
"""this is the RAG graph with human review
"""
