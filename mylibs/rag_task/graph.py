from typing import Annotated, NotRequired, Sequence, TypedDict

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langgraph.graph import END, StateGraph
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from typing_extensions import TypedDict

from mylibs.classes.AppSettings import AppSettings
from mylibs.core.chains import rag_chain
from mylibs.embedding.embedding import get_embeddingsmodel, get_vectorstore
from mylibs.rag_task.TaskRetriever import TaskRetriever

settings = AppSettings()
k_retriever = 10
start_threshold = 0.7
min_threshold = 0.5


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information."""


def get_prompt(instruction: str, new_system_prompt=DEFAULT_SYSTEM_PROMPT) -> str:
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


template = get_prompt(
    instruction="""CONTEXT:/n/n {context}/n Question: {question}""",
    new_system_prompt="""
      Sie sind ein hilfsbereiter, respektvoller und ehrlicher Assistent. Antworten Sie immer auf Deutsch, so hilfreich wie möglich und verwenden Sie den vorgegebenen Kontexttext. Ihre Antworten sollten die Frage nur einmal beantworten und nach der Antwort keinen weiteren Text enthalten.
      Wenn eine Frage keinen Sinn ergibt oder sachlich nicht kohärent ist, erklären Sie bitte, warum, anstatt etwas Falsches zu antworten. Wenn Sie die Antwort auf eine Frage nicht wissen, geben Sie bitte keine falschen Informationen weiter.
      """,
)


def get_last_ai_message(messages: Sequence[AnyMessage]) -> AIMessage | None:
    if messages is None:
        return None
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None


def _get_content(docs):
    page_contents = [doc.dict()["page_content"] for doc in docs]
    return page_contents


@logger.catch(reraise=True)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def retrieve(state):
    logger.info("---ABRUFEN---")
    question = state["question"]

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs=settings.rag_search_kwargs)

    documents = retriever.invoke(question)
    return {"documents": documents}


@logger.catch(reraise=True)
# @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def compress(state):
    logger.info("---COMPRESS---")
    question = state["question"]
    message = get_last_ai_message(state["messages"])

    similarity_threshold = (
        message.response_metadata["similarity_threshold"] - 0.1
        if message
        else start_threshold
    )
    logger.info(f"similarity_threshold: {similarity_threshold}")

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_retriever})
    task_retriever = TaskRetriever(vectorstore=retriever)

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=get_embeddingsmodel())
    relevant_filter = EmbeddingsFilter(
        embeddings=get_embeddingsmodel(), similarity_threshold=similarity_threshold
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=task_retriever,
    )

    docs = compression_retriever.invoke(question)
    compressions = _get_content(docs)

    out_message = AIMessage(
        name="compress",
        content=compressions,
        response_metadata={"similarity_threshold": similarity_threshold},
    )

    return {"messages": [out_message]}


@logger.catch
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def generate(state):
    logger.info("---GENERIEREN---")

    message = get_last_ai_message(state["messages"])
    question = state["question"]

    generation = rag_chain.invoke({"context": message.content, "question": question})
    out_message = AIMessage(
        name="generate",
        content=generation,
    )

    return {"messages": [out_message], "generation": generation}


def generate_err_answer(state):
    logger.info("---GENERIERE ERR_ANTWORT---")
    generation = "Es konnte leider keine relevante Antwort gefunden werden."
    out_message = AIMessage(
        name="generate_err_answer",
        content=generation,
    )

    return {"messages": [out_message], "generation": generation}


### Edges
#########
def decide_to_generate(state):
    message = get_last_ai_message(state["messages"])

    compressions = message.content  # state["compressions"]
    similarity_threshold = message.response_metadata[
        "similarity_threshold"
    ]  # state["similarity_threshold"]

    if len(compressions) > 0:
        logger.info(f"{len(compressions)} Dokumente -> Generieren")
        return "yes"
    else:
        if similarity_threshold <= min_threshold:
            logger.info(
                f"Keine relevanten Dokumente gefunden (Threshold {similarity_threshold})."
            )
            return "threshold_exeeded"
        else:
            logger.info(f"Keine Dokumente -> Neuer Abruf mit weniger Threshold")
            return "no"


### State


def reduce_list(left: list | None, right: list | None) -> list:
    if not left:
        left = []
    if not right:
        right = []
    return left + right


class GraphState(TypedDict):
    """
    Status des Graphen.

    Attribute:
        question: question
    """

    question: str
    generation: NotRequired[str]
    messages: Annotated[NotRequired[Sequence[BaseMessage]], reduce_list]


workflow = StateGraph(GraphState)

workflow.add_node("compress", compress)
workflow.add_node("generate", generate)
workflow.add_node("generate_err_answer", generate_err_answer)

workflow.set_entry_point("compress")
workflow.add_conditional_edges(
    "compress",
    decide_to_generate,
    {"yes": "generate", "no": "compress", "threshold_exeeded": "generate_err_answer"},
)
workflow.add_edge("generate", END)
workflow.add_edge("generate_err_answer", END)
graph = workflow.compile()
