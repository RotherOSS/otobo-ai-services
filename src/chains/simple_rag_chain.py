from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from src.settings import AppSettings
from src.llm_embedding_utils import get_model
from pathlib import Path

settings = AppSettings()
llm = get_model()
json_llm = get_model(use_ollama_json_format=True)


def format_document_context(dictonary_docs):
    docs = dictonary_docs.get("context", [])
    if not docs:
        return ""
    try:
        if type(docs[0]) == str:
            rv = "\n------\n".join(docs)
        elif type(docs[0]) == Document:
            rv = "\n----\n".join(doc.page_content for doc in docs)
        else:
            rv = "\n---\n".join(doc["page_content"] for doc in docs)
    except Exception as e:
        logger.error(
            f"Fehler in format_document_context, docs wird unverändert zurückgegeben: {e}"
        )
        rv = dictonary_docs
    return rv


def get_question(json_in):
    return json_in["question"]


prompt_path = Path(__file__).parent.parent / "prompts" / "simple_rag_prompt.txt"
prompt_template = prompt_path.read_text(encoding="utf-8")

rag_chain_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "context"],
)


rag_chain = (
    RunnableParallel(
        {
            "context": RunnableLambda(format_document_context),
            "question": RunnableLambda(get_question),
        }
    )
    | rag_chain_prompt
    | llm
    | StrOutputParser()
)
