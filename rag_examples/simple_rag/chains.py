from langchain.prompts import PromptTemplate
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
    texts = dictonary_docs.get("docs", [])
    if not texts:
        return ""
    try:
        return "\n\n-----\n\n".join(texts)
    except Exception as e:
        logger.error(f"Error in format_document_context: {e}")
        return dictonary_docs


def get_question(json_in):
    return json_in["question"]


prompt_path = Path(__file__).parent / "prompts" / "prompt.txt"
prompt_template = prompt_path.read_text(encoding="utf-8")

rag_chain_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "docs"],
)


rag_chain = (
    RunnableParallel(
        {
            "docs": RunnableLambda(format_document_context),
            "question": RunnableLambda(get_question),
        }
    )
    | rag_chain_prompt
    | llm
    | StrOutputParser()
)
