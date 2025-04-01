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


def context_formatting_func_generator(key):
    def inner(dictonary_docs):
        texts = dictonary_docs.get(key, [])
        if not texts:
            return ""
        try:
            return "\n\n-----\n\n".join(texts)
        except Exception as e:
            logger.error(f"Error in format_document_context for key '{key}': {e}")
            return dictonary_docs
    return inner


def get_question(dict_in):
    return dict_in["question"]


prompt_path = Path(__file__).parent / "prompts" / "prompt.txt"
prompt_template = prompt_path.read_text(encoding="utf-8")

rag_chain_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "faqs", "docs", "ticket_chunks", "ticket_pairs"],
)


rag_chain = (
    RunnableParallel(
        {
            "faqs": RunnableLambda(context_formatting_func_generator("faqs")),
            "docs": RunnableLambda(context_formatting_func_generator("docs")),
            "ticket_chunks": RunnableLambda(context_formatting_func_generator("ticket_chunks")),
            "ticket_pairs": RunnableLambda(context_formatting_func_generator("ticket_pairs")),
            "question": RunnableLambda(get_question),
        }
    )
    | rag_chain_prompt
    | llm
    | StrOutputParser()
)
