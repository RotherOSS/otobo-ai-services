from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from src.settings import AppSettings
from src.llm_embedding_utils import get_model
from pathlib import Path

settings = AppSettings()
llm = get_model()


def format_document_context(dictonary_docs):
    """
    Joins retrieved document strings into a readable text block,
    separated by a divider. If empty or error, returns original dict.
    """
    texts = dictonary_docs.get("docs", [])
    if not texts:
        return ""
    try:
        return "\n\n-----\n\n".join(texts)
    except Exception as e:
        logger.error(f"Error in format_document_context: {e}")
        return dictonary_docs


def get_question(json_in):
    """Extract the question string from the graph state input."""
    return json_in["question"]


# Load custom prompt template from disk
prompt_path = Path(__file__).parent / "prompts" / "prompt.txt"
prompt_template = prompt_path.read_text(encoding="utf-8")

rag_chain_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "docs"],
)

# Compose the RAG chain:
# 1. Parallel: extract docs and question from state
# 2. Format prompt
# 3. Run LLM
# 4. Parse output to string
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
