from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from src.settings import AppSettings
from src.llm_embedding_utils import get_model
from pathlib import Path
from pydantic import BaseModel
import json

settings = AppSettings()
llm = get_model()
llm_eval = get_model(eval=True)
json_llm = get_model(use_ollama_json_format=True)


# Creates a formatter function to turn lists of text into a single block
def context_formatting_func_generator(key):
    def inner(dictonary_docs):
        texts = dictonary_docs.get(key, [])
        if not texts:
            return ""
        try:
            return "\n\n-----\n\n".join(texts)
        except Exception as e:
            logger.error(f"Error formatting context for '{key}': {e}")
            return dictonary_docs
    return inner


# Extracts the question from the shared state
def get_question(dict_in):
    return dict_in["question"]


# Load RAG prompt from file
prompt_path = Path(__file__).parent / "prompts" / "rag_prompt.txt"
prompt_template = prompt_path.read_text(encoding="utf-8")

rag_chain_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "faqs", "docs", "ticket_chunks", "ticket_pairs"],
)

# Defines the main generation chain:
# 1. Formats context from retrieved sources
# 2. Passes it into the prompt
# 3. Sends to model
# 4. Parses text output
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


# Defines expected structure of evaluation output
class EvalOutput(BaseModel):
    reasoning: str
    faithfulness: int
    completeness: int
    friendliness: int
    solved: bool


# Load evaluation prompt
prompt_path = Path(__file__).parent / "prompts" / "eval_prompt.txt"
prompt_template = prompt_path.read_text(encoding="utf-8")

eval_chain_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "faqs", "docs", "ticket_chunks", "ticket_pairs", "generation"],
)


# Passes model output as string back to invoker
def structure_output(llm_output):
    return llm_output


# Combines numeric scores from evaluation (if answer is marked as solved)
# Note: this is currently disabled in favour of just passing the score to invoker
def combine_score(validated_response):
    if not validated_response or not validated_response.solved:
        return 0
    return (
        validated_response.faithfulness +
        validated_response.completeness +
        validated_response.friendliness
    ) / 4 / 3


# Full evaluation chain: prompt → model → JSON validation → score calculation
eval_chain = (
    eval_chain_prompt
    | llm_eval
    | StrOutputParser()
    | structure_output
#    | combine_score                 # Note: currently disabled
)
