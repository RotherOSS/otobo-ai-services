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
json_llm = get_model(use_ollama_json_format=True)


def log_step(inputs):
    # Combine inputs into the final prompt using the prompt template
    logger.debug(f"Step state: {inputs}")
    return inputs


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


prompt_path = Path(__file__).parent / "prompts" / "rag_prompt.txt"
prompt_template = prompt_path.read_text(encoding="utf-8")

rag_chain_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "faqs", "docs", "ticket_chunks", "ticket_pairs"],
)


def log_final_prompt(inputs):
    # Combine inputs into the final prompt using the prompt template
    final_prompt = rag_chain_prompt.format(**inputs)
    logger.debug(f"Final Prompt: {final_prompt}")
    return inputs


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
    | log_final_prompt
    | rag_chain_prompt
    | llm
    | StrOutputParser()
)


class EvalOutput(BaseModel):
    reasoning: str
    faithfulness: int
    completeness: int
    friendliness: int
    solved: bool


prompt_path = Path(__file__).parent / "prompts" / "eval_prompt.txt"
prompt_template = prompt_path.read_text(encoding="utf-8")


eval_chain_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "faqs", "docs", "ticket_chunks", "ticket_pairs", "generation"],
)


def log_eval_prompt(inputs):
    # Combine inputs into the final prompt using the prompt template
    eval_prompt = eval_chain_prompt.format(**inputs)
    logger.debug(f"Final Prompt: {eval_prompt}")
    return inputs


def structure_output(llm_output):
    validated_response = None
    try:
        response_data = json.loads(llm_output)
        # Validate with Pydantic
        validated_response = EvalOutput(**response_data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid format: {e}")
    return validated_response


def combine_score(validated_response):
    if not validated_response or not validated_response.solved:
        return 0
    else:
        return (
                validated_response.faithfulness +
                validated_response.completeness +
                validated_response.friendliness
        ) / 4 / 3


eval_chain = (
    log_eval_prompt
    | eval_chain_prompt
    | llm
    | StrOutputParser()
    | log_step
    | structure_output
    | combine_score
)
