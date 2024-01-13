def get_content(retriever_docs):
    """Filters the result of a retriever

    Args:
        retriever_docs (List[Document]): result of retriever

    Returns:
        List[str]: List of result text
    """
    docs = []
    for doc in retriever_docs:
        content = doc.dict()["page_content"]
        docs.append(content)
    return docs


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
