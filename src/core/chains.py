from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from loguru import logger
from src.settings import AppSettings
from src.embedding.embedding import get_model

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


rag_chain_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Du bist ein Assistent für Frage-Antwort Aufgaben.
    Beantworte die folgende FRAGE ausschließlich mit dem unten aufgeführten Kontext. Wenn du die Antwort nicht weißt, antworte mit 'Das weiß ich leider nicht'.
    Antworte immer auf Deutsch. Antworte ausführlich und sehr gewissenhaft. <|eot_id|><|start_header_id|>user<|end_header_id|>
    \n ------- \n
    FRAGE: {question}
    \n ------- \n
    Kontext: {context}
    \n ------- \n
    Antwort: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
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
