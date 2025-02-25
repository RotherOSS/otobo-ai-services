from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from loguru import logger
from mylibs.classes.AppSettings import AppSettings
from mylibs.embedding.embedding import get_model

settings = AppSettings()
llm = get_model()
json_llm = get_model(use_ollama_json_format=True)


# LLMs
######

### Retrieval Grader

# prompt = PromptTemplate(
#     template="""You are a grader assessing relevance of a retrieved document to a user question. \n
#     Here is the retrieved document: \n\n {document} \n\n
#     Here is the user question: {question} \n
#     If the document contains keywords related to the user question, grade it as relevant. \n
#     It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
#     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
#     Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
#     input_variables=["question", "document"],
# )

retrieval_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)
retrieval_grader = retrieval_grader_prompt | json_llm | JsonOutputParser()


### Generate
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


# Prompt

# rag_chain_prompt = PromptTemplate(
#     template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
#     Use three sentences maximum and keep the answer concise. \n
#     Question: {question}  \n
#     Context: {context}  \n
#     Answer: """,
#     input_variables=["context", "question"],
# )

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
    input_variables=["question", "document"],
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


### Hallucination Grader
hallucination_grader_prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
hallucination_grader = hallucination_grader_prompt | json_llm | JsonOutputParser()


### Answer Grader
answer_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
answer_grader = answer_grader_prompt | json_llm | JsonOutputParser()


### Question Re-writer
re_write_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Du bist ein Frage-Umformulierer, der eine Eingabefrage in eine bessere Version umwandelt, die
     für die Abfrage im Vektorspeicher optimiert ist. Schau dir die ursprüngliche Frage an und formuliere eine bessere Version. Gib ausschliesslich die umformulierte Frage aus.
     Antworte auf deutsch.\n
     <|eot_id|><|start_header_id|>user<|end_header_id|> Hier ist die Eingabefrage: \n {question}.\n
     Umformulierte Frage: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
question_rewriter = re_write_prompt | llm | StrOutputParser()


def create_chain(promptTemplate: PromptTemplate, json_output: bool = False):
    if json_output:
        return promptTemplate | json_llm | JsonOutputParser()
    else:
        return promptTemplate | llm | StrOutputParser()
