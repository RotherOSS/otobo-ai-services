from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from mylibs.classes.AppSettings import AppSettings
from mylibs.embedding.embedding import get_vectorstore, get_model
from mylibs.utils.utils import get_content, get_prompt

settings = AppSettings()
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs=settings.rag_search_kwargs)

sys_prompt = """Sie sind ein hilfsbereiter, respektvoller und ehrlicher Assistent. Antworten Sie immer auf Deutsch, so hilfreich wie möglich und verwenden Sie den vorgegebenen Kontexttext. Ihre Antworten sollten die Frage nur einmal beantworten und nach der Antwort keinen weiteren Text enthalten.

Wenn eine Frage keinen Sinn ergibt oder sachlich nicht kohärent ist, erklären Sie bitte, warum, anstatt etwas Falsches zu antworten. Wenn Sie die Antwort auf eine Frage nicht wissen, geben Sie bitte keine falschen Informationen weiter. """
instruction = """CONTEXT:/n/n {context}/n
Question: {question}"""
template = get_prompt(instruction, sys_prompt)

# hwchase17/llama-rag:
# template = """
# [INST] <<SYS>>Beantworte die Frage des Benutzers nur unter Berücksichtigung des folgenden Kontextes. Antworte auf Deutsch. Wenn der Benutzer nach Informationen fragt, die nicht im folgenden Kontext zu finden sind, antworte nicht.

# <context>
# {context}
# </context>
# <</SYS>>

#  {question} [/INST]
# """

prompt = ChatPromptTemplate.from_template(template)
model = get_model()

# RAG chain. Hint: wrtitten in LangChain Expression Language (LCEL)
chain = (
    RunnableParallel(
        {
            "context": retriever | RunnableLambda(get_content),
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
