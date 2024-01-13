from langchain.llms.together import Together
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.vectorstores.chroma import Chroma

# from langchain.embeddings import HuggingFaceBgeEmbeddings
from mylibs.classes.AppSettings import AppSettings
from mylibs.embedding.embedding import get_dbclient
from mylibs.utils.utils import get_content, get_prompt
from langchain.embeddings import OllamaEmbeddings

settings = AppSettings()

client = get_dbclient()

if settings.use_huggingface:
    from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

    embedding = HuggingFaceInferenceAPIEmbeddings(
        api_key=settings.HUGGINGFACEHUB_API_TOKEN,
        model_name=settings.embedding_model_name,
    )
else:
    # Self hosted embedding model (+2GB ram)
    # embedding = HuggingFaceBgeEmbeddings(model_name=settings.embedding_model_name)

    embedding = OllamaEmbeddings(
        base_url=settings.LLM_OLLAMA_URL, model=settings.LLM_OLLAMA_MODEL
    )


vectorstore = Chroma(
    collection_name=settings.AI_VECTORSTORE_INDEX, embedding_function=embedding, client=client  # type: ignore
)

# ToDo: Optimize here:
# depending on the chunk bigger k?
# actually only embeddings with type answer are used by the llm
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


# LLM
if settings.use_together:
    model = Together(
        model=settings.TOGETHERAI_MODEL,  # type: ignore
        together_api_key=settings.TOGETHERAI_API_KEY,  # type: ignore
        max_tokens=2048,
        temperature=settings.LLM_TEMPERATURE,
    )
else:
    # Ollama
    from langchain.llms.ollama import Ollama

    model = Ollama(
        base_url=settings.LLM_OLLAMA_URL,
        model=settings.LLM_OLLAMA_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )


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
