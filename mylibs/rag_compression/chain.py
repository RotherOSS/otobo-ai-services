from langchain.llms.ollama import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain.vectorstores.chroma import Chroma

from mylibs.classes.AppSettings import AppSettings
from mylibs.embedding.embedding import get_dbclient
from mylibs.utils.utils import get_content, get_prompt

settings = AppSettings()
client = get_dbclient()

embedding = OllamaEmbeddings(
    base_url=settings.LLM_OLLAMA_URL, model=settings.LLM_OLLAMA_MODEL
)

vectorstore = Chroma(
    collection_name=settings.AI_VECTORSTORE_INDEX, embedding_function=embedding, client=client  # type: ignore
)

# ToDo: Optimize here:
# depending on the chunk bigger k?
# actually only embeddings with type answer are used by the llm
new_settings = settings.rag_search_kwargs.copy()
new_settings["k"] = 20
retriever = vectorstore.as_retriever(search_kwargs=new_settings)


sys_prompt = """Sie sind ein hilfsbereiter, respektvoller und ehrlicher Assistent. Antworten Sie immer auf Deutsch, so hilfreich wie möglich und verwenden Sie den vorgegebenen Kontexttext. Ihre Antworten sollten die Frage nur einmal beantworten und nach der Antwort keinen weiteren Text enthalten.

Wenn eine Frage keinen Sinn ergibt oder sachlich nicht kohärent ist, erklären Sie bitte, warum, anstatt etwas Falsches zu antworten. Wenn Sie die Antwort auf eine Frage nicht wissen, geben Sie bitte keine falschen Informationen weiter. """
instruction = """CONTEXT:/n/n {context}/n
Question: {question}"""
template = get_prompt(instruction, sys_prompt)

from langchain.document_transformers.embeddings_redundant_filter import (
    EmbeddingsRedundantFilter,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
relevant_filter = EmbeddingsFilter(
    embeddings=embedding, similarity_threshold=settings.SIMILARITY_THRESHOLD
)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)


prompt = ChatPromptTemplate.from_template(template)

# LLM
model = Ollama(
    base_url=settings.LLM_OLLAMA_URL,
    model=settings.LLM_OLLAMA_MODEL,
    temperature=settings.LLM_TEMPERATURE,
)

# RAG chain. Hint: wrtitten in LangChain Expression Language (LCEL)
chain = (
    RunnableParallel(
        {
            "context": compression_retriever | RunnableLambda(get_content),
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | model
    | StrOutputParser()
)


# # Add typing for input
# class Question(BaseModel):
#     __root__: str


# chain = chain.with_types(input_type=Question)  # type: ignore
