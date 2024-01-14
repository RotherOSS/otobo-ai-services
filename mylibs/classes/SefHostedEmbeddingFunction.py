from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from chromadb import Documents, EmbeddingFunction, Embeddings
from mylibs.classes.AppSettings import AppSettings

settings = AppSettings()


class HuggingFaceEmbeddingFunction(EmbeddingFunction[Documents]):
    """Helper class needed by Chroma to wrap the embedding into a embedding function

    Args:
        EmbeddingFunction (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __call__(self, input: Documents) -> Embeddings:
        embedding = HuggingFaceBgeEmbeddings(model_name=settings.embedding_model_name)
        # A list is a sequence but a sequence is not necessarily a list. So it's OK
        return embedding.embed_documents(input)  # type: ignore

        # from langchain.embeddings import OllamaEmbeddings
        # embedding = OllamaEmbeddings(
        #     base_url=settings.LLM_OLLAMA_URL, model=settings.LLM_OLLAMA_MODEL
        # )
        # # A list is a sequence but a sequence is not necessarily a list. So it's OK
        # return embedding.embed_documents(input)  # type: ignore


# sh: https://python.langchain.com/docs/integrations/llms/ollama#rag
class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Helper class needed by Chroma to wrap the embedding into a embedding function

    Args:
        EmbeddingFunction (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __call__(self, input: Documents) -> Embeddings:
        embedding = OllamaEmbeddings(
            base_url=settings.LLM_OLLAMA_URL, model=settings.LLM_OLLAMA_MODEL
        )
        # A list is a sequence but a sequence is not necessarily a list. So it's OK
        return embedding.embed_documents(input)  # type: ignore
