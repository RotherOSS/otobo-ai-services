from langchain.embeddings import HuggingFaceBgeEmbeddings
from chromadb import Documents, EmbeddingFunction, Embeddings
from mylibs.classes.AppSettings import AppSettings

settings = AppSettings()


class SefHostedEmbeddingFunction(EmbeddingFunction[Documents]):
    """Helper class needed by Chroma to wrap the embedding into a embedding function

    Args:
        EmbeddingFunction (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __call__(self, input: Documents) -> Embeddings:
        embedding = HuggingFaceBgeEmbeddings(model_name=settings.embedding_model_name)

        # from langchain.embeddings import OllamaEmbeddings
        # sh: https://python.langchain.com/docs/integrations/llms/ollama
        # embedding = OllamaEmbeddings(base_url="http://host:port", model="llama2")

        # A list is a sequence but a sequence is not necessarily a list. So it's OK
        return embedding.embed_documents(input)  # type: ignore
