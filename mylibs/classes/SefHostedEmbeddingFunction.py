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
        # A list is a sequence but a sequence is not necessarily a list. So it's OK
        return embedding.embed_documents(input)  # type: ignore
