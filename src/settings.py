import os
from dotenv import load_dotenv


class AppSettings:
    """
    `AppSetting` class, the one and only source to get your enviroment variables and predefined parameters.

    Will call os.getenv(), not (!) static, create an instance first.
    ### Example

    ```python
    from src.data_models.AppSettings import AppSettings

    settings = AppSettings()

    key = settings.API_KEY
    ```
    """

    true_values = [
        "true",
        "1",
        "t",
        "y",
        "yes",
    ]

    def __init__(self):
        load_dotenv()  # load enviroment variables once
        self.AI_API_KEY = os.getenv("AI_API_KEY")
        self.AI_API_SERVER_HOST = os.getenv("AI_API_SERVER_HOST", "0.0.0.0")
        self.AI_API_SERVER_PORT = int(os.getenv("AI_API_SERVER_PORT", "8080"))

        self.CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "/chroma")

        self.AI_VECTORDB_HOST = os.getenv("AI_VECTORDB_HOST", "http://localhost")
        self.AI_VECTORDB_PORT = os.getenv("AI_VECTORDB_PORT", "9200")
        self.AI_VECTORDB_AUTH_TOKEN = os.getenv("AI_VECTORDB_AUTH_TOKEN", None)
        self.AI_VECTORSTORE_INDEX = os.getenv("AI_VECTORSTORE_INDEX", "documents")
        self.es_url = f"{self.AI_VECTORDB_HOST}:{self.AI_VECTORDB_PORT}"

        self.LLM_OTOBO_API_KEY = os.getenv("LLM_OTOBO_API_KEY", None)
        self.LLM_OLLAMA_URL = os.getenv("LLM_OLLAMA_URL", "http://localhost:11434")
        self.LLM_OLLAMA_MODEL = os.getenv("LLM_OLLAMA_MODEL", "llama3.2")
        self.LLM_OLLAMA_EMBEDDING_MODEL = os.getenv(
            "LLM_OLLAMA_EMBEDDING_MODEL", "bge-m3"
        )

        self.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
        self.REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", None)

        # self.TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY", None)
        # self.TOGETHERAI_MODEL = os.getenv("TOGETHERAI_MODEL", None)
        # self.use_localembedding = False
        # self.use_together = (
        #     os.getenv("USE_TOGETHER", "False").lower() in self.true_values
        # )
        # self.use_chromadb = "False"
        # self.use_chromadb = (
        #     os.getenv("USE_CHROMADB", "False").lower() in self.true_values
        # )

        self.fastapi_title = "Ticket Answering Service"
        self.fastapi_version = "1.0"
        self.fastapi_description = (
            "API server using RAG to answer questions based on tickets"
        )

        self.LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", None)
        self.LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", None)
        self.LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

        self.LOG_FILE = os.getenv("LOG_FILE", "./data/log/apilog.log")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

        self.embedding_chunk_size = 1100
        self.embedding_chunk_overlap = 100
        # TODO: not working yet
        self.rag_search_kwargs = (
            {
                "k": int(os.getenv("RAG_K", "3")),
                "filter": {
                    "bool": {
                        "must": [
                            {"term": {"metadata.type": os.getenv("RAG_FILTER", "")}}
                        ]
                    }
                },
            }
            if os.getenv("RAG_FILTER", "") != ""
            else {"k": int(os.getenv("RAG_K", "3"))}
        )

        self.check_envs()

    def check_envs(self):
        """checks if all necesarry varables are set"""
        if os.getenv("AI_API_KEY") is None:
            raise ValueError("Enviroment variable 'AI_API_KEY' must be set.")

    def getenv(self, key: str, default=None):
        return os.getenv(key, default)
