import os
from dotenv import load_dotenv
from typing_extensions import Annotated, Doc


class AppSettings:
    """
    `AppSetting` class, the one and only source to get your enviroment variables and predefined parameters.

    Will call os.getenv(), not (!) static, create an instance first.
    ### Example

    ```python
    from mylibs.classes.AppSettings import AppSettings

    settings = AppSettings()

    key = settings.API_KEY
    ```
    """

    def __init__(self):
        load_dotenv()  # load enviroment variables once
        self.API_KEY: Annotated[
            str,
            Doc(
                """
                Authentication token for API call.
                To call this API with an client you have to put it in your request header
                ## Exapmle
                ```Python
                import os
                import requests
                api_url = "http://127.0.0.1:8000/embedding/query/"

                headers = {
                'Content-Type': 'application/json',
                'access_token': os.getenv("API_KEY")
                }

                query_texts = [
                "Wann war der Burenkrieg in Südafrika?",
                "Was kann den Verschleiß des seillosen Aufzuges minimieren?",
                "Was führte zur Entwicklung des ersten Tuberkulose-Testes?",
                "Wieso forderte Bismarck die Annexion von Sachsen-Meiningen und Reuß nach dem Krieg von 1866?",
                "Wie hat man am Ende des 19. Jahrhundert in Großbritannien versucht, die Tuberkulose zu bekämpfen?"
                ]


                for query_text in query_texts:
                response = requests.post(api_url, headers=headers, json={
                    "query_texts": [query_text],
                    "where": {"type": "answer"},
                    "include": [
                    "metadatas",
                    "documents"
                    ]
                })
                print(response.json())
                ```
                """
            ),
        ] = os.getenv("API_KEY")
        self.SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
        self.SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))
        self.CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
        self.CHROMADB_PORT = os.getenv("CHROMADB_PORT", "8000")
        self.CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY")
        self.CHROMADB_COLLECTION = os.getenv("CHROMADB_COLLECTION", "documents")
        self.LLL_OLLAMA_URL = os.getenv(
            "LLL_OLLAMA_URL", "http://localhost:11434"
        )  # ToDo default value???
        self.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
        self.HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        self.use_huggingface = os.getenv("USE_HUGGINGFACE", "False").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]
        self.use_together = os.getenv("USE_TOGETHER", "False").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]
        # self. = os.getenv('', '')

        self.fastapi_title = "Ticket Answering Service"
        self.fastapi_version = "1.0"
        self.fastapi_description = (
            "API server using RAG to answer questions based on tickets"
        )

        # had bad results with BAAI/bge-base-en-v1.5
        # WARNING: changing the model will change the vector dimension
        # so you must rebuild the complete embedding
        self.embedding_model_name = "BAAI/bge-large-en-v1.5"
        self.embedding_chunk_size = 1100
        self.embedding_chunk_overlap = 100
        self.rag_search_kwargs = (
            {
                "k": int(os.getenv("RAG_K", "3")),
                "filter": {"type": os.getenv("RAG_FILTER", "")},
            }
            if os.getenv("RAG_FILTER", "") != ""
            else {"k": int(os.getenv("RAG_K", "3"))}
        )

    def getenv(self, key: str, default=None):
        return os.getenv(key, default)
