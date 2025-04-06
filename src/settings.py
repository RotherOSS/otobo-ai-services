import os
from dotenv import load_dotenv


class AppSettings:
    """
    `AppSetting` class, the one and only source to get your environment variables and predefined parameters.

    Will call os.getenv(), not (!) static, create an instance first.
    ### Example

    ```python
    from src.settings import AppSettings

    settings = AppSettings()

    key = settings.API_KEY
    ```
    """

    def __init__(self):
        load_dotenv()
        self.OTOBO_AI_API_KEY = os.getenv("OTOBO_AI_API_KEY")
        self.OTOBO_AI_HOST = os.getenv("OTOBO_AI_HOST", "0.0.0.0")
        self.OTOBO_AI_PORT = int(os.getenv("OTOBO_AI_PORT", "8080"))

        self.OTOBO_AI_CHROMA_DIR = os.getenv("OTOBO_AI_CHROMA_DIR", "/chroma")
        self.OTOBO_AI_CHROMA_DEF_COL_NAME = "documentation"

        self.OTOBO_AI_LLM_API_KEY = os.getenv("OTOBO_AI_LLM_API_KEY", None)
        self.OTOBO_AI_LLM_HOST = os.getenv("OTOBO_AI_LLM_HOST", "http://localhost:11434")
        self.OTOBO_AI_LLM_MODEL = os.getenv("OTOBO_AI_LLM_MODEL", "llama3.2")
        self.OTOBO_AI_EMBEDDING_MODEL = os.getenv(
            "OTOBO_AI_EMBEDDING_MODEL", "bge-m3"
        )

        self.OTOBO_AI_LLM_TEMPERATURE = float(os.getenv("OTOBO_AI_LLM_TEMPERATURE", "0.1"))

        self.fastapi_title = "OTOBO AI"
        self.fastapi_version = "1.0"
        self.fastapi_description = (
            "API server using RAG to recommend ticket answers"
        )

        self.OTOBO_AI_LANGFUSE_PK = os.getenv("LANGFUSE_PUBLIC_KEY", None)
        self.OTOBO_AI_LANGFUSE_SK = os.getenv("LANGFUSE_SECRET_KEY", None)
        self.OTOBO_AI_LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

        self.OTOBO_AI_LOG_FILE = os.getenv("OTOBO_AI_LOG_FILE", "./data/log/apilog.log")
        self.OTOBO_AI_LOG_LEVEL = os.getenv("OTOBO_AI_LOG_LEVEL", "DEBUG")
        self.OTOBO_AI_PG_HOST = os.getenv("OTOBO_AI_PG_HOST", "postgres")
        self.OTOBO_AI_PG_PORT = os.getenv("OTOBO_AI_PG_PORT", "5432")
        self.OTOBO_AI_PG_DB = os.getenv("OTOBO_AI_PG_DB", "fulltext")
        self.OTOBO_AI_PG_USER = os.getenv("OTOBO_AI_PG_USER", "otobo_ai")
        self.OTOBO_AI_PG_PW = os.getenv("OTOBO_AI_PG_PW")
        self.OTOBO_AI_PG_DSN = (f"postgresql://{self.OTOBO_AI_PG_USER}:{self.OTOBO_AI_PG_PW}"
                                f"@{self.OTOBO_AI_PG_HOST}:{self.OTOBO_AI_PG_PORT}/{self.OTOBO_AI_PG_DB}")

        self.embedding_chunk_size = 1100
        self.embedding_chunk_overlap = 100

        self.check_envs()

    def check_envs(self):
        if os.getenv("OTOBO_AI_API_KEY") is None:
            raise ValueError("Enviroment variable 'OTOBO_AI_API_KEY' must be set.")
