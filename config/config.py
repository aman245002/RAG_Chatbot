# config/config.py
import os
from dotenv import load_dotenv

# Only load .env in local dev. In Streamlit Cloud, use secrets/env vars.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

class Config:
    # API keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")

    # Options
    USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() in ("1", "true", "yes")

    # Vectorstore + chunk settings
    VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

    # LLM defaults
    DEFAULT_LLM = os.getenv("DEFAULT_LLM", "openai")  # options: openai, groq, other
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change as needed
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

config = Config()
