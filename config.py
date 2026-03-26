"""
Configuration settings for AI Services
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    FAL_KEY: str = os.getenv("FAL_KEY", "")
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-flash")

    # Embedding model for RAG
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"


config = Config()
