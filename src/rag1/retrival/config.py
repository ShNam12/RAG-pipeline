"""Configuration module for RAG retrieval and generation pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Tìm và nạp file .env từ thư mục gốc dự án
ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """System settings for retrieval and generation components."""

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    text_collection: str = os.getenv("TEXT_COLLECTION_NAME", "text_index")
    table_collection: str = os.getenv("TABLE_COLLECTION_NAME", "table_index")

    # Embeddings
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "thanhtantran/Vietnamese_Embedding"
    )

    # Docker LLM
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "ollama")
    llm_model: str = os.getenv(
        "LLM_MODEL_NAME", "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
    )

    # Retrieval parameters
    text_top_k: int = 15
    table_top_k: int = 10
    rrf_k: int = 60
    final_top_k: int = 5


settings = Settings()
