"""Data schemas for retrieval, fusion, reranking and generation."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Type of retrieved knowledge chunk."""

    TEXT = "text"
    TABLE = "table"


class SearchCandidate(BaseModel):
    """Represents a retrieved item from Qdrant vector search."""

    id: str | int
    score: float = 0.0
    source_type: SourceType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    rank: int = 0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class Citation(BaseModel):
    """Citation metadata attached to generated answers."""

    index: int
    source_type: SourceType
    doc_name: str
    page_number: int | str
    section_or_caption: str


class RAGResponse(BaseModel):
    """Final output response returned to user."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    candidates: list[SearchCandidate] = Field(default_factory=list)

class QueryRequest(BaseModel):
    """Khuôn mẫu dữ liệu gửi lên từ người dùng."""
    query: str = Field(..., min_length=1, description="Câu hỏi của người dùng")

class ChatResponse(BaseModel):
    """Khuôn mẫu dữ liệu kết quả trả về cho Frontend (Dùng cho non-streaming nếu cần)."""
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
