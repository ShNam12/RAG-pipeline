"""Module xử lý vector hóa câu hỏi của người dùng."""

from __future__ import annotations

import threading
from typing import Optional
from sentence_transformers import SentenceTransformer
from rag1.retrival.config import settings


class QueryEmbedder:
    """Lớp quản lý mô hình embedding (áp dụng Singleton để tối ưu tài nguyên)."""

    _instance: Optional[QueryEmbedder] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Khởi tạo và nạp mô hình embedding từ HuggingFace/Local."""
        if not hasattr(self, "_initialized"):
            # Nạp mô hình sentence-transformers đã cấu hình trong .env
            self.model: SentenceTransformer = SentenceTransformer(
                settings.embedding_model
            )
            self._initialized: bool = True

    @classmethod
    def get_instance(cls) -> QueryEmbedder:
        """Lấy thể hiện duy nhất của QueryEmbedder (Thread-safe)."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def encode_query(self, query: str) -> list[float]:
        """Chuyển đổi một chuỗi câu hỏi thành vector biểu diễn ngữ nghĩa.

        Tham số:
            query: Nội dung câu hỏi từ người dùng.

        Trả về:
            Danh sách số thực (list[float]) biểu diễn vector ngữ nghĩa.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Câu hỏi không được để trống khi tạo vector.")

        # Mã hóa câu hỏi thành numpy array sau đó chuyển sang dạng danh sách chuẩn
        vector = self.model.encode(cleaned_query)
        return vector.tolist()
