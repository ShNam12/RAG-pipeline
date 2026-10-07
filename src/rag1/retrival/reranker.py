"""Module tái xếp hạng (Reranking) các đoạn văn bản và bảng biểu theo câu hỏi."""

from __future__ import annotations

import logging
from typing import Optional
from sentence_transformers import CrossEncoder

from rag1.retrival.config import settings
from rag1.retrival.schemas import SearchCandidate

logger = logging.getLogger(__name__)


class Reranker:
    """Lớp tái xếp hạng ứng viên dựa trên mô hình Cross-Encoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        """Khởi tạo mô hình Reranker."""
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> CrossEncoder:
        """Nạp mô hình CrossEncoder (Lazy Loading)."""
        if self._model is None:
            logger.info("Đang nạp mô hình Reranker: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[SearchCandidate],
        top_k: int | None = None,
    ) -> list[SearchCandidate]:
        """Tái xếp hạng danh sách ứng viên theo mức độ phù hợp với câu hỏi.

        Tham số:
            query: Câu hỏi của người dùng.
            candidates: Danh sách ứng viên từ bước RRF.
            top_k: Số lượng kết quả tốt nhất giữ lại (mặc định lấy theo settings).

        Trả về:
            Danh sách top_k ứng viên có điểm rerank cao nhất.
        """
        if not candidates:
            return []

        limit = top_k or settings.final_top_k
        if len(candidates) <= limit:
            return candidates

        try:
            model = self._load_model()
            # Chuẩn bị các cặp (câu hỏi, nội dung ứng viên) để chấm điểm
            pairs = [[query, candidate.content] for candidate in candidates]
            scores = model.predict(pairs)

            # Gán điểm rerank_score cho từng ứng viên
            for candidate, score in zip(candidates, scores):
                candidate.rerank_score = float(score)

            # Sắp xếp giảm dần theo điểm Rerank
            sorted_candidates = sorted(
                candidates, key=lambda c: c.rerank_score, reverse=True
            )
            return sorted_candidates[:limit]

        except Exception as error:
            logger.warning(
                "Không thể chạy Reranker (%s), tự động fallback theo điểm RRF.", error
            )
            # Fallback an toàn: nếu máy chưa tải được model reranker nặng thì dùng điểm RRF
            return sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[:limit]
