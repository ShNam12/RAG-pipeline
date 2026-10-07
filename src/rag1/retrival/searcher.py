"""Module thực hiện tìm kiếm vector trên các collection của Qdrant."""

from __future__ import annotations

from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from rag1.retrival.config import settings
from rag1.retrival.schemas import SearchCandidate, SourceType


class QdrantSearcher:
    """Lớp xử lý tìm kiếm tương đồng trên Qdrant cho văn bản và bảng biểu."""

    def __init__(self) -> None:
        """Khởi tạo kết nối tới Qdrant Server."""
        self.client: QdrantClient = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    def _convert_point_to_candidate(
        self,
        point: ScoredPoint,
        source_type: SourceType,
        rank: int,
    ) -> SearchCandidate:
        """Chuyển đổi một ScoredPoint từ Qdrant sang model SearchCandidate chuẩn."""
        payload: dict[str, Any] = point.payload or {}

        # 1. Trích xuất tên file tài liệu từ source.path
        source_info = payload.get("source", {})
        if isinstance(source_info, dict) and "path" in source_info:
            doc_name = source_info["path"].split("/")[-1].replace("+", " ")
        else:
            doc_name = payload.get("doc_name", "Tài liệu không xác định")

        # 2. Trích xuất số trang
        page_number = payload.get("page_number", 1)

        # 3. Trích xuất nội dung (Ưu tiên text OCR có sẵn trong payload)
        if source_type == SourceType.TEXT:
            content = payload.get("parent_context") or payload.get("text") or payload.get("content", "")
            section_or_caption = payload.get("section", f"Trang {page_number}")
        else:
            # Đối với Bảng biểu:
            content = payload.get("text") or payload.get("table_markdown", "")
            proposal = payload.get("proposal", {})
            table_id = proposal.get("id") if isinstance(proposal, dict) else ""
            section_or_caption = payload.get("caption") or f"Bảng {table_id}".strip()

        # Lưu lại metadata đã chuẩn hóa để các bước sau (Prompt, Citation) dùng thuận tiện
        normalized_metadata = {
            **payload,
            "doc_name": doc_name,
            "page_number": page_number,
            "section_or_caption": section_or_caption,
        }

        return SearchCandidate(
            id=str(point.id),
            score=float(point.score),
            source_type=source_type,
            content=content,
            metadata=normalized_metadata,
            rank=rank,
        )

    
    def search_text(
        self,
        query_vector: list[float],
        limit: int | None = None,
    ) -> list[SearchCandidate]:
        """Tìm kiếm các đoạn văn bản tương đồng nhất trong text_index."""
        top_k = limit or settings.text_top_k

        if not self.client.collection_exists(settings.text_collection):
            return []

        scored_points = self.client.query_points(
            collection_name=settings.text_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            self._convert_point_to_candidate(point, SourceType.TEXT, rank=index + 1)
            for index, point in enumerate(scored_points)
        ]

    def search_table(
        self,
        query_vector: list[float],
        limit: int | None = None,
    ) -> list[SearchCandidate]:
        """Tìm kiếm các bảng biểu tương đồng nhất trong table_index."""
        top_k = limit or settings.table_top_k

        if not self.client.collection_exists(settings.table_collection):
            return []

        scored_points = self.client.query_points(
            collection_name=settings.table_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            self._convert_point_to_candidate(point, SourceType.TABLE, rank=index + 1)
            for index, point in enumerate(scored_points)
        ]

    def search_all(
        self,
        query_vector: list[float],
    ) -> tuple[list[SearchCandidate], list[SearchCandidate]]:
        """Tìm kiếm đồng thời trên cả hai collection văn bản và bảng."""
        text_results = self.search_text(query_vector)
        table_results = self.search_table(query_vector)
        return text_results, table_results
