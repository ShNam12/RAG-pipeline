"""Module thực hiện tìm kiếm vector trên các collection của Qdrant."""

from __future__ import annotations

from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint, Filter, FieldCondition, MatchValue

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

        # 1. Trích xuất tên file tài liệu từ source
        source = payload.get("source", "")
        doc_name = "Tài liệu không xác định"
        if isinstance(source, str) and source:
            parts = source.replace("\\", "/").split("/")
            folder_name = parts[-2] if len(parts) > 1 and parts[-1] == "pages" else parts[-1]
            doc_name = folder_name.split("-")[0] if "-" in folder_name else folder_name
            doc_name = doc_name.replace("_", " ")

        # 2. Trích xuất số trang
        page_numbers = payload.get("page_numbers", [])
        page_number = page_numbers[0] if isinstance(page_numbers, list) and page_numbers else "Không rõ"

        # 3. Trích xuất nội dung
        if source_type == SourceType.TEXT:
            content = payload.get("text") or payload.get("content", "")
            section_heading = payload.get("section_heading")
            section_or_caption = section_heading if section_heading else f"Trang {page_number}"
        else:
            # Đối với Bảng biểu:
            content = payload.get("text") or payload.get("table_markdown", "")
            section_or_caption = f"Bảng (Trang {page_number})"

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

        text_filter = Filter(must=[FieldCondition(key="level", match=MatchValue(value=2))])

        scored_points = self.client.query_points(
            collection_name=settings.text_collection,
            query=query_vector,
            using="text",
            limit=top_k,
            query_filter=text_filter,
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

        table_filter = Filter(must_not=[FieldCondition(key="level", match=MatchValue(value=2))])

        scored_points = self.client.query_points(
            collection_name=settings.table_collection,
            query=query_vector,
            using="text",
            limit=top_k,
            query_filter=table_filter,
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
