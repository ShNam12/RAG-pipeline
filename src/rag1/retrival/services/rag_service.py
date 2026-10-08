import time
from typing import Any

from rag1.retrival.embedder import QueryEmbedder
from rag1.retrival.fusion import reciprocal_rank_fusion
from rag1.retrival.prompt_builder import PromptBuilder
from rag1.retrival.reranker import Reranker
from rag1.retrival.searcher import QdrantSearcher


class RAGService:
    """Service điều phối truy xuất tài liệu và xây dựng prompt."""

    def __init__(self) -> None:
        self.embedder: QueryEmbedder = QueryEmbedder.get_instance()
        self.searcher: QdrantSearcher = QdrantSearcher()
        self.reranker: Reranker = Reranker()
        self.prompt_builder: PromptBuilder = PromptBuilder()

    def process_query(self, query: str) -> tuple[str, str, list, dict[str, Any]]:
        """Xử lý câu hỏi, tìm tài liệu, và tạo prompt.

        Trả về: (system_prompt, user_prompt, citations, metrics)
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return "", "Câu hỏi không hợp lệ.", [], {}

        t_total_start = time.perf_counter()

        # Bước 1: Vector hóa
        t_embed_start = time.perf_counter()
        query_vector = self.embedder.encode_query(cleaned_query)
        t_embed = (time.perf_counter() - t_embed_start) * 1000

        # Bước 2: Tìm kiếm
        t_search_start = time.perf_counter()
        text_candidates, table_candidates = self.searcher.search_all(query_vector)
        t_search = (time.perf_counter() - t_search_start) * 1000

        # Bước 3: RRF Fusion
        t_fusion_start = time.perf_counter()
        fused_candidates = reciprocal_rank_fusion(
            text_candidates=text_candidates,
            table_candidates=table_candidates,
            top_n=15,
        )
        t_fusion = (time.perf_counter() - t_fusion_start) * 1000

        # Bước 4: Reranking
        t_rerank_start = time.perf_counter()
        reranked_candidates = self.reranker.rerank(
            query=cleaned_query,
            candidates=fused_candidates,
        )
        t_rerank = (time.perf_counter() - t_rerank_start) * 1000
        t_total = (time.perf_counter() - t_total_start) * 1000

        # Thu thập thông số đánh giá (Evaluation metrics & Candidate ranking details)
        selected_ids = {c.id for c in reranked_candidates}
        eval_candidates = []
        for c in fused_candidates[:10]:
            eval_candidates.append({
                "id": str(c.id),
                "source_type": c.source_type.value,
                "doc_name": c.metadata.get("doc_name", "Tài liệu"),
                "page_number": c.metadata.get("page_number", 1),
                "section_or_caption": c.metadata.get("section_or_caption", ""),
                "vector_score": round(float(c.score), 4) if c.score else 0.0,
                "rrf_score": round(float(c.rrf_score), 4) if c.rrf_score else 0.0,
                "rerank_score": round(float(c.rerank_score), 4) if c.rerank_score else 0.0,
                "is_selected": c.id in selected_ids,
            })

        metrics: dict[str, Any] = {
            "latencies": {
                "embed_ms": round(t_embed, 1),
                "search_ms": round(t_search, 1),
                "fusion_ms": round(t_fusion, 1),
                "rerank_ms": round(t_rerank, 1),
                "total_retrieval_ms": round(t_total, 1),
            },
            "counts": {
                "text_candidates": len(text_candidates),
                "table_candidates": len(table_candidates),
                "fused_candidates": len(fused_candidates),
                "selected_candidates": len(reranked_candidates),
            },
            "candidates": eval_candidates,
        }

        if not reranked_candidates:
            return "Bạn là trợ lý AI.", "Không tìm thấy tài liệu phù hợp để trả lời.", [], metrics

        # Bước 5: Build Prompt
        system_prompt, user_prompt, citations = self.prompt_builder.build_prompt(
            query=cleaned_query,
            candidates=reranked_candidates,
        )

        return system_prompt, user_prompt, citations, metrics
