"""Module điều phối (Orchestrator) toàn bộ luồng RAG Pipeline."""

from __future__ import annotations

import sys
from rag1.retrival.embedder import QueryEmbedder
from rag1.retrival.fusion import reciprocal_rank_fusion
from rag1.retrival.llm_client import LLMClient
from rag1.retrival.prompt_builder import PromptBuilder
from rag1.retrival.reranker import Reranker
from rag1.retrival.schemas import RAGResponse
from rag1.retrival.searcher import QdrantSearcher


class RAGPipeline:
    """Lớp điều phối chính thực hiện toàn bộ quy trình RAG Agentic."""

    def __init__(self) -> None:
        """Khởi tạo tất cả các thành phần trong pipeline."""
        self.embedder: QueryEmbedder = QueryEmbedder.get_instance()
        self.searcher: QdrantSearcher = QdrantSearcher()
        self.reranker: Reranker = Reranker()
        self.prompt_builder: PromptBuilder = PromptBuilder()
        self.llm_client: LLMClient = LLMClient()

    def run(self, query: str) -> RAGResponse:
        """Chạy toàn bộ pipeline từ câu hỏi đến câu trả lời hoàn chỉnh.

        Quy trình:
            1. Vector hóa câu hỏi.
            2. Tìm kiếm song song text_index và table_index.
            3. Hợp nhất kết quả bằng RRF.
            4. Tái xếp hạng bằng Reranker.
            5. Tạo Prompt và danh sách Citation.
            6. Sinh câu trả lời qua Qwen 3.5 4B.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return RAGResponse(query="", answer="Câu hỏi không hợp lệ.", citations=[], candidates=[])

        # Bước 1: Vector hóa câu hỏi
        query_vector = self.embedder.encode_query(cleaned_query)

        # Bước 2: Tìm kiếm song song trên Qdrant
        text_candidates, table_candidates = self.searcher.search_all(query_vector)

        # Bước 3: Hợp nhất bằng RRF
        fused_candidates = reciprocal_rank_fusion(
            text_candidates=text_candidates,
            table_candidates=table_candidates,
            top_n=15,
        )

        # Bước 4: Tái xếp hạng bằng Reranker lấy Top kết quả tốt nhất
        reranked_candidates = self.reranker.rerank(
            query=cleaned_query,
            candidates=fused_candidates,
        )

        if not reranked_candidates:
            return RAGResponse(
                query=cleaned_query,
                answer="Không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu.",
                citations=[],
                candidates=[],
            )

        # Bước 5: Xây dựng Prompt và gán tag trích dẫn
        system_prompt, user_prompt, citations = self.prompt_builder.build_prompt(
            query=cleaned_query,
            candidates=reranked_candidates,
        )

        # Bước 6: Gọi Qwen 3.5 4B trên Docker
        answer = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return RAGResponse(
            query=cleaned_query,
            answer=answer,
            citations=citations,
            candidates=reranked_candidates,
        )


def main() -> None:
    """Hàm chạy thử nghiệm pipeline trực tiếp từ terminal."""
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Ai là Trưởng ban kiểm soát trong báo cáo tài chính?"
    )

    print(f"\n=== ĐANG XỬ LÝ CÂU HỎI: {query} ===\n")
    pipeline = RAGPipeline()
    response = pipeline.run(query)

    print("--- CÂU TRẢ LỜI TỪ QWEN 3.5 ---")
    print(response.answer)
    print("\n--- DANH SÁCH TRÍCH DẪN (CITATIONS) ---")
    for cit in response.citations:
        print(f"[{cit.source_type.value.upper()} {cit.index}] {cit.doc_name} | Trang {cit.page_number} | {cit.section_or_caption}")


if __name__ == "__main__":
    main()
