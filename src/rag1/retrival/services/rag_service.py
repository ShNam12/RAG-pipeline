"""Service điều phối logic RAG (Tìm kiếm, Rerank, Tạo Prompt)."""

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

    def process_query(self, query: str) -> tuple[str, str, list]:
        """Xử lý câu hỏi, tìm tài liệu, và tạo prompt.
        
        Trả về: (system_prompt, user_prompt, citations)
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return "", "Câu hỏi không hợp lệ.", []

        # Bước 1: Vector hóa
        query_vector = self.embedder.encode_query(cleaned_query)

        # Bước 2: Tìm kiếm
        text_candidates, table_candidates = self.searcher.search_all(query_vector)

        # Bước 3: RRF Fusion
        fused_candidates = reciprocal_rank_fusion(
            text_candidates=text_candidates,
            table_candidates=table_candidates,
            top_n=15,
        )

        # Bước 4: Reranking
        reranked_candidates = self.reranker.rerank(
            query=cleaned_query,
            candidates=fused_candidates,
        )

        if not reranked_candidates:
            return "Bạn là trợ lý AI.", "Không tìm thấy tài liệu phù hợp để trả lời.", []

        # Bước 5: Build Prompt
        system_prompt, user_prompt, citations = self.prompt_builder.build_prompt(
            query=cleaned_query,
            candidates=reranked_candidates,
        )

        return system_prompt, user_prompt, citations
