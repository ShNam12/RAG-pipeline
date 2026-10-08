import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from rag1.retrival.schemas import QueryRequest
from rag1.retrival.services.rag_service import RAGService
from rag1.retrival.services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency Injection function
def get_rag_service():
    return RAGService()

def get_llm_service():
    return LLMService()

@router.post("/api/chat")
async def chat_endpoint(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """API tiếp nhận câu hỏi và trả về luồng chữ (streaming) ngay lập tức."""
    try:
        # 1. Tìm kiếm và tạo prompt
        system_prompt, user_prompt, citations, metrics = rag_service.process_query(request.query)

        # 2. Hàm Generator (Sinh text đến đâu gửi đến đó)
        async def stream_generator():
            # Trả về từng chữ từ LLM
            async for chunk in llm_service.generate_stream(system_prompt, user_prompt):
                yield chunk

            # Sau khi LLM trả lời xong, chèn thêm phần Nguồn Tham Khảo vào cuối câu
            if citations:
                yield "\n\n---\n**📚 Nguồn tham khảo:**\n"
                for cit in citations:
                    yield f"- **[{cit.source_type.value.upper()} {cit.index}]** {cit.doc_name} (Trang {cit.page_number} | Mục: {cit.section_or_caption})\n"

            # Đính kèm metadata đánh giá ở cuối luồng (kèm thẻ delimiter)
            if metrics:
                metrics_json = json.dumps(metrics, ensure_ascii=False)
                yield f"\n\n__RAG_EVAL_METADATA_START__{metrics_json}__RAG_EVAL_METADATA_END__"

        # Trả về kết nối luồng
        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as error:
        logger.error("Lỗi khi xử lý câu hỏi: %s", error)
        raise HTTPException(
            status_code=500, detail=f"Lỗi xử lý nội bộ: {str(error)}"
        ) from error
