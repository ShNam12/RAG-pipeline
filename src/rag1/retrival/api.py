"""Module máy chủ Backend FastAPI cung cấp API hỏi đáp RAG."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag1.retrival.pipeline import RAGPipeline
from rag1.retrival.schemas import Citation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Biến toàn cục lưu trữ thể hiện Pipeline
pipeline_instance: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Quản lý vòng đời khởi tạo và giải phóng tài nguyên của Backend."""
    global pipeline_instance
    logger.info("Đang khởi tạo RAG Pipeline backend...")
    pipeline_instance = RAGPipeline()
    logger.info("RAG Pipeline đã sẵn sàng phục vụ yêu cầu!")
    yield
    logger.info("Đang tắt Backend Service...")


app = FastAPI(
    title="RAG Backend API",
    description="Dịch vụ hỏi đáp báo cáo tài chính sử dụng Agentic RAG Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# Cho phép kết nối CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Khuôn mẫu dữ liệu gửi lên từ người dùng."""

    query: str = Field(..., min_length=1, description="Câu hỏi của người dùng")


class ChatResponse(BaseModel):
    """Khuôn mẫu dữ liệu kết quả trả về cho Frontend."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Kiểm tra trạng thái hoạt động của Backend."""
    return {"status": "healthy", "service": "rag-backend"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: QueryRequest) -> ChatResponse:
    """API tiếp nhận câu hỏi, thực hiện RAG và trả về câu trả lời có trích dẫn."""
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng.")

    try:
        response = pipeline_instance.run(request.query)
        return ChatResponse(
            query=response.query,
            answer=response.answer,
            citations=response.citations,
        )
    except Exception as error:
        logger.error("Lỗi khi xử lý câu hỏi: %s", error)
        raise HTTPException(
            status_code=500, detail=f"Lỗi xử lý nội bộ: {str(error)}"
        ) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag1.retrival.api:app", host="0.0.0.0", port=8000, reload=False)
