"""Điểm khởi chạy chính của RAG Backend API."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from rag1.retrival.routers import chat, health
from rag1.retrival.services.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời khởi tạo và giải phóng tài nguyên."""
    logger.info("Đang khởi tạo các module tĩnh (Embedding, DB) cho Backend...")
    # Tải trước (warm-up) Embedder bằng cách khởi tạo RAGService 1 lần
    RAGService()
    logger.info("Backend đã khởi tạo xong, sẵn sàng phục vụ!")
    yield
    logger.info("Đang tắt Backend Service...")

app = FastAPI(
    title="RAG Backend API (Streaming)",
    description="Dịch vụ hỏi đáp báo cáo tài chính tốc độ cao",
    version="2.0.0",
    lifespan=lifespan,
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các Router
app.include_router(health.router)
app.include_router(chat.router)

if __name__ == "__main__":
    uvicorn.run("rag1.retrival.api_main:app", host="0.0.0.0", port=8000, reload=True)
