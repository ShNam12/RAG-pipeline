from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check() -> dict[str, str]:
    """Kiểm tra trạng thái hoạt động của Backend."""
    return {"status": "healthy", "service": "rag-backend"}
