"""Module giao tiếp với Docker Qwen 3.5 4B qua OpenAI API protocol."""

from __future__ import annotations

import logging
from openai import OpenAI
from rag1.retrival.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Lớp kết nối và gửi yêu cầu sinh văn bản tới mô hình ngôn ngữ lớn."""

    def __init__(self) -> None:
        """Khởi tạo OpenAI client trỏ về máy chủ Docker cục bộ."""
        self.client: OpenAI = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self.model_name: str = settings.llm_model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Gửi prompt tới LLM và nhận câu trả lời dạng văn bản.

        Tham số:
            system_prompt: Chỉ dẫn hệ thống và quy tắc trích dẫn.
            user_prompt: Ngữ cảnh trích xuất và câu hỏi.
            temperature: Nhiệt độ sinh văn bản (mặc định 0.1 để đảm bảo tính chính xác).

        Trả về:
            Nội dung phản hồi hoàn chỉnh từ LLM.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        except Exception as error:
            logger.error("Lỗi khi gọi LLM (%s): %s", self.model_name, error)
            raise RuntimeError(f"Không thể kết nối hoặc sinh câu trả lời từ LLM: {error}") from error
