"""Service quản lý kết nối và gọi API tới LLM (LM Studio) với tính năng Streaming."""

import logging
from typing import AsyncGenerator
from openai import AsyncOpenAI
from rag1.retrival.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service sinh văn bản có hỗ trợ Streaming và vô hiệu hóa thinking."""

    def __init__(self) -> None:
        # Sử dụng AsyncOpenAI cho streaming bất đồng bộ
        self.client: AsyncOpenAI = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self.model_name: str = settings.llm_model

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        """Gửi prompt tới LLM và nhận câu trả lời dạng stream (từng chữ)."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1024,
                stream=True, # Bật streaming
            )
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as error:
            logger.error("Lỗi khi stream từ LLM (%s): %s", self.model_name, error)
            yield f"\n[Lỗi kết nối LLM: {str(error)}]"
