"""Module xây dựng Prompt ngữ cảnh và danh sách trích dẫn nguồn."""

from __future__ import annotations

from rag1.retrival.schemas import Citation, SearchCandidate, SourceType
from rag1.retrival.table_formatter import format_table_to_markdown

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên nghiệp về phân tích tài liệu và báo cáo tài chính.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA HOÀN TOÀN vào các ngữ cảnh văn bản và bảng biểu được cung cấp.

QUY TẮC BẮT BUỘC:
1. Trả lời chính xác, trung thực, ngắn gọn và rõ ràng.
2. Với mọi thông tin hoặc số liệu đưa ra, bạn BẮT BUỘC phải trích dẫn nguồn gốc theo định dạng: [Tài liệu X, Trang Y] hoặc [Bảng Z, Trang Y].
3. Nếu thông tin không xuất hiện trong ngữ cảnh, hãy nói rõ là tài liệu không đề cập, tuyệt đối không tự suy đoán hoặc bịa số liệu.
"""


class PromptBuilder:
    """Lớp xử lý ghép ngữ cảnh và xây dựng prompt hoàn chỉnh cho mô hình ngôn ngữ."""

    def build_prompt(
        self,
        query: str,
        candidates: list[SearchCandidate],
    ) -> tuple[str, str, list[Citation]]:
        """Tạo system prompt, user prompt và danh sách citation từ các kết quả tìm kiếm.

        Tham số:
            query: Câu hỏi gốc của người dùng.
            candidates: Danh sách ứng viên tốt nhất sau bước Reranker.

        Trả về:
            Tuple gồm (system_prompt, user_prompt, list[Citation]).
        """
        citations: list[Citation] = []
        context_blocks: list[str] = []

        text_idx = 1
        table_idx = 1

        for candidate in candidates:
            meta = candidate.metadata
            doc_name = meta.get("doc_name", "Tài liệu")
            page_number = meta.get("page_number", 1)
            section_or_caption = meta.get("section_or_caption", "")

            if candidate.source_type == SourceType.TABLE:
                label = f"[BẢNG {table_idx}]"
                header_info = f"{label} (Tài liệu: {doc_name} | Trang: {page_number} | {section_or_caption})"

                # Format nội dung bảng sang markdown lưới hoặc danh sách
                table_content = format_table_to_markdown(
                    table_data=meta.get("table", {}).get("structure") or meta.get("table_json"),
                    raw_text=candidate.content,
                )
                context_blocks.append(f"{header_info}\n{table_content}")

                citations.append(
                    Citation(
                        index=table_idx,
                        source_type=SourceType.TABLE,
                        doc_name=doc_name,
                        page_number=page_number,
                        section_or_caption=section_or_caption,
                    )
                )
                table_idx += 1
            else:
                label = f"[TÀI LIỆU {text_idx}]"
                header_info = f"{label} (Tài liệu: {doc_name} | Trang: {page_number} | Mục: {section_or_caption})"
                context_blocks.append(f"{header_info}\n{candidate.content.strip()}")

                citations.append(
                    Citation(
                        index=text_idx,
                        source_type=SourceType.TEXT,
                        doc_name=doc_name,
                        page_number=page_number,
                        section_or_caption=section_or_caption,
                    )
                )
                text_idx += 1

        joined_context = "\n\n".join(context_blocks)
        user_prompt = f"""Dưới đây là các tài liệu và số liệu tham chiếu:

{joined_context}

---------------------------------
CÂU HỎI: {query}

HÃY TRẢ LỜI CÂU HỎI TRÊN VÀ DẪN NGUỒN TRÍCH DẪN ĐẦY ĐỦ:"""

        return SYSTEM_PROMPT, user_prompt, citations
