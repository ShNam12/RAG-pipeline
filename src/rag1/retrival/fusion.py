"""Module hợp nhất kết quả tìm kiếm đa nguồn bằng thuật toán RRF."""

from __future__ import annotations

from rag1.retrival.config import settings
from rag1.retrival.schemas import SearchCandidate


def reciprocal_rank_fusion(
    text_candidates: list[SearchCandidate],
    table_candidates: list[SearchCandidate],
    k: int | None = None,
    top_n: int = 15,
) -> list[SearchCandidate]:
    """Hợp nhất và tính điểm thứ hạng tương hỗ (RRF) cho kết quả từ văn bản và bảng.

    Công thức: RRF_Score(d) = sum( 1 / (k + rank(d)) )

    Tham số:
        text_candidates: Danh sách kết quả từ text_index đã được xếp hạng.
        table_candidates: Danh sách kết quả từ table_index đã được xếp hạng.
        k: Hệ số làm dịu (mặc định 60 theo chuẩn bài báo RRF).
        top_n: Số lượng ứng viên tối đa trả về sau khi hợp nhất.

    Trả về:
        Danh sách ứng viên đã được gộp, khử trùng lặp và sắp xếp theo rrf_score giảm dần.
    """
    smoothing_k = k or settings.rrf_k
    fused_scores: dict[str, float] = {}
    candidate_map: dict[str, SearchCandidate] = {}

    def _generate_dedup_key(candidate: SearchCandidate) -> str:
        """Tạo khóa định danh duy nhất để khử trùng lặp tài liệu."""
        doc_name = candidate.metadata.get("doc_name", "")
        page_num = candidate.metadata.get("page_number", "")
        # Nếu có id riêng thì dùng id, nếu không kết hợp tên file + trang + loại
        return f"{candidate.source_type.value}_{candidate.id}_{doc_name}_{page_num}"

    # 1. Tính điểm RRF cho nhánh văn bản
    for candidate in text_candidates:
        key = _generate_dedup_key(candidate)
        score_contrib = 1.0 / (smoothing_k + candidate.rank)
        fused_scores[key] = fused_scores.get(key, 0.0) + score_contrib
        if key not in candidate_map:
            candidate_map[key] = candidate

    # 2. Tính điểm RRF cho nhánh bảng biểu
    for candidate in table_candidates:
        key = _generate_dedup_key(candidate)
        score_contrib = 1.0 / (smoothing_k + candidate.rank)
        fused_scores[key] = fused_scores.get(key, 0.0) + score_contrib
        if key not in candidate_map:
            candidate_map[key] = candidate

    # 3. Gán điểm rrf_score vào đối tượng candidate
    result_list: list[SearchCandidate] = []
    for key, total_score in fused_scores.items():
        item = candidate_map[key]
        item.rrf_score = total_score
        result_list.append(item)

    # 4. Sắp xếp giảm dần theo rrf_score
    result_list.sort(key=lambda x: x.rrf_score, reverse=True)

    # 5. Cập nhật lại thứ hạng mới sau khi hợp nhất và lấy top_n
    for new_rank, item in enumerate(result_list, start=1):
        item.rank = new_rank

    return result_list[:top_n]
