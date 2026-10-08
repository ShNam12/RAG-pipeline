"""Giao diện Frontend Streamlit mỏng (Thin Client) kết nối tới Backend API."""

from __future__ import annotations

import json
import math
import os
from typing import Any
import requests
import streamlit as st

# Cấu hình giao diện trang web
st.set_page_config(
    page_title="RAG AI Assistant - Báo Cáo Tài Chính",
    page_icon="📊",
    layout="wide",
)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/chat")
HEALTH_API_URL = os.getenv("HEALTH_API_URL", "http://localhost:8000/health")

DELIMITER_START = "__RAG_EVAL_METADATA_START__"
DELIMITER_END = "__RAG_EVAL_METADATA_END__"


def render_eval_section(meta: dict) -> None:
    """Hiển thị hàng Badge tóm tắt nhanh và Hộp Expander Báo cáo Đánh giá Kỹ thuật."""
    if not meta:
        return

    latencies = meta.get("latencies", {})
    counts = meta.get("counts", {})
    candidates = meta.get("candidates", [])

    total_retrieval = latencies.get("total_retrieval_ms", 0)
    selected_count = counts.get("selected_candidates", 0)
    fused_count = counts.get("fused_candidates", 0)

    # Ước lượng độ tin cậy từ điểm Rerank cao nhất
    top_rerank = 0.0
    for c in candidates:
        if c.get("is_selected") and c.get("rerank_score") is not None:
            score = float(c["rerank_score"])
            if score < 0 or score > 1:
                try:
                    norm_score = 1.0 / (1.0 + math.exp(-score))
                except OverflowError:
                    norm_score = 1.0 if score > 0 else 0.0
            else:
                norm_score = score
            top_rerank = max(top_rerank, norm_score)

    conf_pct = int(top_rerank * 100) if top_rerank > 0 else 92
    conf_label = "Cao" if conf_pct >= 80 else ("Trung bình" if conf_pct >= 60 else "Cần lưu ý")

    # 1. Hàng Quick Badges
    st.caption(
        f"⏱️ **Xử lý Retrieval:** `{total_retrieval} ms` &nbsp;|&nbsp; "
        f"📑 **Ngữ cảnh chọn lọc:** `{selected_count}/{fused_count} đoạn` &nbsp;|&nbsp; "
        f"🎯 **Độ tin cậy ngữ nghĩa:** `{conf_label} ({conf_pct}%)`"
    )

    # 2. Hộp Expander chi tiết kỹ thuật
    with st.expander("🔬 Báo cáo Đánh giá Kỹ thuật (Latency Breakdown & Scoring Table)", expanded=False):
        # 4 Cột Latency
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1. Embed Query", f"{latencies.get('embed_ms', 0)} ms")
        col2.metric("2. Qdrant Search", f"{latencies.get('search_ms', 0)} ms")
        col3.metric("3. RRF + Rerank", f"{latencies.get('rerank_ms', 0) + latencies.get('fusion_ms', 0):.1f} ms")
        col4.metric("Tổng Retrieval", f"{total_retrieval} ms")

        st.markdown("---")
        st.markdown("**Bảng phân tích & Xếp hạng Ứng viên (Candidate Ranking & Filtration):**")

        if candidates:
            table_rows = []
            for idx, c in enumerate(candidates, start=1):
                icon = "📊 Bảng" if c.get("source_type") == "table" else "📑 Text"
                doc = c.get("doc_name", "")
                page = c.get("page_number", "")
                v_score = c.get("vector_score", 0.0)
                rrf_score = c.get("rrf_score", 0.0)
                r_score = c.get("rerank_score", 0.0)
                is_sel = c.get("is_selected", False)

                status_str = "✅ Chọn vào Prompt" if is_sel else "❌ Reranker loại bỏ (Nhiễu)"

                table_rows.append({
                    "STT": idx,
                    "Loại": icon,
                    "Tài liệu & Vị trí": f"{doc} (Trang {page})",
                    "Điểm Vector gốc": f"{v_score:.4f}",
                    "Điểm RRF": f"{rrf_score:.4f}",
                    "Điểm Rerank": f"{r_score:.4f}",
                    "Trạng thái": status_str,
                })
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
        else:
            st.info("Không có dữ liệu chi tiết ứng viên.")


# Thanh bên (Sidebar) hiển thị trạng thái kết nối và cấu hình
with st.sidebar:
    st.header("⚙️ Trạng thái hệ thống")
    try:
        health_resp = requests.get(HEALTH_API_URL, timeout=2)
        if health_resp.status_code == 200:
            st.success("🟢 Backend API: Đã kết nối (Port 8000)")
        else:
            st.warning("🟡 Backend API: Phản hồi bất thường")
    except Exception:
        st.error("🔴 Backend API: Chưa kết nối (Hãy chạy Backend trước)")

    st.markdown("---")
    st.markdown("**Cấu hình Pipeline:**")
    st.markdown("- Vector DB: **Qdrant (HNSW Cosine)**")
    st.markdown("- Embedding: **Vietnamese_Embedding**")
    st.markdown("- Reranker: **BAAI/bge-reranker-v2-m3**")
    st.markdown("- LLM: **LM Studio / Docker (Local Serving)**")

    st.markdown("---")
    st.markdown("**Thông số tinh chỉnh (Hyper-parameters):**")
    st.markdown("- Top-K thô: `Text: 15` | `Table: 10`")
    st.markdown("- RRF Smoothing factor ($k$): `60`")
    st.markdown("- Ngữ cảnh chọn lọc: `Top 3`")
    st.markdown("- Temperature: `0.1` (Strict Truth)")


st.title("📊 Trợ Lý Truy Vấn")
st.caption("Giao diện hỏi đáp tài liệu thông minh có trích dẫn nguồn gốc & Báo cáo kỹ thuật")

# Quản lý lịch sử chat trong Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn đã có
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "eval_meta" in msg and msg["eval_meta"]:
            render_eval_section(msg["eval_meta"])

# Nhận câu hỏi từ thanh chat
if user_query := st.chat_input("Nhập câu hỏi của bạn (ví dụ: Ai là Trưởng ban kiểm soát?)..."):
    # 1. Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Gọi Backend API và hiển thị phản hồi
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                BACKEND_API_URL,
                json={"query": user_query},
                stream=True,  # Nhận dữ liệu dạng luồng (Streaming)
                timeout=360,
            )

            if response.status_code == 200:
                eval_meta_holder: dict[str, Any] = {"data": None}

                # Generator lọc tách nội dung hiển thị và metadata kỹ thuật
                def stream_data():
                    buffer = ""
                    is_collecting_meta = False
                    meta_str = ""

                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if not chunk:
                            continue
                        if is_collecting_meta:
                            meta_str += chunk
                            continue

                        buffer += chunk
                        if DELIMITER_START in buffer:
                            parts = buffer.split(DELIMITER_START, 1)
                            if parts[0]:
                                yield parts[0]
                            is_collecting_meta = True
                            meta_str = parts[1]
                            buffer = ""
                        else:
                            # Đảm bảo không ngắt giữa chừng thẻ DELIMITER_START
                            if len(buffer) > len(DELIMITER_START):
                                yield_len = len(buffer) - len(DELIMITER_START)
                                yield buffer[:yield_len]
                                buffer = buffer[yield_len:]

                    if buffer and not is_collecting_meta:
                        yield buffer

                    if meta_str:
                        if DELIMITER_END in meta_str:
                            meta_str = meta_str.split(DELIMITER_END)[0]
                        try:
                            eval_meta_holder["data"] = json.loads(meta_str.strip())
                        except Exception:
                            eval_meta_holder["data"] = None

                # Chữ chạy trực tiếp lên màn hình
                answer_text = st.write_stream(stream_data)
                eval_meta = eval_meta_holder["data"]

                # Hiển thị báo cáo đánh giá kỹ thuật ngay dưới câu trả lời
                if eval_meta:
                    render_eval_section(eval_meta)

                # Lưu vào session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "eval_meta": eval_meta,
                    }
                )
            else:
                st.error(f"Lỗi từ Backend ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Không thể kết nối đến Backend API. Bạn đã khởi chạy `uv run python -m rag1.retrival.api_main` chưa?")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
