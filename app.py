"""Giao diện Frontend Streamlit mỏng (Thin Client) kết nối tới Backend API."""

from __future__ import annotations

import os
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

# Thanh bên (Sidebar) hiển thị trạng thái kết nối
with st.sidebar:
    st.header("⚙️ Trạng thái hệ thống")
    try:
        health_resp = requests.get(HEALTH_API_URL, timeout=2)
        if health_resp.status_code == 200:
            st.success("🟢 Backend API: Đã kết nối")
        else:
            st.warning("🟡 Backend API: Phản hồi bất thường")
    except Exception:
        st.error("🔴 Backend API: Chưa kết nối (Hãy chạy Backend trước)")

    st.markdown("---")
    st.markdown("**Cấu hình Pipeline:**")
    st.markdown("- Vector DB: **Qdrant**")
    st.markdown("- Embedding: **Vietnamese_Embedding**")
    st.markdown("- LLM: **Qwen 3.5 4B (Docker)**")

st.title("📊 Trợ Lý Truy Vấn")
st.caption("Giao diện hỏi đáp tài liệu thông minh có trích dẫn nguồn gốc")

# Quản lý lịch sử chat trong Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn đã có
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("📚 Nguồn trích dẫn chi tiết"):
                for cit in msg["citations"]:
                    icon = "📑" if cit["source_type"] == "text" else "📊"
                    st.markdown(
                        f"- {icon} **[{cit['source_type'].upper()} {cit['index']}]** "
                        f"`{cit['doc_name']}` | Trang: **{cit['page_number']}** | {cit['section_or_caption']}"
                    )

# Nhận câu hỏi từ thanh chat
if user_query := st.chat_input("Nhập câu hỏi của bạn (ví dụ: Ai là Trưởng ban kiểm soát?)..."):
    # 1. Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Gọi Backend API và hiển thị phản hồi
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và sinh câu trả lời từ Backend..."):
            try:
                response = requests.post(
                    BACKEND_API_URL,
                    json={"query": user_query},
                    timeout=360,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer_text = data.get("answer", "")
                    citations_list = data.get("citations", [])

                    st.markdown(answer_text)

                    if citations_list:
                        with st.expander("📚 Nguồn trích dẫn chi tiết", expanded=True):
                            for cit in citations_list:
                                icon = "📑" if cit["source_type"] == "text" else "📊"
                                st.markdown(
                                    f"- {icon} **[{cit['source_type'].upper()} {cit['index']}]** "
                                    f"`{cit['doc_name']}` | Trang: **{cit['page_number']}** | {cit['section_or_caption']}"
                                )

                    # Lưu vào session
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "citations": citations_list,
                        }
                    )
                else:
                    st.error(f"Lỗi từ Backend ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Không thể kết nối đến Backend API. Bạn đã khởi chạy `uv run python -m rag1.retrival.api` chưa?")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
