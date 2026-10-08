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
    st.markdown("- LLM: **LM Studio (Qwen 3.5 4B)**")

st.title("📊 Trợ Lý Truy Vấn")
st.caption("Giao diện hỏi đáp tài liệu thông minh có trích dẫn nguồn gốc")

# Quản lý lịch sử chat trong Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn đã có
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
                stream=True, # Nhận dữ liệu dạng luồng (Streaming)
                timeout=360,
            )

            if response.status_code == 200:
                # Trình bày chữ chạy lên màn hình (Streaming output)
                def stream_data():
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            yield chunk
                            
                answer_text = st.write_stream(stream_data)

                # Lưu vào session
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                    }
                )
            else:
                st.error(f"Lỗi từ Backend ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Không thể kết nối đến Backend API. Bạn đã khởi chạy `uv run python -m rag1.retrival.api_main` chưa?")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
