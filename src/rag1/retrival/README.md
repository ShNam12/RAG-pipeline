# ĐẶC TẢ PHÁT TRIỂN MODULE RETRIEVAL & GENERATION (AI AGENT SPEC)

Tài liệu này đóng vai trò là kim chỉ nam kỹ thuật (Technical Specification & Prompt Guide) để AI hoặc lập trình viên phát triển toàn bộ module RAG Serving nằm trong thư mục `src/rag1/retrival/`.

The two-collection examples below describe the earlier mock contract.
For the current Qdrant indexer and the hybrid and extracted-table migration, follow [INTEGRATION_INSTRUCTIONS.md](INTEGRATION_INSTRUCTIONS.md) and [RETRIEVAL_HANDOFF.md](../../../schemas/qdrant/RETRIEVAL_HANDOFF.md).

---

## 1. TỔNG QUAN DỰ ÁN & BỐI CẢNH

* **Vị trí trong hệ thống:** Xử lý từ giai đoạn người dùng gửi câu hỏi cho đến khi trả về câu trả lời hoàn chỉnh kèm trích dẫn nguồn (tài liệu, số trang, mục và bảng).
* **Môi trường & Công nghệ:**
  - Ngôn ngữ: Python 3.12, quản lý qua `uv`.
  - Cơ sở dữ liệu Vector: **Qdrant** (chạy trên Docker local, mặc định `http://localhost:6333`).
  - Embedding Model: `thanhtantran/Vietnamese_Embedding` qua `sentence-transformers`.
  - LLM Model: **Qwen 3.5 4B** (chạy trên Docker local, giao thức OpenAI-compatible, chế độ `enable_thinking=false`).
  - Toàn bộ text vector, bảng (JSON/Markdown) và metadata ngữ cảnh gốc đều được lưu trực tiếp trong **Payload của Qdrant** (không dùng MongoDB).

---

## 2. QUY CHUẨN MÃ NGUỒN (CODING CONVENTIONS)

1. **Tuân thủ AGENTS.md:**
   - Thụt lề 4 dấu cách (spaces).
   - Đặt tên theo chuẩn PEP 8: `snake_case` cho hàm và biến, `PascalCase` cho lớp (Class), `UPPER_CASE` cho hằng số.
   - Bắt buộc có **Type Annotations** đầy đủ trên toàn bộ hàm/phương thức công khai (`def func(arg: str) -> list[dict]:`).
   - Sử dụng `Pydantic` (v2) để định nghĩa cấu trúc dữ liệu truyền giữa các module.
2. **Nguyên tắc phát triển:**
   - Mỗi file đảm nhiệm đúng một trách nhiệm đơn nhất (Single Responsibility).
   - Có cơ chế Mock/Seed dữ liệu để kiểm thử độc lập mà không cần chờ dữ liệu thật từ partner.

---

## 3. HỢP ĐỒNG DỮ LIỆU QDRANT (DATA CONTRACT VỚI PARTNER)

Hệ thống sử dụng 2 collections độc lập trên Qdrant:

### 3.1. Collection: `text_index`
* **Vector:** Embed từ nội dung chunk bằng `thanhtantran/Vietnamese_Embedding` (Cosine Distance).
* **Cấu trúc Payload:**
```json
{
  "type": "text",
  "doc_name": "Bao_cao_tai_chinh_Q3_2025.pdf",
  "page_number": 5,
  "section": "3. Kết quả kinh doanh",
  "content": "Doanh thu thuần hợp nhất quý 3 năm 2025 đạt 15.200 tỷ đồng...",
  "parent_context": "Ngữ cảnh cha hoàn chỉnh hoặc đoạn văn bao quanh chunk này"
}
```

### 3.2. Collection: `table_index`
* **Vector:** Embed từ tóm tắt / tiêu đề bảng (Cosine Distance).
* **Cấu trúc Payload:**
```json
{
  "type": "table",
  "doc_name": "Bao_cao_tai_chinh_Q3_2025.pdf",
  "page_number": 6,
  "caption": "Bảng cân đối kế toán quý 3/2025",
  "table_markdown": "| Chỉ tiêu | Q3/2024 | Q3/2025 |\n|---|---|---|\n| Doanh thu | 13.500 | 15.200 |",
  "table_json": {
    "headers": ["Chỉ tiêu", "Q3/2024", "Q3/2025"],
    "rows": [["Doanh thu", "13.500", "15.200"]]
  }
}
```

---

## 4. CẤU TRÚC THƯ MỤC VÀ ĐẶC TẢ CÁC TẬP TIN

```text
src/rag1/retrival/
├── __init__.py            # Export RAGPipeline, RAGResponse
├── config.py              # Đọc cấu hình từ .env
├── schemas.py             # Pydantic data models
├── embedder.py            # SentenceTransformers wrapper
├── searcher.py            # Truy vấn Qdrant song song
├── fusion.py              # Reciprocal Rank Fusion (RRF)
├── reranker.py            # Mô hình Cross-Encoder/Reranker
├── table_formatter.py     # Chuyển đổi JSON bảng sang Markdown
├── prompt_builder.py      # Xây dựng prompt kèm tag trích dẫn
├── llm_client.py          # Kết nối Qwen 3.5 Docker
├── pipeline.py            # Orchestrator xâu chuỗi toàn bộ luồng
└── seed_mock.py           # Script nạp mock data vào Qdrant để test
```

---

## 5. ĐẶC TẢ KỸ THUẬT CHI TIẾT TỪNG MODULE

### 5.1. `config.py`
* Đọc các biến từ file `.env` bằng `pydantic_settings` hoặc `os.getenv`:
  - `QDRANT_URL`: Mặc định `http://localhost:6333`
  - `QDRANT_API_KEY`: Tuỳ chọn (None nếu local không yêu cầu)
  - `TEXT_COLLECTION_NAME`: Mặc định `text_index`
  - `TABLE_COLLECTION_NAME`: Mặc định `table_index`
  - `EMBEDDING_MODEL_NAME`: Mặc định `thanhtantran/Vietnamese_Embedding`
  - `LLM_BASE_URL`: Mặc định `http://localhost:11434/v1` (hoặc cổng Docker Qwen)
  - `LLM_API_KEY`: Mặc định `ollama` hoặc `local`
  - `LLM_MODEL_NAME`: Mặc định `qwen3.5:4b`

### 5.2. `schemas.py`
* `SourceType`: Enum (`text`, `table`).
* `Citation`: Pydantic Model chứa: `index` (int), `doc_name` (str), `page_number` (int), `section_or_caption` (str), `source_type` (str).
* `SearchCandidate`: Pydantic Model chứa: `id` (str|int), `score` (float), `source_type` (SourceType), `content` (str), `metadata` (dict), `rank` (int, default=0), `rrf_score` (float, default=0.0).
* `RAGResponse`: Pydantic Model chứa: `answer` (str), `citations` (list[Citation]), `used_context` (list[SearchCandidate]).

### 5.3. `embedder.py`
* Class `QueryEmbedder`:
  - Khởi tạo `SentenceTransformer(model_name)`.
  - Singleton / Lazy initialization để tránh load model lặp lại.
  - Method `encode_query(query: str) -> list[float]`: Trả về vector dạng float list.

### 5.4. `searcher.py`
* Class `QdrantSearcher`:
  - Kết nối tới `QdrantClient`.
  - Method `search_text(query_vector: list[float], limit: int = 15) -> list[SearchCandidate]`
  - Method `search_table(query_vector: list[float], limit: int = 10) -> list[SearchCandidate]`
  - Method `search_all(query_vector: list[float]) -> tuple[list[SearchCandidate], list[SearchCandidate]]` thực hiện tìm kiếm song song hoặc tuần tự cả text và table.

### 5.5. `fusion.py`
* Function `reciprocal_rank_fusion(text_results: list[SearchCandidate], table_results: list[SearchCandidate], k: int = 60, top_n: int = 15) -> list[SearchCandidate]`
  - Thuật toán RRF:
    $$RRF\_Score(d) = \sum_{source \in \{text, table\}} \frac{1}{k + rank_{source}(d)}$$
  - Khử trùng lặp theo `(doc_name, page_number, content[:50])`.
  - Sắp xếp giảm dần theo `rrf_score`, trả về `top_n` ứng viên cao nhất.

### 5.6. `reranker.py`
* Class `Reranker`:
  - Tích hợp mô hình Cross-Encoder (ví dụ `BAAI/bge-reranker-v2-m3` hoặc module FlashRank/CrossEncoder nhẹ).
  - Có cơ chế fallback: Nếu chưa cài GPU hoặc model nặng chưa sẵn sàng, cho phép fallback xếp hạng theo RRF Score kèm warning.
  - Method `rerank(query: str, candidates: list[SearchCandidate], top_k: int = 5) -> list[SearchCandidate]`:
    So sánh độ tương quan ngữ nghĩa giữa `query` và `candidate.content`, trả về `top_k` kết quả tốt nhất.

### 5.7. `table_formatter.py`
* Function `format_table_to_markdown(table_json: dict) -> str`:
  - Chuyển cấu trúc `headers` và `rows` trong JSON thành chuỗi Markdown bảng hợp lệ:
    ```markdown
    | Header 1 | Header 2 |
    |---|---|
    | Val 1 | Val 2 |
    ```
  - Nếu payload đã có sẵn `table_markdown`, sử dụng trực tiếp.

### 5.8. `prompt_builder.py`
* Class `PromptBuilder`:
  - Nhận vào `query: str` và `candidates: list[SearchCandidate]`.
  - Xây dựng phần ngữ cảnh đánh số thứ tự rõ ràng:
    `[TÀI LIỆU 1] (Tài liệu: {doc_name} | Trang: {page_number} | Mục: {section})`
    `[BẢNG 1] (Tài liệu: {doc_name} | Trang: {page_number} | Bảng: {caption})`
  - Đóng gói System Prompt yêu cầu:
    - Trả lời trung thực dựa trên ngữ cảnh cung cấp.
    - Trích dẫn rõ `[Tài liệu X, Trang Y]` hoặc `[Bảng Z, Trang Y]`.
  - Trả về tuple `(system_prompt: str, user_prompt: str, citations: list[Citation])`.

### 5.9. `llm_client.py`
* Class `LLMClient`:
  - Sử dụng thư viện `openai` trỏ tới `LLM_BASE_URL`.
  - Gọi Chat Completion API với các tham số:
    - `model`: `LLM_MODEL_NAME`
    - `temperature`: 0.1
    - `extra_body`: `{"enable_thinking": False}` (Đảm bảo tắt tính năng thinking theo sơ đồ pipeline).
  - Method `generate(system_prompt: str, user_prompt: str) -> str`: Trả về chuỗi câu trả lời dạng văn bản thô.

### 5.10. `pipeline.py` (Orchestrator)
* Class `RAGPipeline`:
  - Khởi tạo và kết nối toàn bộ các thành phần trên.
  - Method `run(query: str) -> RAGResponse`:
    1. Embed câu hỏi.
    2. Search song song `text_index` và `table_index`.
    3. Hợp nhất bằng RRF.
    4. Tái xếp hạng bằng Reranker lấy Top 3–5.
    5. Định dạng văn bản và bảng (lấy ngữ cảnh gốc).
    6. Tạo prompt có gán nhãn trích dẫn.
    7. Gọi LLM sinh phản hồi.
    8. Trả về `RAGResponse(answer=..., citations=..., used_context=...)`.

### 5.11. `seed_mock.py`
* Script độc lập:
  - Kiểm tra và tạo 2 collections `text_index` và `table_index` trên Qdrant nếu chưa có.
  - Nạp 3 đoạn text mẫu và 2 bảng mẫu liên quan đến Báo cáo tài chính (doanh thu, lợi nhuận, bảng cân đối kế toán) kèm vector embedding thật từ `QueryEmbedder`.
  - Cho phép người phát triển chạy lệnh `python -m rag1.retrival.seed_mock` để sẵn sàng dữ liệu test.

---

## 6. HƯỚNG DẪN KIỂM THỬ ĐỘC LẬP (TESTING & RUN)

1. **Nạp dữ liệu mẫu:**
   ```powershell
   uv run python -m rag1.retrival.seed_mock
   ```
2. **Chạy thử một câu hỏi:**
   ```powershell
   uv run python -m rag1.retrival.pipeline "Doanh thu và lợi nhuận quý 3 năm 2025 là bao nhiêu?"
   ```
3. **Chạy Unit Test:**
   ```powershell
   uv run python -m unittest discover -s tests
   ```

The current retrieval code needs the changes in [INTEGRATION_INSTRUCTIONS.md](INTEGRATION_INSTRUCTIONS.md) before it can read the indexed payloads or use hybrid search.
