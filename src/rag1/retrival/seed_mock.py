"""Script nạp dữ liệu giả lập chuẩn schema vào Qdrant để kiểm thử độc lập."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag1.retrival.config import settings
from rag1.retrival.embedder import QueryEmbedder


def seed_mock_data() -> None:
    """Tạo collection và nạp dữ liệu giả lập cho text_index và table_index."""
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = QueryEmbedder.get_instance()

    # 1. Tự động lấy số chiều thực tế của model (1024) và khởi tạo collections
    sample_vec = embedder.encode_query("kiểm tra số chiều")
    vector_size = len(sample_vec)

    for col_name in [settings.text_collection, settings.table_collection]:
        # Nếu collection đã tồn tại nhưng sai số chiều thì xóa tạo lại
        if client.collection_exists(col_name):
            collection_info = client.get_collection(col_name)
            existing_dim = collection_info.config.params.vectors.size
            if existing_dim != vector_size:
                client.delete_collection(col_name)
                print(f"Đã xóa collection cũ sai số chiều: {col_name} ({existing_dim} -> {vector_size})")

        if not client.collection_exists(col_name):
            client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Đã tạo collection mới: {col_name} (kích thước {vector_size})")


    # 2. Dữ liệu mẫu văn bản (text_index)
    mock_texts = [
        {
            "id": 1,
            "content": "Doanh thu thuần hợp nhất quý 3 năm 2025 đạt 15.200 tỷ đồng, tăng trưởng 12% so với cùng kỳ năm 2024.",
            "doc_name": "Bao cao tai chinh hop nhat Q3.2025.pdf",
            "page_number": 5,
            "section": "3. Kết quả hoạt động kinh doanh",
        },
        {
            "id": 2,
            "content": "Lợi nhuận sau thuế thu nhập doanh nghiệp quý 3 đạt 1.850 tỷ đồng nhờ cải thiện biên lợi nhuận.",
            "doc_name": "Bao cao tai chinh hop nhat Q3.2025.pdf",
            "page_number": 6,
            "section": "3. Kết quả hoạt động kinh doanh",
        },
    ]

    text_points: list[PointStruct] = []
    for item in mock_texts:
        vector = embedder.encode_query(item["content"])
        text_points.append(
            PointStruct(
                id=item["id"],
                vector=vector,
                payload={
                    "schema_version": "1.0",
                    "record_type": "text",
                    "source": {"path": f"data/raw/{item['doc_name']}"},
                    "page_number": item["page_number"],
                    "section": item["section"],
                    "text": item["content"],
                    "parent_context": item["content"],
                },
            )
        )

    client.upsert(collection_name=settings.text_collection, points=text_points)
    print(f"Đã nạp {len(text_points)} bản ghi mẫu vào text_index!")

    # 3. Dữ liệu mẫu bảng biểu (table_index) - Đúng chuẩn payload thực tế của partner
    mock_tables = [
        {
            "id": "a48cab7a-64ad-5afe-aa99-b96a41bd01b5",
            "caption": "Ban kiểm soát và chức vụ",
            "text": "Họ tên\nChức vụ\nBà Nguyễn Thị Thu Hương\nTrưởng ban kiểm soát này\nÔng Thái Duy Nghĩa\nThành viên chuyên trách trị\nBà Nguyễn Thị Thu Nguyệt\nThành viên không chuyên trách thuận",
            "doc_path": "data/raw/Bao+cao+tai+chinh+hop+nhat+Q3.2025.pdf",
            "page_number": 4,
            "proposal_id": "p0004-r0005",
        }
    ]

    table_points: list[PointStruct] = []
    for tbl in mock_tables:
        vector = embedder.encode_query(f"{tbl['caption']} {tbl['text']}")
        table_points.append(
            PointStruct(
                id=tbl["id"],
                vector=vector,
                payload={
                    "schema_version": "1.0",
                    "record_type": "table",
                    "source": {"path": tbl["doc_path"]},
                    "page_number": tbl["page_number"],
                    "proposal": {"id": tbl["proposal_id"], "kind": "table"},
                    "text": tbl["text"],
                    "caption": tbl["caption"],
                    "table": {"structure_status": "pending", "structure": None},
                },
            )
        )

    client.upsert(collection_name=settings.table_collection, points=table_points)
    print(f"Đã nạp {len(table_points)} bảng biểu mẫu vào table_index!")


if __name__ == "__main__":
    seed_mock_data()
