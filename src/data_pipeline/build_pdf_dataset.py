"""
build_pdf_dataset.py
────────────────────
Tạo pdf_dataset.csv từ thesis_extracted.csv với schema y hệt merged_dataset.csv
(98 cột). KHÔNG đọc hay merge với khaosat_kltn.csv hay merged_dataset.csv.

Chỉ điền các cột thuộc nhóm "Sinh viên đã hoàn thành":
  - Identity: họ tên, mã SV, chuyên ngành, năm, tên đề tài, lĩnh vực, GV, điểm
  - Tech (41–62 + bản sao 71–92): tất cả tech columns từ Gemini extraction
  - Metadata: source_file, extraction_status, nhom_doi_tuong, nguon_du_lieu
Tất cả cột thuộc nhóm "đang thực hiện" / "sắp thực hiện" → None

Input:  data/processed/thesis_extracted.csv
Output: data/processed/pdf_dataset.csv

Run:
    python -m src.data_pipeline.build_pdf_dataset
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import THESIS_CSV, PROCESSED_DIR, LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "build_pdf_dataset.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

OUTPUT_CSV = PROCESSED_DIR / "pdf_dataset.csv"

# ── Schema 98 cột — y hệt header của merged_dataset.csv ──────
MERGED_COLUMNS = [
    # col 0–16: Nhóm "Đã hoàn thành"
    "Dấu thời gian",
    "1. Bạn thuộc nhóm đối tượng nào?",
    "Họ và tên",
    "Mã sinh viên",
    "Chuyên ngành",
    "Năm hoàn thành khóa luận",
    "Tên đề tài khóa luận đã thực hiện",
    "Đề tài của bạn thuộc lĩnh vực/định hướng chính nào?",
    "Giảng viên hướng dẫn",
    "Nhận xét về giảng viên hướng dẫn [Chuyên môn và mức độ phù hợp với đề tài]",
    "Nhận xét về giảng viên hướng dẫn [Khả năng định hướng và xây dựng đề tài]",
    "Nhận xét về giảng viên hướng dẫn [Mức độ hỗ trợ và tương tác]",
    "Nhận xét về giảng viên hướng dẫn [Chất lượng phản hồi và góp ý]",
    "Nhận xét về giảng viên hướng dẫn [Quản lý tiến độ và trách nhiệm hướng dẫn]",
    "Nhận xét về giảng viên hướng dẫn [Phong cách hướng dẫn và mức độ hài lòng]",
    "Điểm KLTN",
    "Nhận xét / Kinh nghiệm rút ra khi làm khóa luận",
    # col 17–31: Nhóm "Đang thực hiện" → None
    "Họ và tên.1",
    "Mã sinh viên.1",
    "Chuyên ngành.1",
    "GPA hiện tại",
    "Trạng thái KLTN",
    "Bạn quan tâm tới các lĩnh vực nào? (có thể chọn nhiều)",
    "Tự đánh giá mức độ thành thạo tổng quát theo từng nhóm lĩnh vực [Phát triển Web]",
    "Tự đánh giá mức độ thành thạo tổng quát theo từng nhóm lĩnh vực [Phát triển App (Mobile/Desktop)]",
    "Tự đánh giá mức độ thành thạo tổng quát theo từng nhóm lĩnh vực [Hệ thống tích hợp Web & App]",
    "Tự đánh giá mức độ thành thạo tổng quát theo từng nhóm lĩnh vực [AI / Học sâu / GenAI]",
    "Tự đánh giá mức độ thành thạo tổng quát theo từng nhóm lĩnh vực [Khoa học dữ liệu / Big Data / FinTech]",
    "Tự đánh giá mức độ thành thạo tổng quát theo từng nhóm lĩnh vực [Game / Đồ họa máy tính / VR-AR]",
    "Tự đánh giá mức độ thành thạo tổng quát theo từng nhóm lĩnh vực [An toàn thông tin / Cloud / Blockchain]",
    "Tự đánh giá mức độ thành thạo tổng quát theo từng nhóm lĩnh vực [IoT / Hệ thống nhúng / Digital Twin]",
    "Tự đánh giá mức độ thành thạo tổng quát theo từng nhóm lĩnh vực [Nghiên cứu khoa học / Thuật toán lý thuyết]",
    # col 32–40: Nhóm "Sắp thực hiện" (identity) → None
    "Họ và tên.2",
    "Mã sinh viên.2",
    "Chuyên ngành.2",
    "GPA hiện tại.1",
    "Tên đề tài khóa luận đã thực hiện.1",
    "Đề tài của bạn thuộc lĩnh vực/định hướng chính nào?.1",
    "Tên Đề Tài Khoá Luận",
    "Đề tài của bạn thuộc lĩnh vực/định hướng chính nào?.2",
    "Giảng viên hướng dẫn.1",
    # col 41–62: Tech block chính (điền từ PDF)
    "Ngôn ngữ lập trình Web đã dùng",
    "Frontend Framework/Library",
    "Backend Framework",
    "Database & Cache",
    "Ngôn ngữ lập trình App đã dùng",
    "Nền tảng / Framework App",
    "Local DB & Nền tảng Backend cho App",
    "Công nghệ phía Web & API Server",
    "Công nghệ phía Mobile / Client",
    "Kiến trúc kết nối & Hạ tầng",
    "Frameworks & Thư viện AI đã sử dụng",
    "Bài toán AI chính trong đề tài",
    "Công cụ phân tích & Xử lý Big Data",
    "Mô hình & Nghiệp vụ ứng dụng",
    "Game Engine & Đồ họa",
    "Thể loại game & Nền tảng đích",
    "Lĩnh vực chuyên môn đã thực hiện",
    "Công cụ & Môi trường sử dụng",
    "Phần cứng & Vi điều khiển",
    "Giao thức & Nền tảng IoT",
    "Phương pháp & Hướng nghiên cứu chính",
    "Công cụ & Kết quả đầu ra đề tài",
    # col 63–70: Preferences → None
    "Giảng viên mong muốn được hướng dẫn (Chọn từ 1 đến 3 GV)",
    "Mức độ quan trọng của các tiêu chí khi bạn chọn GVHD [Chuyên môn và mức độ phù hợp với đề tài]",
    "Mức độ quan trọng của các tiêu chí khi bạn chọn GVHD [Khả năng định hướng và xây dựng đề tài]",
    "Mức độ quan trọng của các tiêu chí khi bạn chọn GVHD [Mức độ hỗ trợ và tương tác]",
    "Mức độ quan trọng của các tiêu chí khi bạn chọn GVHD [Chất lượng phản hồi và góp ý]",
    "Mức độ quan trọng của các tiêu chí khi bạn chọn GVHD [Quản lý tiến độ và trách nhiệm hướng dẫn]",
    "Mức độ quan trọng của các tiêu chí khi bạn chọn GVHD [Phong cách hướng dẫn và mức độ hài lòng]",
    "Nguyện vọng hoặc đề xuất bổ sung khác",
    # col 71–92: Tech block bản sao .1 (điền giống 41–62)
    "Ngôn ngữ lập trình Web đã dùng.1",
    "Frontend Framework/Library.1",
    "Backend Framework.1",
    "Database & Cache.1",
    "Ngôn ngữ lập trình App đã dùng.1",
    "Nền tảng / Framework App.1",
    "Local DB & Nền tảng Backend cho App.1",
    "Công nghệ phía Web & API Server.1",
    "Công nghệ phía Mobile / Client.1",
    "Kiến trúc kết nối & Hạ tầng.1",
    "Frameworks & Thư viện AI đã sử dụng.1",
    "Bài toán AI chính trong đề tài.1",
    "Công cụ phân tích & Xử lý Big Data.1",
    "Mô hình & Nghiệp vụ ứng dụng.1",
    "Game Engine & Đồ họa.1",
    "Thể loại game & Nền tảng đích.1",
    "Lĩnh vực chuyên môn đã thực hiện.1",
    "Công cụ & Môi trường sử dụng.1",
    "Phần cứng & Vi điều khiển.1",
    "Giao thức & Nền tảng IoT.1",
    "Phương pháp & Hướng nghiên cứu chính.1",
    "Công cụ & Kết quả đầu ra đề tài.1",
    # col 93–97: Metadata
    "source_file",
    "extraction_status",
    "extraction_notes",
    "nhom_doi_tuong",
    "nguon_du_lieu",
]

# ── Mapping: thesis_extracted col → (vi_col, vi_col_dup) ─────
# Tech columns điền vào CẢ HAI block (41–62 và 71–92)
TECH_MAP = {
    "web_languages":       ("Ngôn ngữ lập trình Web đã dùng",        "Ngôn ngữ lập trình Web đã dùng.1"),
    "frontend_frameworks": ("Frontend Framework/Library",             "Frontend Framework/Library.1"),
    "backend_frameworks":  ("Backend Framework",                      "Backend Framework.1"),
    "database_cache":      ("Database & Cache",                       "Database & Cache.1"),
    "app_languages":       ("Ngôn ngữ lập trình App đã dùng",        "Ngôn ngữ lập trình App đã dùng.1"),
    "app_frameworks":      ("Nền tảng / Framework App",               "Nền tảng / Framework App.1"),
    "app_db_backend":      ("Local DB & Nền tảng Backend cho App",   "Local DB & Nền tảng Backend cho App.1"),
    "web_api_tech":        ("Công nghệ phía Web & API Server",        "Công nghệ phía Web & API Server.1"),
    "mobile_client_tech":  ("Công nghệ phía Mobile / Client",         "Công nghệ phía Mobile / Client.1"),
    "architecture":        ("Kiến trúc kết nối & Hạ tầng",           "Kiến trúc kết nối & Hạ tầng.1"),
    "ai_frameworks":       ("Frameworks & Thư viện AI đã sử dụng",   "Frameworks & Thư viện AI đã sử dụng.1"),
    "ai_problems":         ("Bài toán AI chính trong đề tài",         "Bài toán AI chính trong đề tài.1"),
    "data_tools":          ("Công cụ phân tích & Xử lý Big Data",    "Công cụ phân tích & Xử lý Big Data.1"),
    "data_models":         ("Mô hình & Nghiệp vụ ứng dụng",          "Mô hình & Nghiệp vụ ứng dụng.1"),
    "game_engine":         ("Game Engine & Đồ họa",                   "Game Engine & Đồ họa.1"),
    "game_type":           ("Thể loại game & Nền tảng đích",          "Thể loại game & Nền tảng đích.1"),
    "specialty_field":     ("Lĩnh vực chuyên môn đã thực hiện",      "Lĩnh vực chuyên môn đã thực hiện.1"),
    "tools_environment":   ("Công cụ & Môi trường sử dụng",          "Công cụ & Môi trường sử dụng.1"),
    "hardware":            ("Phần cứng & Vi điều khiển",              "Phần cứng & Vi điều khiển.1"),
    "iot_protocol":        ("Giao thức & Nền tảng IoT",               "Giao thức & Nền tảng IoT.1"),
    "research_methods":    ("Phương pháp & Hướng nghiên cứu chính",  "Phương pháp & Hướng nghiên cứu chính.1"),
    "research_output":     ("Công cụ & Kết quả đầu ra đề tài",       "Công cụ & Kết quả đầu ra đề tài.1"),
}


def _build_row(src: dict) -> dict:
    """Map 1 record thesis_extracted → dict 98 cột của merged_dataset schema."""
    row = {col: None for col in MERGED_COLUMNS}

    # ── Nhóm "Đã hoàn thành" (col 0–16) ─────────────────────
    row["Dấu thời gian"]                          = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    row["1. Bạn thuộc nhóm đối tượng nào?"]       = "Sinh viên đã hoàn thành khóa luận tốt nghiệp"
    row["Họ và tên"]                               = src.get("student_name")
    row["Mã sinh viên"]                            = src.get("student_id")
    row["Chuyên ngành"]                            = src.get("major")
    row["Năm hoàn thành khóa luận"]                = src.get("completion_year")
    row["Tên đề tài khóa luận đã thực hiện"]       = src.get("thesis_title")
    row["Đề tài của bạn thuộc lĩnh vực/định hướng chính nào?"] = src.get("field_category")
    row["Giảng viên hướng dẫn"]                    = src.get("advisor_name")
    row["Điểm KLTN"]                               = src.get("thesis_grade")

    # ── Tech (col 41–62 và bản sao 71–92) ────────────────────
    for src_col, (vi_col, vi_col_dup) in TECH_MAP.items():
        val = src.get(src_col)
        row[vi_col]     = val
        row[vi_col_dup] = val

    # ── Metadata (col 93–97) ──────────────────────────────────
    row["source_file"]       = src.get("source_file")
    row["extraction_status"] = src.get("extraction_status")
    row["extraction_notes"]  = src.get("extraction_notes")
    row["nhom_doi_tuong"]    = "Sinh viên đã hoàn thành khóa luận tốt nghiệp"
    row["nguon_du_lieu"]     = "PDF extraction"

    return row


def run() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not THESIS_CSV.exists():
        logger.error(
            f"thesis_extracted.csv không tìm thấy tại {THESIS_CSV}.\n"
            "Chạy: python -m src.data_pipeline.clean_data trước."
        )
        sys.exit(1)

    thesis_df = pd.read_csv(THESIS_CSV)
    total = len(thesis_df)
    status_counts = thesis_df["extraction_status"].value_counts().to_dict()
    logger.info(f"Đọc {total} dòng từ thesis_extracted.csv — {status_counts}")

    # Chỉ lấy dòng extract thành công
    ok_df = thesis_df[thesis_df["extraction_status"] == "success"].copy()
    if ok_df.empty:
        logger.warning("Không có dòng nào extraction_status=success. File không được tạo.")
        return pd.DataFrame(columns=MERGED_COLUMNS)

    logger.info(f"Xử lý {len(ok_df)} dòng thành công...")
    rows = [_build_row(r) for r in ok_df.to_dict(orient="records")]
    out_df = pd.DataFrame(rows, columns=MERGED_COLUMNS)

    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Đã tạo pdf_dataset.csv: {len(out_df)} dòng, {len(MERGED_COLUMNS)} cột → {OUTPUT_CSV}")

    # Thống kê nhanh: cột nào có dữ liệu
    print(f"\n{'='*60}")
    print(f"PDF DATASET  |  {len(out_df)} dòng  |  {len(MERGED_COLUMNS)} cột")
    print(f"{'='*60}")
    print(f"{'Cột':<55} {'Có data':>7}")
    print("-" * 64)
    for col in MERGED_COLUMNS:
        cnt = out_df[col].notna().sum()
        if cnt > 0:
            bar = "█" * int(cnt / len(out_df) * 20)
            print(f"  {col[:52]:<52} {cnt:3d}/{len(out_df)}  {bar}")

    return out_df


if __name__ == "__main__":
    run()
