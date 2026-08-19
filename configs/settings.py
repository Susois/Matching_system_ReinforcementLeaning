"""
Centralized configuration for the Student-Advisor RL project.
All paths, API keys and constants live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Root paths ────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR    = ROOT_DIR / "logs"

# Input
PDF_INPUT_DIR  = DATA_DIR / "1.pdfs"   # raw PDFs
DOCX_INPUT_DIR = DATA_DIR / "2.docx"   # raw DOCX

# Output
PROCESSED_DIR        = DATA_DIR / "processed"
EMBEDDINGS_DIR       = DATA_DIR / "embeddings"
RESULTS_DIR          = OUTPUT_DIR / "results"
CLEAN_DATA_DIR       = ROOT_DIR / "clean_data"

# Key output files
THESIS_CSV           = PROCESSED_DIR / "thesis_extracted.csv"
ADVISOR_SKILLS_CSV   = PROCESSED_DIR / "advisor_skills.csv"
ADVISOR_PROFILE_CSV  = PROCESSED_DIR / "advisor_profiles.csv"
SURVEY_CSV           = CLEAN_DATA_DIR / "khaosat_kltn.csv"

# ── Gemini API ────────────────────────────────────────────────
GOOGLE_API_KEY       = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL         = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_MAX_RETRIES   = 3
GEMINI_RETRY_DELAY   = 2          # seconds between retries
GEMINI_TEMPERATURE   = 0.1
GEMINI_MAX_TOKENS    = 4096
PDF_TEXT_LIMIT       = 12000      # chars sent to Gemini per PDF

# ── Processing ────────────────────────────────────────────────
BATCH_SIZE           = 5          # parallel workers (ThreadPoolExecutor)
MAX_PDF_PAGES        = None       # None = all pages

# ── CSV columns (must match khaosat_kltn.csv — Section "Đã hoàn thành") ──────
#   We only write the columns relevant to completed-thesis students extracted
#   from PDF.  Survey columns for "sắp làm" / "đang làm" are left as NaN.
THESIS_CSV_COLUMNS = [
    # ── Identity
    "student_id",
    "student_name",
    "major",
    "completion_year",
    # ── Thesis
    "thesis_title",
    "field_category",
    "advisor_name",
    "thesis_grade",
    # ── Web
    "web_languages",
    "frontend_frameworks",
    "backend_frameworks",
    "database_cache",
    "web_api_tech",
    # ── App / Mobile
    "app_languages",
    "app_frameworks",
    "app_db_backend",
    "mobile_client_tech",
    # ── Infrastructure
    "architecture",
    # ── AI / ML
    "ai_frameworks",
    "ai_problems",
    # ── Data Science
    "data_tools",
    "data_models",
    # ── Game
    "game_engine",
    "game_type",
    # ── Security / IoT / Research
    "specialty_field",
    "tools_environment",
    "hardware",
    "iot_protocol",
    "research_methods",
    "research_output",
    # ── Metadata
    "source_file",
    "extraction_status",   # success | failed | skipped
    "extraction_notes",
]

# Mapping: CSV column → Vietnamese alias (used when writing merged CSV)
COL_VI_ALIAS = {
    "student_id":           "Mã sinh viên",
    "student_name":         "Họ và tên",
    "major":                "Chuyên ngành",
    "completion_year":      "Năm hoàn thành khóa luận",
    "thesis_title":         "Tên đề tài khóa luận đã thực hiện",
    "field_category":       "Đề tài của bạn thuộc lĩnh vực/định hướng chính nào?",
    "advisor_name":         "Giảng viên hướng dẫn",
    "thesis_grade":         "Điểm KLTN",
    "web_languages":        "Ngôn ngữ lập trình Web đã dùng",
    "frontend_frameworks":  "Frontend Framework/Library",
    "backend_frameworks":   "Backend Framework",
    "database_cache":       "Database & Cache",
    "web_api_tech":         "Công nghệ phía Web & API Server",
    "app_languages":        "Ngôn ngữ lập trình App đã dùng",
    "app_frameworks":       "Nền tảng / Framework App",
    "app_db_backend":       "Local DB & Nền tảng Backend cho App",
    "mobile_client_tech":   "Công nghệ phía Mobile / Client",
    "architecture":         "Kiến trúc kết nối & Hạ tầng",
    "ai_frameworks":        "Frameworks & Thư viện AI đã sử dụng",
    "ai_problems":          "Bài toán AI chính trong đề tài",
    "data_tools":           "Công cụ phân tích & Xử lý Big Data",
    "data_models":          "Mô hình & Nghiệp vụ ứng dụng",
    "game_engine":          "Game Engine & Đồ họa",
    "game_type":            "Thể loại game & Nền tảng đích",
    "specialty_field":      "Lĩnh vực chuyên môn đã thực hiện",
    "tools_environment":    "Công cụ & Môi trường sử dụng",
    "hardware":             "Phần cứng & Vi điều khiển",
    "iot_protocol":         "Giao thức & Nền tảng IoT",
    "research_methods":     "Phương pháp & Hướng nghiên cứu chính",
    "research_output":      "Công cụ & Kết quả đầu ra đề tài",
}

# ── Advisor skill columns ─────────────────────────────────────
SKILL_TECH_COLS = [
    "web_languages", "frontend_frameworks", "backend_frameworks",
    "database_cache", "web_api_tech", "app_languages", "app_frameworks",
    "app_db_backend", "mobile_client_tech", "architecture",
    "ai_frameworks", "ai_problems", "data_tools", "data_models",
    "game_engine", "game_type", "specialty_field", "tools_environment",
    "hardware", "iot_protocol", "research_methods", "research_output",
]

# Field category labels (for classification)
FIELD_LABELS = [
    "Phát triển Web",
    "Phát triển App (Mobile/Desktop)",
    "Hệ thống tích hợp đa nền tảng",
    "AI / Học sâu / GenAI",
    "Khoa học dữ liệu / Big Data",
    "Game / Đồ họa",
    "An toàn thông tin / Cloud",
    "IoT / Hệ thống nhúng",
    "Nghiên cứu khoa học",
]
