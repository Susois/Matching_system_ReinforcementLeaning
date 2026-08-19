# Hệ thống RL Phân bổ Sinh viên – Giảng viên Hướng dẫn

> **Reinforcement Learning-Based Student–Advisor Matching System**  
> Sử dụng dữ liệu khóa luận lịch sử, NLP/Embedding, PPO, DQN và Continuous Learning.

---

## Giới thiệu

Đây là source code cho khóa luận tốt nghiệp nghiên cứu bài toán **phân bổ tối ưu sinh viên – giảng viên hướng dẫn** bằng Reinforcement Learning.

Thay vì gán thủ công hoặc dùng greedy matching, hệ thống mô hình hóa bài toán thành **Sequential Decision Making**: RL agent chọn assignment cho từng sinh viên trong cohort, môi trường trả reward dựa trên độ phù hợp chủ đề, nguyện vọng, quota và fairness. Dữ liệu khóa luận lịch sử được dùng để xây dựng **Matching Simulator** — nơi agent được train an toàn trước khi đưa ra hỗ trợ phân công thật.

**Công nghệ cốt lõi:** Python · PyTorch · Stable-Baselines3 (PPO / DQN) · Gymnasium · sentence-transformers · FastAPI · React + TypeScript · PostgreSQL · Docker

---

## Cấu trúc dự án

```
student-advisor-rl/
├── data/
│   ├── *.pdf / *.docx            ← file khóa luận gốc (input)
│   ├── processed/                ← CSV đầu ra của pipeline
│   └── embeddings/               ← vector embedding GV / SV
│
├── src/
│   ├── data_pipeline/            ← Giai đoạn 1: thu thập & làm sạch dữ liệu
│   │   ├── pdf_ocr.py            ← trích text từ PDF (pypdf → OCR fallback)
│   │   ├── gemini_extractor.py   ← gọi Gemini API, trả về JSON có cấu trúc
│   │   ├── clean_data.py         ← pipeline chính: PDF → thesis_extracted.csv
│   │   ├── build_advisor_skills.py  ← tổng hợp skill profile của giảng viên
│   │   ├── crawl_advisor_data.py ← crawl Semantic Scholar, enrich hồ sơ GV
│   │   ├── merge_to_survey.py    ← gộp với khaosat_kltn.csv
│   │   └── run_pipeline.py       ← chạy toàn bộ pipeline một lệnh
│   │
│   ├── nlp/                      ← Giai đoạn 2: embedding & similarity
│   ├── environment/              ← Giai đoạn 3: Gymnasium matching env
│   ├── rl/
│   │   ├── ppo/                  ← PPO — mô hình RL chính
│   │   └── dqn/                  ← DQN — mô hình RL đối chứng
│   ├── baselines/                ← Random, Greedy, Gale-Shapley, SPA
│   └── simulator/                ← Matching Simulator (train offline)
│
├── configs/
│   ├── settings.py               ← tất cả path, hằng số, schema CSV
│   └── advisor_overrides.json    ← thông tin GV bổ sung thủ công
│
├── clean_data/
│   └── khaosat_kltn.csv          ← dữ liệu khảo sát (xuất từ Google Forms)
│
├── outputs/
│   ├── results/                  ← kết quả thí nghiệm (CSV)
│   ├── figures/                  ← biểu đồ
│   └── models/                   ← checkpoint RL
│
├── notebooks/                    ← EDA, phân tích NLP, kết quả thí nghiệm
├── backend/                      ← FastAPI service
├── frontend/                     ← React + TypeScript
├── tests/
├── .env.example                  ← mẫu biến môi trường
└── requirements.txt
```

---

## Cài đặt

### Yêu cầu
- Python 3.10+
- Google Gemini API key ([lấy tại đây](https://aistudio.google.com/app/apikey))

### Các bước

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Tạo file .env từ mẫu
copy .env.example .env
# Mở .env và điền GOOGLE_API_KEY=<your-key>
```

### OCR (tùy chọn — chỉ cần nếu PDF bị scan)

```bash
pip install pdf2image pytesseract Pillow
# Cài Tesseract binary: https://github.com/tesseract-ocr/tesseract
# Tải thêm gói tiếng Việt: vie.traineddata
```

---

## Giai đoạn 1 — Data Pipeline

### Đặt file PDF vào thư mục `data/`

```
data/
├── 11221234_NguyenVanA_TenDeTai.pdf
├── 11221235_TranThiB_TenDeTai.pdf
└── ...
```

### Chạy pipeline

```bash
# Chạy toàn bộ (extract → skill profiles → merge → crawl)
python -m src.data_pipeline.run_pipeline

# Bỏ qua bước crawl web (nhanh hơn, không cần mạng)
python -m src.data_pipeline.run_pipeline --no-crawl

# Test với 3 file trước khi chạy hết
python -m src.data_pipeline.run_pipeline --limit 3 --no-crawl

# Re-process toàn bộ kể cả file đã xử lý
python -m src.data_pipeline.run_pipeline --force --no-crawl
```

### Chạy từng bước riêng lẻ

```bash
# Bước 1: trích xuất thông tin từ PDF → thesis_extracted.csv
python -m src.data_pipeline.clean_data

# Bước 2: tổng hợp skill của giảng viên
python -m src.data_pipeline.build_advisor_skills

# Bước 3: gộp với dữ liệu khảo sát
python -m src.data_pipeline.merge_to_survey

# Bước 4: crawl thêm dữ liệu GV từ Semantic Scholar
python -m src.data_pipeline.crawl_advisor_data

# Crawl cho một GV cụ thể
python -m src.data_pipeline.crawl_advisor_data --advisor "Phạm Xuân Lâm"
```

### Kết quả đầu ra

| File | Nội dung |
|------|----------|
| `data/processed/thesis_extracted.csv` | Mỗi dòng là 1 khóa luận, đầy đủ các trường công nghệ |
| `data/processed/advisor_skills.csv` | Tổng hợp skill tech của từng GV (comma-separated, sort theo tần suất) |
| `data/processed/advisor_profiles.csv` | Hồ sơ đầy đủ: số lượng SV, điểm TB, phân bổ lĩnh vực, skill_score, rating |
| `data/processed/merged_dataset.csv` | Gộp PDF extraction + khảo sát Google Forms |
| `data/processed/advisor_crawled.csv` | Dữ liệu thô từ Semantic Scholar (h-index, papers, citations) |
| `logs/pipeline.log` | Log toàn bộ quá trình xử lý |

---

## Luồng xử lý dữ liệu

```
data/*.pdf / *.docx
    │
    ├─ pdf_ocr.py
    │   ├─ pypdf (native text extraction)
    │   └─ pytesseract OCR fallback (nếu PDF scan, < 300 ký tự)
    │
    ├─ gemini_extractor.py
    │   └─ Gemini API → JSON có cấu trúc (30 fields)
    │
    ├─ clean_data.py
    │   └─ thesis_extracted.csv  (1 dòng / khóa luận)
    │
    ├─ build_advisor_skills.py
    │   ├─ advisor_skills.csv    (skill tech tổng hợp)
    │   └─ advisor_profiles.csv  (hồ sơ đầy đủ + rating từ khảo sát)
    │
    ├─ merge_to_survey.py
    │   └─ merged_dataset.csv    (PDF + Google Forms)
    │
    └─ crawl_advisor_data.py
        ├─ Semantic Scholar API  (h-index, paper count, citations)
        ├─ Google Scholar        (best-effort)
        ├─ advisor_overrides.json (thủ công)
        └─ advisor_profiles.csv  (updated với skill_score 0–100)
```

---

## Điểm skill của Giảng viên (`skill_score`)

Mỗi giảng viên được tính điểm tổng hợp 0–100 từ các nguồn:

| Thành phần | Điểm tối đa | Nguồn |
|---|---|---|
| h-index | 30 | Semantic Scholar |
| Số lượng bài báo | 30 | Semantic Scholar |
| Số lượt trích dẫn | 20 | Semantic Scholar |
| Đánh giá của sinh viên | 20 | khaosat_kltn.csv |

Điểm đánh giá của sinh viên lấy trung bình 6 tiêu chí: chuyên môn, định hướng đề tài, hỗ trợ tương tác, chất lượng phản hồi, quản lý tiến độ, phong cách hướng dẫn.

Để bổ sung thông tin thủ công cho một GV (homepage, research interests, Google Scholar URL), chỉnh sửa `configs/advisor_overrides.json`.

---

## Schema CSV

`thesis_extracted.csv` dùng cùng tên cột với `khaosat_kltn.csv`. Các trường không tìm thấy trong PDF để là `None` / `NaN` — **không bao giờ tự điền giả**.

| Nhóm | Trường | Ví dụ |
|------|--------|-------|
| Sinh viên | `student_id`, `student_name`, `major`, `completion_year` | `11221234` |
| Khóa luận | `thesis_title`, `field_category`, `advisor_name`, `thesis_grade` | `AI / Học sâu / GenAI` |
| Web | `web_languages`, `frontend_frameworks`, `backend_frameworks`, `database_cache` | `React.js`, `FastAPI` |
| App | `app_languages`, `app_frameworks`, `app_db_backend`, `mobile_client_tech` | `Flutter`, `SQLite` |
| Hạ tầng | `architecture` | `Docker, Microservices` |
| AI/ML | `ai_frameworks`, `ai_problems` | `PyTorch, YOLO`, `Object Detection` |
| Data | `data_tools`, `data_models` | `Pandas, Power BI` |
| Game | `game_engine`, `game_type` | `Unity (C#)`, `Game 2D Mobile` |
| IoT/Security | `specialty_field`, `hardware`, `iot_protocol` | `Arduino`, `MQTT` |
| Nghiên cứu | `research_methods`, `research_output` | `LaTeX, Journal Paper` |
| Meta | `source_file`, `extraction_status`, `extraction_notes` | `success` / `failed` |

---

## Lộ trình phát triển

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 1. Data Pipeline | PDF OCR + Gemini extraction + advisor skill profiles | 🔄 Đang thực hiện |
| 2. NLP / Embedding | sentence-transformers, BGE, FAISS similarity | ⏳ Tiếp theo |
| 3. Matching Environment | Gymnasium env, state/action/reward, action masking | ⏳ |
| 4. RL Training | PPO (chính) + DQN (đối chứng), Stable-Baselines3 | ⏳ |
| 5. Baselines | Random, Greedy, Gale-Shapley, SPA | ⏳ |
| 6. Continuous Learning | Candidate V2, Evaluation Gate, Rollback | ⏳ |
| 7. Web Application | FastAPI + React + PostgreSQL + Docker | ⏳ |
| 8. Thực nghiệm & Viết | E1–E10, ablation, thesis | ⏳ |

**Deadline:** 20/11/2026
