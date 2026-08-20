# Hệ thống RL Phân bổ Sinh viên – Giảng viên Hướng dẫn

> **Reinforcement Learning-Based Student–Advisor Matching System**


---

## Giới thiệu

Đề tài xây dựng hệ thống hỗ trợ phân bổ tối ưu sinh viên – giảng viên hướng dẫn khóa luận tốt nghiệp bằng **Reinforcement Learning**, kết hợp:

- **NLP / Embedding** để biểu diễn nội dung khóa luận và hồ sơ nghiên cứu giảng viên
- **Matching Simulator** xây từ dữ liệu khóa luận lịch sử để train RL offline
- **PPO** (mô hình chính) và **DQN** (đối chứng) tối ưu assignment toàn cohort
- **Continuous Learning** — candidate model V2 được tạo và đánh giá sau mỗi cohort mới
- **Web Application** cho Student / Advisor / Admin

**Điểm phân biệt với Greedy/Gale-Shapley:** RL học policy có thể cải thiện theo thời gian khi thu thập thêm feedback, không chỉ tối ưu một lần.

---

## Cấu trúc dự án

```
student-advisor-rl/
│
├── data/
│   ├── 1.pdfs/                    ← ~200 file khóa luận PDF/DOCX (input)
│   ├── processed/                 ← CSV/JSON đầu ra của pipeline
│   │   ├── thesis_extracted.csv   ← thông tin trích từ PDF (Gemini/LLM)
│   │   ├── pdf_dataset.csv        ← dataset chuẩn hóa (schema khảo sát)
│   │   ├── advisor_skills.csv     ← skill tổng hợp từ PDF
│   │   ├── advisor_profiles.csv   ← hồ sơ GV đầy đủ + rating
│   │   ├── advisor_research_profiles.json  ← research profile từ crawler
│   │   └── advisor_skills_extracted.csv    ← skill + evidence + strength
│   ├── raw/
│   │   ├── lecturers_list.json    ← danh sách 29 GV (slug, URL)
│   │   ├── lecturers_raw.json     ← raw HTML data của tất cả GV
│   │   └── profiles/              ← raw profile từng GV (<slug>.json)
│   └── embeddings/                ← vector embeddings (Phase 2)
│
├── src/
│   ├── data_pipeline/             ← Phase 1: thu thập & làm sạch dữ liệu PDF
│   │   ├── pdf_ocr.py             ← pypdf + OCR fallback
│   │   ├── llm_extractor.py       ← LLM extraction (9router / Gemini) với model rotation
│   │   ├── clean_data.py          ← pipeline chính: PDF → thesis_extracted.csv
│   │   ├── build_pdf_dataset.py   ← chuẩn hóa schema → pdf_dataset.csv
│   │   ├── build_advisor_skills.py ← tổng hợp skill GV từ PDF data
│   │   ├── crawl_advisor_data.py  ← enrich từ Semantic Scholar (tùy chọn)
│   │   └── run_pipeline.py        ← entry point: chạy toàn bộ pipeline
│   │
│   ├── crawler/                   ← Phase 1b: crawl website fit.neu.edu.vn
│   │   ├── lecturer_list_crawler.py  ← Step 1: lấy danh sách GV
│   │   ├── lecturer_detail_crawler.py ← Step 2: crawl từng profile (raw)
│   │   ├── skill_extractor.py     ← Step 3: taxonomy match + evidence + strength
│   │   └── pipeline.py            ← entry point crawler
│   │
│   ├── nlp/                       ← Phase 2: embedding & similarity (TODO)
│   ├── environment/               ← Phase 3: Gymnasium matching env (TODO)
│   ├── rl/
│   │   ├── ppo/                   ← PPO — mô hình RL chính (TODO)
│   │   └── dqn/                   ← DQN — mô hình RL đối chứng (TODO)
│   ├── baselines/                 ← Random, Greedy, Gale-Shapley, SPA (TODO)
│   ├── simulator/                 ← Matching Simulator (TODO)
│   └── model_registry/            ← Model versioning, evaluation gate (TODO)
│
├── configs/
│   ├── settings.py                ← tất cả path, hằng số, schema CSV
│   └── advisor_overrides.json     ← thông tin GV bổ sung thủ công
│
├── clean_data/
│   └── khaosat_kltn.csv           ← khảo sát Google Forms (input tham khảo)
│
├── outputs/
│   ├── results/                   ← kết quả thí nghiệm
│   ├── figures/                   ← biểu đồ
│   └── models/                    ← checkpoint RL
│
├── notebooks/                     ← EDA, NLP analysis, experiment results
├── backend/                       ← FastAPI service (Phase 4)
├── frontend/                      ← React + TypeScript (Phase 4)
├── tests/
├── .env.example
└── requirements.txt
```

---

## Cài đặt

```bash
# Cài dependencies
pip install -r requirements.txt

# Cài Playwright browsers (dùng cho crawler)
python -m playwright install chromium

# Tạo .env
copy .env.example .env
# Điền NINEROUTER_BASE_URL, NINEROUTER_API_KEY, NINEROUTER_MODEL
```

### Cấu hình `.env`

```env
# LLM Provider: "openai_compat" (9router) hoặc "gemini"
LLM_PROVIDER=openai_compat

# 9router endpoint
NINEROUTER_BASE_URL=http://localhost:20128/v1
NINEROUTER_API_KEY=sk-...
NINEROUTER_MODEL=kr/claude-haiku-4.5

# Model rotation — tự động khi model bị limit
# NINEROUTER_MODEL_LIST=kr/claude-haiku-4.5,ag/gemini-3-flash,...
```

---

## Phase 1 — Data Pipeline (đang thực hiện)

### A. Trích xuất thông tin từ PDF khóa luận

```bash
# Chạy toàn bộ (extract → build dataset → skill profiles)
python -m src.data_pipeline.run_pipeline --no-crawl

# Test với 5 file trước
python -m src.data_pipeline.run_pipeline --no-crawl --limit 5

# Re-process tất cả (kể cả đã xong)
python -m src.data_pipeline.run_pipeline --no-crawl --force

# Từng bước riêng
python -m src.data_pipeline.clean_data          # PDF → thesis_extracted.csv
python -m src.data_pipeline.build_pdf_dataset   # → pdf_dataset.csv
python -m src.data_pipeline.build_advisor_skills # → advisor_skills.csv
```

**LLM model rotation:** Khi một model bị rate-limit hoặc từ chối, pipeline tự động chuyển sang model tiếp theo trong `NINEROUTER_MODEL_LIST`. Progress được lưu sau mỗi file — ngắt bất cứ lúc nào không mất dữ liệu.

### B. Crawl hồ sơ giảng viên từ fit.neu.edu.vn

```bash
# Chạy full pipeline crawler (3 bước)
python -m src.crawler.pipeline

# Từng bước riêng
python -m src.crawler.pipeline --step 1   # lấy danh sách 29 GV
python -m src.crawler.pipeline --step 2   # crawl từng profile
python -m src.crawler.pipeline --step 3   # extract skill + research topics

# Crawl một GV cụ thể
python -m src.crawler.pipeline --step 2 --slug ts-pham-xuan-lam
python -m src.crawler.pipeline --step 3 --slug ts-pham-xuan-lam
```

**Incremental crawl:** Bổ sung GV vào `data/raw/lecturers_list.json` rồi chạy lại `--step 2` — chỉ crawl người chưa có file trong `data/raw/profiles/`.

---

## Ghi chú học thuật

**Về vai trò của RL:** Nếu thiếu grade/feedback thực tế, reward được xây từ compatibility + fairness. RL có giá trị hơn Greedy/Gale-Shapley ở điểm **Continuous Learning** — policy cải thiện theo cohort mới, không chỉ tối ưu một lần.

**Metric đánh giá (không cần grade):**
- Compatibility score (cosine similarity thesis ↔ advisor research)
- Recall@K / MRR so với historical assignment
- Load variance & quota violations
- So sánh PPO vs DQN vs Greedy vs Gale-Shapley trên cùng dataset
