"""
lecturer_detail_crawler.py
──────────────────────────
Tầng 2: Crawl chi tiết từng giảng viên — raw data, KHÔNG extract skill.

Input:  data/raw/lecturers_list.json
Output: data/raw/profiles/<slug>.json  — raw profile per lecturer
        data/raw/lecturers_raw.json    — tất cả profiles gộp lại

Schema mỗi profile:
{
  "slug":             "ts-pham-xuan-lam",
  "name":             "Phạm Xuân Lâm",
  "title":            "TS",
  "profile_url":      "https://fit.neu.edu.vn/lecturer/ts-pham-xuan-lam",
  "department":       "Khoa Công nghệ thông tin",
  "email":            "lampx@neu.edu.vn",
  "phone":            "...",
  "bio_raw":          "...",        ← text thô phần Giới thiệu
  "education_raw":    ["..."],      ← raw strings, chưa parse
  "work_history_raw": ["..."],
  "teaching_raw":     ["Lập trình Java", "Thiết kế Web", ...],
  "research_areas_raw": ["Công nghệ giáo dục", ...],
  "publications_raw": [            ← raw strings, chưa parse author/year
    "Chu Văn Huy, ... (2025). Title...",
    ...
  ],
  "crawled_at":       "2026-08-20T..."
}

Run:
    python -m src.crawler.lecturer_detail_crawler
    python -m src.crawler.lecturer_detail_crawler --slug ts-pham-xuan-lam
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_DIR      = ROOT / "data" / "raw"
PROFILES_DIR = RAW_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
LIST_JSON    = RAW_DIR / "lecturers_list.json"
ALL_RAW_JSON = RAW_DIR / "lecturers_raw.json"

BASE_URL = "https://fit.neu.edu.vn"
DELAY_MS = 1500   # polite delay between pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── HTML → structured raw sections ───────────────────────────
def _extract_profile(html: str, meta: dict) -> dict:
    """Parse detail page HTML into raw profile dict (no skill inference)."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    profile = {
        "slug":               meta.get("slug", ""),
        "name":               meta.get("name", ""),
        "title":              meta.get("title", ""),
        "profile_url":        meta.get("profile_url", ""),
        "department":         meta.get("department", "Khoa Công nghệ thông tin"),
        "email":              None,
        "phone":              None,
        "bio_raw":            "",
        "education_raw":      [],
        "work_history_raw":   [],
        "teaching_raw":       [],
        "research_areas_raw": [],
        "publications_raw":   [],
        "crawled_at":         datetime.now().isoformat(),
    }

    # Email & phone
    for line in lines:
        if "@" in line and "Email" in line:
            m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", line)
            if m:
                profile["email"] = m.group()
        if "Số điện thoại" in line or "Phone" in line:
            m = re.search(r"[\d\s\-+()]{8,}", line)
            if m:
                profile["phone"] = m.group().strip()

    # Section detection — walk lines looking for section headers
    SECTION_HEADERS = {
        "GIỚI THIỆU":          "bio",
        "GIỚI THIỆU CHUNG":    "bio",
        "QUÁ TRÌNH ĐÀO TẠO":  "education",
        "QUÁ TRÌNH CÔNG TÁC":  "work",
        "GIẢNG DẠY":           "teaching",
        "LĨNH VỰC NGHIÊN CỨU":"research",
        "CÔNG TRÌNH KHOA HỌC": "publications",
        "LIÊN HỆ":             "contact",
    }

    current_section = None
    pub_buffer: list[str] = []

    def _flush_pub():
        if pub_buffer:
            profile["publications_raw"].append(" ".join(pub_buffer).strip())
            pub_buffer.clear()

    for line in lines:
        upper = line.upper().strip()

        # Check section header
        matched = False
        for header, sec in SECTION_HEADERS.items():
            if upper == header or upper.startswith(header):
                _flush_pub()
                current_section = sec
                matched = True
                break
        if matched:
            continue

        # Skip nav / boilerplate lines
        if len(line) < 3:
            continue
        if any(nav in line for nav in [
            "Khoa Công nghệ thông tin", "Đại học Kinh tế Quốc dân",
            "Faculty of Information Technology", "ICETAI",
            "Mobile App", "Email\n", "Courses", "Tín chỉ",
        ]):
            continue

        # Route to section
        if current_section == "bio":
            profile["bio_raw"] += line + "\n"

        elif current_section == "education":
            if line.startswith("-") or re.match(r"\d{4}", line):
                profile["education_raw"].append(line.lstrip("- ").strip())

        elif current_section == "work":
            if line.startswith("-") or re.match(r"\d{4}|Từ", line):
                profile["work_history_raw"].append(line.lstrip("- ").strip())

        elif current_section == "teaching":
            profile["teaching_raw"].append(line)

        elif current_section == "research":
            profile["research_areas_raw"].append(line)

        elif current_section == "publications":
            # Publications span multiple lines — buffer until we detect new pub
            # New pub typically starts with author name (Vietnamese capitalized)
            is_new_pub = bool(re.match(r"[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]", line))
            if is_new_pub and pub_buffer:
                _flush_pub()
            pub_buffer.append(line)

    _flush_pub()

    # Clean bio
    profile["bio_raw"] = profile["bio_raw"].strip()

    # Deduplicate lists
    for key in ["teaching_raw", "research_areas_raw"]:
        profile[key] = list(dict.fromkeys(profile[key]))

    return profile


# ── Playwright crawler ────────────────────────────────────────
async def crawl_one(page: Page, meta: dict) -> dict:
    url = meta["profile_url"]
    logger.info(f"Crawling: {meta['title']} {meta['name']} → {url}")
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        html = await page.content()
        profile = _extract_profile(html, meta)

        # Save individual file
        out = PROFILES_DIR / f"{meta['slug']}.json"
        out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(
            f"  OK — teaching:{len(profile['teaching_raw'])} "
            f"research:{len(profile['research_areas_raw'])} "
            f"pubs:{len(profile['publications_raw'])}"
        )
        return profile

    except Exception as e:
        logger.warning(f"  FAILED {url}: {e}")
        return {**meta, "error": str(e), "crawled_at": datetime.now().isoformat()}


async def crawl_all(slug_filter: str | None = None) -> list[dict]:
    if not LIST_JSON.exists():
        logger.error(f"lecturers_list.json not found. Run lecturer_list_crawler.py first.")
        sys.exit(1)

    lecturers = json.loads(LIST_JSON.read_text(encoding="utf-8"))

    # Filter & skip already-done
    todo = []
    for lec in lecturers:
        slug = lec["slug"]
        if slug_filter and slug_filter.lower() not in slug.lower():
            continue
        out_file = PROFILES_DIR / f"{slug}.json"
        if out_file.exists():
            logger.info(f"Skip (already done): {slug}")
            continue
        todo.append(lec)

    if not todo:
        logger.info("Nothing to crawl.")
        # Load existing
        return [
            json.loads((PROFILES_DIR / f"{lec['slug']}.json").read_text(encoding="utf-8"))
            for lec in lecturers
            if (PROFILES_DIR / f"{lec['slug']}.json").exists()
        ]

    logger.info(f"Crawling {len(todo)} lecturer profiles...")
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page()

        for lec in todo:
            profile = await crawl_one(page, lec)
            results.append(profile)
            await asyncio.sleep(DELAY_MS / 1000)

        await browser.close()

    # Merge all profiles into one file
    all_profiles = []
    for lec in lecturers:
        f = PROFILES_DIR / f"{lec['slug']}.json"
        if f.exists():
            all_profiles.append(json.loads(f.read_text(encoding="utf-8")))

    ALL_RAW_JSON.write_text(
        json.dumps(all_profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"\nAll raw profiles ({len(all_profiles)}) → {ALL_RAW_JSON}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", type=str, default=None,
                        help="Crawl only lecturer matching this slug (partial match).")
    args = parser.parse_args()
    asyncio.run(crawl_all(slug_filter=args.slug))
