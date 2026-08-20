"""
lecturer_list_crawler.py
────────────────────────
Tầng 1: Crawl danh sách giảng viên từ https://fit.neu.edu.vn/lecturer

Output:
    data/raw/lecturers_list.json   — list of {name, title, slug, profile_url}

Run:
    python -m src.crawler.lecturer_list_crawler
"""

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT   = RAW_DIR / "lecturers_list.json"

BASE_URL  = "https://fit.neu.edu.vn"
LIST_URL  = BASE_URL + "/lecturer"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _parse_title_name(text: str) -> tuple[str, str]:
    """
    'TS PHẠM XUÂN LÂM' → ('TS', 'Phạm Xuân Lâm')
    'ThS Nguyễn Văn A' → ('ThS', 'Nguyễn Văn A')
    """
    prefixes = ["PGS.TS", "GS.TS", "GS", "PGS", "TS", "ThS", "CN", "NCS"]
    text = text.strip()
    title = ""
    for p in prefixes:
        if text.upper().startswith(p.upper()):
            title = p
            text = text[len(p):].strip().strip(".")
            break
    # Title-case name
    name = " ".join(w.capitalize() for w in text.split())
    return title, name


async def crawl() -> list[dict]:
    logger.info(f"Crawling lecturer list from {LIST_URL}")
    results = []
    seen_slugs = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page()
        await page.goto(LIST_URL, wait_until="networkidle", timeout=30000)
        html = await page.content()
        await browser.close()

    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Pattern: /lecturer/<slug>  (but NOT just /lecturer)
        m = re.match(r"^/lecturer/([^/]+)$", href)
        if not m:
            continue
        slug = m.group(1)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        raw_text = a.get_text(separator=" ", strip=True)
        title, name = _parse_title_name(raw_text) if raw_text else ("", "")

        # Fallback: derive name from slug
        # e.g. ts-pham-xuan-lam → Phạm Xuân Lâm
        if not name:
            parts = slug.split("-")
            # Drop known title prefixes in slug
            if parts[0].lower() in ["ts", "ths", "pgs", "gs", "cn", "ncs", "th"]:
                if parts[1].lower() == "s":   # th-s-... pattern
                    parts = parts[2:]
                else:
                    parts = parts[1:]
            name = " ".join(w.capitalize() for w in parts)

        profile_url = BASE_URL + href
        record = {
            "slug":        slug,
            "name":        name,
            "title":       title,
            "profile_url": profile_url,
            "department":  "Khoa Công nghệ thông tin",
        }
        results.append(record)
        logger.info(f"  Found: {title} {name} → {slug}")

    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"\nTotal: {len(results)} lecturers → {OUTPUT}")
    return results


if __name__ == "__main__":
    asyncio.run(crawl())
