"""
crawl_advisor_data.py
──────────────────────
Crawl publicly available advisor information to enrich advisor_profiles.csv.

Sources (in order of priority):
  1. NEU faculty directory  (scraping HTML)
  2. Google Scholar public profiles
  3. Semantic Scholar API   (no key required, rate-limited)
  4. Manual JSON override   (configs/advisor_overrides.json)

Outputs:
    data/processed/advisor_crawled.csv   — raw crawled data
    data/processed/advisor_profiles.csv  — updated with crawled fields

Run:
    python -m src.data_pipeline.crawl_advisor_data
    python -m src.data_pipeline.crawl_advisor_data --advisor "TS Phạm Xuân Lâm"
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import (
    ADVISOR_PROFILE_CSV,
    PROCESSED_DIR,
    LOG_DIR,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "crawl_advisor.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

CRAWLED_CSV  = PROCESSED_DIR / "advisor_crawled.csv"
OVERRIDE_JSON = ROOT / "configs" / "advisor_overrides.json"

# HTTP session with polite headers
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; KLTN-Research-Bot/1.0; "
        "+https://github.com/student-advisor-rl)"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
})
REQUEST_DELAY = 2.0   # seconds between requests (polite crawling)


# ── Name utilities ────────────────────────────────────────────
def _normalize_name(raw: str) -> str:
    """
    Strip academic title prefix and normalize Vietnamese name for search.
    'TS Phạm Xuân Lâm' → 'Phạm Xuân Lâm'
    'PGS.TS. Nguyễn Văn A' → 'Nguyễn Văn A'
    """
    raw = raw.strip()
    prefixes = r"^(PGS\.TS\.|TS\.?|ThS\.?|GS\.?|PGS\.?|NCS\.?)\s*"
    return re.sub(prefixes, "", raw, flags=re.IGNORECASE).strip()


def _build_query(name: str, institution: str = "Đại học Kinh tế Quốc dân") -> str:
    return f"{name} {institution}"


# ── Source 1: Semantic Scholar ────────────────────────────────
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/author/search"
SEMANTIC_SCHOLAR_PAPER_API = "https://api.semanticscholar.org/graph/v1/author/{}/papers"

def _crawl_semantic_scholar(name: str) -> dict:
    """Search Semantic Scholar for an author and return structured data."""
    clean = _normalize_name(name)
    try:
        resp = _SESSION.get(
            SEMANTIC_SCHOLAR_API,
            params={"query": clean, "limit": 3,
                    "fields": "name,affiliations,paperCount,citationCount,hIndex"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("data", [])
        if not items:
            return {}

        # Pick the first result (most likely match)
        author = items[0]
        author_id = author.get("authorId", "")
        result = {
            "ss_author_id":    author_id,
            "ss_paper_count":  author.get("paperCount"),
            "ss_citation_count": author.get("citationCount"),
            "ss_h_index":      author.get("hIndex"),
            "ss_affiliation":  (author.get("affiliations") or [None])[0],
        }

        # Fetch recent publications (up to 10)
        if author_id:
            time.sleep(REQUEST_DELAY)
            paper_resp = _SESSION.get(
                SEMANTIC_SCHOLAR_PAPER_API.format(author_id),
                params={"limit": 10, "fields": "title,year,fieldsOfStudy"},
                timeout=10,
            )
            if paper_resp.ok:
                papers = paper_resp.json().get("data", [])
                titles = [p.get("title", "") for p in papers if p.get("title")]
                fields = list({
                    f
                    for p in papers
                    for f in (p.get("fieldsOfStudy") or [])
                })
                result["ss_recent_papers"] = " | ".join(titles[:10])
                result["ss_research_fields"] = ", ".join(fields)

        logger.info(f"Semantic Scholar: {name} → papers={result.get('ss_paper_count')}")
        return result

    except Exception as e:
        logger.warning(f"Semantic Scholar failed for {name}: {e}")
        return {}


# ── Source 2: Generic Google Scholar scrape ───────────────────
SCHOLAR_URL = "https://scholar.google.com/scholar"

def _crawl_google_scholar(name: str) -> dict:
    """
    Scrape Google Scholar search results for an advisor's name.
    NOTE: Google Scholar blocks bots aggressively.  This is a best-effort
    attempt; failures are handled gracefully.
    """
    clean  = _normalize_name(name)
    query  = f'author:"{clean}"'
    try:
        resp = _SESSION.get(
            SCHOLAR_URL,
            params={"q": query, "hl": "vi"},
            timeout=12,
        )
        if resp.status_code == 429:
            logger.warning("Google Scholar rate-limited. Skipping.")
            return {}
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        titles = [h3.get_text() for h3 in soup.select("h3.gs_rt")][:5]
        return {
            "gs_titles_sample": " | ".join(titles),
        }
    except Exception as e:
        logger.warning(f"Google Scholar failed for {name}: {e}")
        return {}


# ── Source 3: Manual override JSON ────────────────────────────
def _load_overrides() -> dict:
    """
    Load hand-crafted advisor data from configs/advisor_overrides.json.
    Format:
    {
      "TS Phạm Xuân Lâm": {
        "research_interests": "AI, NLP, Computer Vision",
        "homepage": "https://...",
        "note": "..."
      }
    }
    """
    if not OVERRIDE_JSON.exists():
        # Create a template if missing
        template = {
            "_readme": (
                "Add advisor overrides here. "
                "Keys = exact advisor_name as in advisor_profiles.csv"
            ),
        }
        OVERRIDE_JSON.write_text(
            json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Created override template at {OVERRIDE_JSON}")
        return {}
    try:
        data = json.loads(OVERRIDE_JSON.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception as e:
        logger.warning(f"Could not load overrides: {e}")
        return {}


# ── Aggregate skill score ─────────────────────────────────────
def _compute_skill_score(row: pd.Series) -> Optional[float]:
    """
    Heuristic skill score (0–100) combining:
      - h-index (0–30)
      - paper count (0–30)
      - citation count (0–20)
      - avg student rating (0–20)
    All components normalised with soft caps.
    """
    def soft(val, cap):
        if val is None:
            return 0.0
        return min(float(val), cap) / cap

    h_score   = soft(row.get("ss_h_index"),        20) * 30
    p_score   = soft(row.get("ss_paper_count"),    50) * 30
    c_score   = soft(row.get("ss_citation_count"), 500) * 20

    rating_cols = [
        "rating_expertise", "rating_guidance", "rating_support",
        "rating_feedback", "rating_management", "rating_style",
    ]
    ratings = [float(row[c]) for c in rating_cols if row.get(c) is not None]
    r_score = (sum(ratings) / (len(ratings) * 5)) * 20 if ratings else 0.0

    return round(h_score + p_score + c_score + r_score, 1)


# ── Main crawl loop ───────────────────────────────────────────
def crawl_all(advisor_filter: Optional[str] = None) -> pd.DataFrame:
    """Crawl data for all (or one) advisor and return enriched DataFrame."""

    if not ADVISOR_PROFILE_CSV.exists():
        logger.error(
            f"{ADVISOR_PROFILE_CSV} not found. "
            "Run build_advisor_skills.py first."
        )
        sys.exit(1)

    profiles = pd.read_csv(ADVISOR_PROFILE_CSV)
    overrides = _load_overrides()

    if advisor_filter:
        profiles = profiles[
            profiles["advisor_name"].str.contains(advisor_filter, case=False, na=False)
        ]
        if profiles.empty:
            logger.error(f"No advisor matching '{advisor_filter}'")
            sys.exit(1)

    crawled_rows = []
    for _, row in profiles.iterrows():
        name = row["advisor_name"]
        logger.info(f"\n── Crawling: {name} ──")

        result: dict = {"advisor_name": name}

        # Apply manual overrides first
        if name in overrides:
            result.update(overrides[name])
            logger.info(f"Applied manual override for {name}.")

        # Semantic Scholar
        time.sleep(REQUEST_DELAY)
        result.update(_crawl_semantic_scholar(name))

        # Google Scholar (best-effort)
        time.sleep(REQUEST_DELAY)
        result.update(_crawl_google_scholar(name))

        # Computed skill score
        combined = {**row.to_dict(), **result}
        result["skill_score"] = _compute_skill_score(pd.Series(combined))

        crawled_rows.append(result)

    crawled_df = pd.DataFrame(crawled_rows)
    crawled_df.to_csv(CRAWLED_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Crawled data saved → {CRAWLED_CSV}")

    # Merge back into profiles
    profiles_updated = profiles.merge(
        crawled_df[["advisor_name"] + [c for c in crawled_df.columns if c != "advisor_name"]],
        on="advisor_name",
        how="left",
        suffixes=("", "_crawled"),
    )
    # Prefer crawled values for overlapping columns
    for col in crawled_df.columns:
        if col == "advisor_name":
            continue
        crawled_col = col + "_crawled"
        if crawled_col in profiles_updated.columns:
            profiles_updated[col] = profiles_updated[crawled_col].combine_first(
                profiles_updated.get(col, pd.Series(dtype=object))
            )
            profiles_updated.drop(columns=[crawled_col], inplace=True)

    profiles_updated.to_csv(ADVISOR_PROFILE_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Updated advisor_profiles.csv → {ADVISOR_PROFILE_CSV}")

    return profiles_updated


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl public data for advisor skill enrichment."
    )
    parser.add_argument(
        "--advisor", type=str, default=None,
        help="Filter to a single advisor name (partial match)."
    )
    args = parser.parse_args()
    df = crawl_all(advisor_filter=args.advisor)
    print(f"\nDone. {len(df)} advisor(s) updated.")
