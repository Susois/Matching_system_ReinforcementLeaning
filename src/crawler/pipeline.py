"""
pipeline.py
───────────
Chạy toàn bộ crawler pipeline theo 3 tầng:

  Step 1 — lecturer_list_crawler   → data/raw/lecturers_list.json
  Step 2 — lecturer_detail_crawler → data/raw/profiles/<slug>.json
  Step 3 — skill_extractor         → data/processed/advisor_research_profiles.json
                                    data/processed/advisor_skills_extracted.csv

Run:
    python -m src.crawler.pipeline              # full pipeline
    python -m src.crawler.pipeline --step 1     # only list
    python -m src.crawler.pipeline --step 2     # only details
    python -m src.crawler.pipeline --step 3     # only skill extraction
    python -m src.crawler.pipeline --slug ts-pham-xuan-lam  # single advisor
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Faculty Research Profile Crawler")
    parser.add_argument("--step", type=int, default=0,
                        help="Run only step N (1=list, 2=details, 3=skills). 0=all.")
    parser.add_argument("--slug", type=str, default=None,
                        help="Process only this lecturer slug (partial match).")
    args = parser.parse_args()

    steps = [args.step] if args.step > 0 else [1, 2, 3]

    if 1 in steps:
        logger.info("=" * 60)
        logger.info("STEP 1 — Crawl lecturer list")
        logger.info("=" * 60)
        from src.crawler.lecturer_list_crawler import crawl
        lecturers = asyncio.run(crawl())
        logger.info(f"Found {len(lecturers)} lecturers.\n")

    if 2 in steps:
        logger.info("=" * 60)
        logger.info("STEP 2 — Crawl lecturer details")
        logger.info("=" * 60)
        from src.crawler.lecturer_detail_crawler import crawl_all
        asyncio.run(crawl_all(slug_filter=args.slug))
        logger.info("")

    if 3 in steps:
        logger.info("=" * 60)
        logger.info("STEP 3 — Extract skills & research topics")
        logger.info("=" * 60)
        from src.crawler.skill_extractor import run as run_skills
        run_skills(slug_filter=args.slug)
        logger.info("")

    logger.info("Crawler pipeline complete.")


if __name__ == "__main__":
    main()
