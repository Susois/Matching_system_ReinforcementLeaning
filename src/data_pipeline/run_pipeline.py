"""
run_pipeline.py
───────────────
One-shot entry point that runs the full data pipeline in sequence:

  Step 1: clean_data.py        — PDF OCR + Gemini extraction → thesis_extracted.csv
  Step 2: build_advisor_skills — aggregate skills → advisor_skills.csv + advisor_profiles.csv
  Step 3: merge_to_survey      — merge with khaosat_kltn.csv → merged_dataset.csv
  Step 4: crawl_advisor_data   — (optional) enrich profiles via web crawl

Usage:
    # Full pipeline
    python -m src.data_pipeline.run_pipeline

    # Skip crawling (fast, no network)
    python -m src.data_pipeline.run_pipeline --no-crawl

    # Only process first N PDFs (useful for testing)
    python -m src.data_pipeline.run_pipeline --limit 3 --no-crawl

    # Force re-process all PDFs
    python -m src.data_pipeline.run_pipeline --force
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run full data pipeline.")
    parser.add_argument("--limit",    type=int, default=0,
                        help="Limit number of PDFs to process (0=all).")
    parser.add_argument("--force",    action="store_true",
                        help="Re-process all PDFs even if already extracted.")
    parser.add_argument("--no-crawl", action="store_true",
                        help="Skip the web crawl step.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("STEP 1 — PDF extraction (OCR + Gemini)")
    logger.info("=" * 60)
    from src.data_pipeline.clean_data import run_pipeline
    df = run_pipeline(limit=args.limit, force=args.force)
    logger.info(f"Extraction complete: {len(df)} rows produced.\n")

    logger.info("=" * 60)
    logger.info("STEP 2 — Build advisor skill profiles")
    logger.info("=" * 60)
    from src.data_pipeline.build_advisor_skills import run as run_skills
    run_skills()
    logger.info("")

    logger.info("=" * 60)
    logger.info("STEP 3 — Merge with survey data")
    logger.info("=" * 60)
    from src.data_pipeline.merge_to_survey import run as run_merge
    run_merge()
    logger.info("")

    if not args.no_crawl:
        logger.info("=" * 60)
        logger.info("STEP 4 — Crawl advisor public data")
        logger.info("=" * 60)
        from src.data_pipeline.crawl_advisor_data import crawl_all
        crawl_all()
        logger.info("")
    else:
        logger.info("STEP 4 — Crawl skipped (--no-crawl).")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
