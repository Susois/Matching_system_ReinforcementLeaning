"""
clean_data.py
─────────────
Main pipeline: PDF/DOCX → OCR → Gemini → thesis_extracted.csv

Usage:
    python -m src.data_pipeline.clean_data
    python -m src.data_pipeline.clean_data --limit 5   # process only first 5 files
    python -m src.data_pipeline.clean_data --force      # re-process already-done files

Output files (in data/processed/):
    thesis_extracted.csv   — one row per thesis, columns = THESIS_CSV_COLUMNS
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Make project root importable ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import (
    PDF_INPUT_DIR,
    PROCESSED_DIR,
    THESIS_CSV,
    THESIS_CSV_COLUMNS,
    BATCH_SIZE,
    MAX_PDF_PAGES,
    LOG_DIR,
)
from src.data_pipeline.pdf_ocr import extract_text, get_source_files
from src.data_pipeline.gemini_extractor import GeminiExtractor

# ── Logging ───────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "clean_data.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────
def _load_existing(csv_path: Path) -> set[str]:
    """Return set of source_file names already processed successfully."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["source_file", "extraction_status"])
        done = df.loc[df["extraction_status"] == "success", "source_file"].tolist()
        return set(done)
    except Exception:
        return set()


def _process_one(file_path: Path, extractor: GeminiExtractor) -> dict:
    """Extract text then call Gemini for one file. Returns record dict."""
    text = extract_text(file_path, max_pages=MAX_PDF_PAGES)
    record = extractor.extract(text, source_file=file_path.name)
    return record


def _save(records: list[dict], csv_path: Path, append: bool = True) -> None:
    """Save / append records to CSV."""
    new_df = pd.DataFrame(records, columns=THESIS_CSV_COLUMNS)
    if append and csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        new_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(records)} records → {csv_path}")


# ── Main pipeline ─────────────────────────────────────────────
def run_pipeline(limit: int = 0, force: bool = False) -> pd.DataFrame:
    """
    Process all source files in PDF_INPUT_DIR.

    Args:
        limit: max number of files to process (0 = all)
        force: if True, re-process files already in CSV

    Returns:
        DataFrame of newly extracted records.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Discover source files
    files = get_source_files(PDF_INPUT_DIR)
    if not files:
        logger.warning(f"No PDF/DOCX files found in {PDF_INPUT_DIR}")
        return pd.DataFrame(columns=THESIS_CSV_COLUMNS)

    # Filter already-processed unless --force
    if not force:
        done = _load_existing(THESIS_CSV)
        todo = [f for f in files if f.name not in done]
        skipped = len(files) - len(todo)
        if skipped:
            logger.info(f"Skipping {skipped} already-processed files (use --force to re-process).")
    else:
        todo = files

    if limit > 0:
        todo = todo[:limit]

    if not todo:
        logger.info("Nothing new to process.")
        return pd.DataFrame(columns=THESIS_CSV_COLUMNS)

    logger.info(f"Processing {len(todo)} files with {BATCH_SIZE} workers...")

    # Initialise extractor (validates API key early)
    try:
        extractor = GeminiExtractor()
    except EnvironmentError as e:
        logger.error(str(e))
        sys.exit(1)

    # Parallel processing
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
        futures = {pool.submit(_process_one, f, extractor): f for f in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            file_path = futures[future]
            try:
                record = future.result()
            except Exception as e:
                logger.error(f"Unexpected error for {file_path.name}: {e}")
                record = GeminiExtractor._failed_record(file_path.name, str(e))
            records.append(record)

    # Save
    _save(records, THESIS_CSV, append=(not force))

    ok  = sum(1 for r in records if r.get("extraction_status") == "success")
    fail = len(records) - ok
    logger.info(f"Done. Success: {ok}  Failed: {fail}  Total: {len(records)}")

    return pd.DataFrame(records, columns=THESIS_CSV_COLUMNS)


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract thesis info from PDFs using OCR + Gemini."
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N files (default: 0 = all)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process files already present in the output CSV."
    )
    args = parser.parse_args()
    run_pipeline(limit=args.limit, force=args.force)
