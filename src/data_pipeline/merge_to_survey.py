"""
merge_to_survey.py
──────────────────
Merge PDF-extracted thesis data (thesis_extracted.csv) with the
existing survey data (khaosat_kltn.csv) into a unified dataset.

The survey has 3 respondent groups:
  A. "Sinh viên đã hoàn thành" — provided by PDF extraction pipeline
  B. "Sinh viên đang thực hiện" — from survey form
  C. "Sinh viên sắp thực hiện" — from survey form

This script:
  1. Reads both sources.
  2. Maps thesis_extracted.csv columns → survey column names (using COL_VI_ALIAS).
  3. Fills missing tech columns with pd.NA (not zeros, per data integrity rules).
  4. Appends PDF rows to survey (group A) avoiding duplicates by student_id.
  5. Writes merged output to data/processed/merged_dataset.csv.

Run:
    python -m src.data_pipeline.merge_to_survey
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import (
    THESIS_CSV,
    SURVEY_CSV,
    PROCESSED_DIR,
    COL_VI_ALIAS,
    LOG_DIR,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "merge_to_survey.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

MERGED_CSV = PROCESSED_DIR / "merged_dataset.csv"


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load thesis extracted ─────────────────────────────────
    if not THESIS_CSV.exists():
        logger.error(f"thesis_extracted.csv not found at {THESIS_CSV}. Run clean_data.py first.")
        sys.exit(1)

    thesis_df = pd.read_csv(THESIS_CSV)
    thesis_df = thesis_df[thesis_df["extraction_status"] == "success"].copy()
    logger.info(f"Loaded {len(thesis_df)} successfully extracted thesis rows.")

    # Rename columns to Vietnamese aliases for consistency
    rename_map = {k: v for k, v in COL_VI_ALIAS.items() if k in thesis_df.columns}
    thesis_vi = thesis_df.rename(columns=rename_map)

    # Tag group
    thesis_vi["nhom_doi_tuong"] = "Sinh viên đã hoàn thành khóa luận tốt nghiệp"
    thesis_vi["nguon_du_lieu"]  = "PDF extraction"

    # ── Load survey ───────────────────────────────────────────
    survey_df: pd.DataFrame
    if SURVEY_CSV.exists():
        survey_df = pd.read_csv(SURVEY_CSV, encoding="utf-8-sig")
        logger.info(f"Loaded {len(survey_df)} survey rows.")
    else:
        logger.warning("khaosat_kltn.csv not found — creating empty survey frame.")
        survey_df = pd.DataFrame()

    # ── Align columns ─────────────────────────────────────────
    all_cols = list(
        dict.fromkeys(
            list(survey_df.columns) + list(thesis_vi.columns)
        )
    )
    survey_aligned  = survey_df.reindex(columns=all_cols)
    thesis_aligned  = thesis_vi.reindex(columns=all_cols)

    # ── Dedup: remove PDF rows whose student_id already exists in survey ─
    sv_id_col = COL_VI_ALIAS.get("student_id", "Mã sinh viên")
    if sv_id_col in survey_aligned.columns and sv_id_col in thesis_aligned.columns:
        existing_ids = set(
            survey_aligned[sv_id_col].dropna().astype(str).str.strip()
        )
        n_before = len(thesis_aligned)
        thesis_aligned = thesis_aligned[
            ~thesis_aligned[sv_id_col].astype(str).str.strip().isin(existing_ids)
        ]
        n_duped = n_before - len(thesis_aligned)
        if n_duped:
            logger.info(f"Removed {n_duped} duplicate student_id(s) already in survey.")

    # ── Concatenate ───────────────────────────────────────────
    merged = pd.concat([survey_aligned, thesis_aligned], ignore_index=True)
    merged = merged.where(pd.notnull(merged), other=None)   # keep NA as None

    merged.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Merged dataset ({len(merged)} rows) → {MERGED_CSV}")

    # Summary
    group_col = "nhom_doi_tuong" if "nhom_doi_tuong" in merged.columns else (
        "1. Bạn thuộc nhóm đối tượng nào?"
        if "1. Bạn thuộc nhóm đối tượng nào?" in merged.columns else None
    )
    if group_col:
        print("\nRow counts by group:")
        print(merged[group_col].value_counts().to_string())


if __name__ == "__main__":
    run()
