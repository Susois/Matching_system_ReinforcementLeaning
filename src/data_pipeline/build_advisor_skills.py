"""
build_advisor_skills.py
───────────────────────
Aggregate per-thesis technology extractions into per-advisor skill profiles.

Inputs:
    data/processed/thesis_extracted.csv   (from clean_data.py)

Outputs:
    data/processed/advisor_skills.csv     — one row per advisor, tech columns
                                            contain comma-separated unique values
    data/processed/advisor_profiles.csv   — richer profile including counts,
                                            field distribution, rating columns
                                            ready for the RL environment

Run:
    python -m src.data_pipeline.build_advisor_skills
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import (
    THESIS_CSV,
    ADVISOR_SKILLS_CSV,
    ADVISOR_PROFILE_CSV,
    SKILL_TECH_COLS,
    FIELD_LABELS,
    LOG_DIR,
    PROCESSED_DIR,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "build_advisor_skills.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────
def _tokenize(value: Optional[str]) -> list[str]:
    """
    Split a comma/semicolon separated tech string into a clean token list.
    e.g. "PyTorch, TensorFlow / Keras" → ["PyTorch", "TensorFlow / Keras"]
    """
    if not value or not isinstance(value, str):
        return []
    tokens = [t.strip() for t in value.replace(";", ",").split(",")]
    return [t for t in tokens if t and t.lower() not in {"none", "null", "không áp dụng", "n/a"}]


def _merge_tokens(series: pd.Series) -> str:
    """Merge all token lists in a column into a deduplicated comma-separated string."""
    seen: dict[str, int] = {}   # token → frequency
    for val in series:
        for tok in _tokenize(val):
            seen[tok] = seen.get(tok, 0) + 1
    # Sort by frequency descending
    sorted_toks = sorted(seen, key=lambda t: (-seen[t], t))
    return ", ".join(sorted_toks) if sorted_toks else ""


def _field_distribution(series: pd.Series) -> dict[str, int]:
    """Count how many theses fall into each field category."""
    dist: dict[str, int] = {label: 0 for label in FIELD_LABELS}
    for val in series:
        if isinstance(val, str) and val.strip():
            for label in FIELD_LABELS:
                if label.lower() in val.lower():
                    dist[label] += 1
                    break
    return dist


def _average_grade(series: pd.Series) -> Optional[float]:
    """Parse and average numeric grades."""
    grades = []
    for val in series:
        if val is None:
            continue
        try:
            grades.append(float(str(val).replace(",", ".")))
        except (ValueError, TypeError):
            pass
    return round(sum(grades) / len(grades), 2) if grades else None


# ── Core aggregation ──────────────────────────────────────────
def build_advisor_skills(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given the thesis DataFrame, return:
      - skills_df   : one row per advisor, columns = SKILL_TECH_COLS (merged tokens)
      - profiles_df : richer profile with counts, field dist, avg grade
    """
    # Keep only successfully extracted rows with a valid advisor
    df = df[df["extraction_status"] == "success"].copy()
    df["advisor_name"] = df["advisor_name"].str.strip()
    df = df[df["advisor_name"].notna() & (df["advisor_name"] != "")]

    if df.empty:
        logger.warning("No successful extractions with advisor names found.")
        return pd.DataFrame(), pd.DataFrame()

    advisors = sorted(df["advisor_name"].unique())
    logger.info(f"Building skill profiles for {len(advisors)} advisor(s).")

    skills_rows  = []
    profile_rows = []

    for advisor in advisors:
        adf = df[df["advisor_name"] == advisor]

        # ── Skills row ────────────────────────────────────────
        skill_row: dict = {"advisor_name": advisor}
        for col in SKILL_TECH_COLS:
            if col in adf.columns:
                skill_row[col] = _merge_tokens(adf[col])
            else:
                skill_row[col] = ""
        skills_rows.append(skill_row)

        # ── Profile row ───────────────────────────────────────
        field_dist = _field_distribution(adf["field_category"])
        primary_field = (
            max(field_dist, key=field_dist.get)
            if any(field_dist.values()) else "Unknown"
        )

        profile_row: dict = {
            "advisor_name":       advisor,
            "thesis_count":       len(adf),
            "avg_grade":          _average_grade(adf.get("thesis_grade", pd.Series(dtype=str))),
            "primary_field":      primary_field,
            # Field distribution columns
            **{f"field_{i}": cnt for i, (_, cnt) in enumerate(field_dist.items())},
            # Skill summary (comma-sep, top-frequency first)
            "top_ai_frameworks":  _merge_tokens(adf.get("ai_frameworks", pd.Series(dtype=str))),
            "top_web_stack":      _merge_tokens(adf.get("frontend_frameworks", pd.Series(dtype=str))),
            "top_backend":        _merge_tokens(adf.get("backend_frameworks", pd.Series(dtype=str))),
            "top_db":             _merge_tokens(adf.get("database_cache", pd.Series(dtype=str))),
            "top_data_tools":     _merge_tokens(adf.get("data_tools", pd.Series(dtype=str))),
            # Rating placeholders — to be filled later via crawled data
            "rating_expertise":   None,   # 1–5, filled by crawl_advisor_data.py
            "rating_guidance":    None,
            "rating_support":     None,
            "rating_feedback":    None,
            "rating_management":  None,
            "rating_style":       None,
            # Embedding placeholder (filled by nlp/embed_advisors.py)
            "research_embedding": None,
        }
        # Add field distribution as named columns
        for label, cnt in field_dist.items():
            safe_key = "field_" + label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            profile_row[safe_key] = cnt

        # Remove the positional field_0..N duplicates we added above
        for i in range(len(field_dist)):
            profile_row.pop(f"field_{i}", None)

        profile_rows.append(profile_row)

    skills_df  = pd.DataFrame(skills_rows)
    profile_df = pd.DataFrame(profile_rows)

    return skills_df, profile_df


# ── Survey data merge ─────────────────────────────────────────
def merge_survey_ratings(
    profile_df: pd.DataFrame,
    survey_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Merge advisor ratings from khaosat_kltn.csv (Section: đã hoàn thành KLTN)
    into the profile DataFrame.

    Rating columns in survey (Vietnamese):
        Chuyên môn và mức độ phù hợp với đề tài
        Khả năng định hướng và xây dựng đề tài
        Mức độ hỗ trợ và tương tác
        Chất lượng phản hồi và góp ý
        Quản lý tiến độ và trách nhiệm hướng dẫn
        Phong cách hướng dẫn và mức độ hài lòng
    """
    if survey_csv is None or not Path(survey_csv).exists():
        logger.info("No survey CSV supplied; skipping rating merge.")
        return profile_df

    survey = pd.read_csv(survey_csv, encoding="utf-8-sig")

    # Keep only completed-thesis rows
    col_group = "1. Bạn thuộc nhóm đối tượng nào?"
    if col_group in survey.columns:
        survey = survey[survey[col_group].str.contains("đã hoàn thành", case=False, na=False)]

    rating_cols = {
        "Nhận xét về giảng viên hướng dẫn [Chuyên môn và mức độ phù hợp với đề tài]": "rating_expertise",
        "Nhận xét về giảng viên hướng dẫn [Khả năng định hướng và xây dựng đề tài]": "rating_guidance",
        "Nhận xét về giảng viên hướng dẫn [Mức độ hỗ trợ và tương tác]":             "rating_support",
        "Nhận xét về giảng viên hướng dẫn [Chất lượng phản hồi và góp ý]":           "rating_feedback",
        "Nhận xét về giảng viên hướng dẫn [Quản lý tiến độ và trách nhiệm hướng dẫn]": "rating_management",
        "Nhận xét về giảng viên hướng dẫn [Phong cách hướng dẫn và mức độ hài lòng]":  "rating_style",
    }

    # Parse "5 – Rất phù hợp" → 5.0
    def _parse_rating(val):
        if pd.isna(val):
            return None
        try:
            return float(str(val).split("–")[0].split("-")[0].strip())
        except ValueError:
            return None

    for vi_col, en_col in rating_cols.items():
        if vi_col in survey.columns:
            survey[en_col] = survey[vi_col].apply(_parse_rating)

    advisor_col = "Giảng viên hướng dẫn"
    if advisor_col not in survey.columns:
        logger.warning(f"Column '{advisor_col}' not found in survey; skipping merge.")
        return profile_df

    survey = survey.rename(columns={advisor_col: "advisor_name"})
    keep_cols = ["advisor_name"] + list(rating_cols.values())
    survey_agg = (
        survey[keep_cols]
        .groupby("advisor_name", as_index=False)
        .mean(numeric_only=True)
        .round(2)
    )

    # Merge into profile_df, overwrite rating columns
    for col in rating_cols.values():
        profile_df.pop(col, None)   # drop old placeholder

    merged = profile_df.merge(survey_agg, on="advisor_name", how="left")
    logger.info(f"Merged survey ratings for {len(survey_agg)} advisor(s).")
    return merged


# ── Entry point ───────────────────────────────────────────────
def run(survey_csv: Optional[Path] = None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not THESIS_CSV.exists():
        logger.error(
            f"thesis_extracted.csv not found at {THESIS_CSV}. "
            "Run clean_data.py first."
        )
        sys.exit(1)

    logger.info(f"Loading {THESIS_CSV} ...")
    df = pd.read_csv(THESIS_CSV)
    logger.info(f"Loaded {len(df)} rows, {df['extraction_status'].value_counts().to_dict()}")

    skills_df, profile_df = build_advisor_skills(df)

    if skills_df.empty:
        logger.warning("Empty result — no output files written.")
        return

    # Merge survey ratings
    from configs.settings import SURVEY_CSV
    profile_df = merge_survey_ratings(profile_df, survey_csv or SURVEY_CSV)

    # Save
    skills_df.to_csv(ADVISOR_SKILLS_CSV,  index=False, encoding="utf-8-sig")
    profile_df.to_csv(ADVISOR_PROFILE_CSV, index=False, encoding="utf-8-sig")

    logger.info(f"advisor_skills.csv   → {ADVISOR_SKILLS_CSV}")
    logger.info(f"advisor_profiles.csv → {ADVISOR_PROFILE_CSV}")
    logger.info(
        f"Advisors processed: {len(skills_df)} | "
        f"Avg theses per advisor: {profile_df['thesis_count'].mean():.1f}"
    )

    # Quick summary to console
    print("\n" + "="*60)
    print("ADVISOR SKILL SUMMARY")
    print("="*60)
    for _, row in profile_df.iterrows():
        print(f"\n► {row['advisor_name']}")
        print(f"   Theses: {row['thesis_count']}  |  Avg grade: {row.get('avg_grade', 'N/A')}")
        print(f"   Primary field: {row.get('primary_field', 'N/A')}")
        if row.get("top_ai_frameworks"):
            print(f"   AI: {row['top_ai_frameworks']}")
        if row.get("top_backend"):
            print(f"   Backend: {row['top_backend']}")
        ratings = {
            k: row.get(k) for k in [
                "rating_expertise", "rating_guidance", "rating_support",
                "rating_feedback", "rating_management", "rating_style"
            ] if row.get(k) is not None
        }
        if ratings:
            avg_r = round(sum(ratings.values()) / len(ratings), 2)
            print(f"   Avg rating: {avg_r}/5")


if __name__ == "__main__":
    run()
