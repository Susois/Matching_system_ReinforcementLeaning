"""
skill_extractor.py
──────────────────
Tầng 3: Trích xuất Skill + Research Topic từ raw profile.
TÁCH BIỆT hoàn toàn khỏi crawler — chạy sau khi đã có raw JSON.

Pipeline 3 bước theo đề xuất:
  Step 1 — Taxonomy keyword matching   (rule-based, fast, high precision)
  Step 2 — LLM extraction              (flexible, handles Vietnamese text)
  Step 3 — Evidence scoring            (skill_strength với time-decay)

Output:
    data/processed/advisor_research_profiles.json   — full profiles
    data/processed/advisor_skills_extracted.csv     — flat CSV cho RL

Run:
    python -m src.crawler.skill_extractor
    python -m src.crawler.skill_extractor --slug ts-pham-xuan-lam
"""

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_DIR      = ROOT / "data" / "raw"
PROFILES_DIR = RAW_DIR / "profiles"
PROCESSED    = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON  = PROCESSED / "advisor_research_profiles.json"
OUTPUT_CSV   = PROCESSED / "advisor_skills_extracted.csv"

CURRENT_YEAR = datetime.now().year
DECAY_LAMBDA = 0.05   # e^(-λ·Δyear) — weight decay per year

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  TAXONOMY
#  Chỉ gán skill khi có bằng chứng rõ ràng (không suy diễn).
# ══════════════════════════════════════════════════════════════
SKILL_TAXONOMY: dict[str, list[str]] = {
    # AI / ML
    "Machine Learning":        ["machine learning", "học máy", "ml "],
    "Deep Learning":           ["deep learning", "học sâu", "neural network", "mạng nơ-ron"],
    "Reinforcement Learning":  ["reinforcement learning", "học tăng cường", "rl ", "dqn", "ppo", "q-learning"],
    "Natural Language Processing": ["nlp", "natural language", "xử lý ngôn ngữ", "text classification",
                                     "sentiment analysis", "n-gram", "name detection", "information extraction"],
    "Computer Vision":         ["computer vision", "image", "object detection", "yolo", "cnn",
                                 "nhận dạng hình ảnh", "phát hiện vật thể"],
    "Large Language Models":   ["llm", "large language model", "gpt", "chatgpt", "bert", "transformer",
                                 "generative ai", "genai", "rag", "retrieval augmented"],
    "Recommendation Systems":  ["recommendation", "gợi ý", "collaborative filtering", "expert finding"],
    "Predictive Analytics":    ["prediction", "forecasting", "dự báo", "dự đoán", "predictive"],
    "Sentiment Analysis":      ["sentiment", "opinion mining", "phân tích cảm xúc"],
    "RPA":                     ["rpa", "robotic process automation"],
    # Data
    "Data Science":            ["data science", "khoa học dữ liệu", "data analysis", "phân tích dữ liệu"],
    "Big Data":                ["big data", "hadoop", "spark", "distributed"],
    "Database":                ["database", "cơ sở dữ liệu", "sql", "mysql", "postgresql"],
    "NoSQL":                   ["nosql", "mongodb", "cassandra", "redis", "dữ liệu phi cấu trúc"],
    # Web / App
    "Web Development":         ["web development", "thiết kế web", "web application", "html", "css"],
    "Mobile Development":      ["mobile", "android", "ios", "flutter", "react native",
                                 "phát triển ứng dụng di động", "mobile learning", "mobile app"],
    "Java":                    ["java", "lập trình java", "spring"],
    "Python":                  ["python"],
    "JavaScript":              ["javascript", "nodejs", "react", "vue", "angular"],
    # Cloud / Security
    "Cloud Computing":         ["cloud", "aws", "azure", "gcp", "điện toán đám mây"],
    "Cybersecurity":           ["security", "an toàn thông tin", "mật mã", "blockchain"],
    "IoT":                     ["iot", "internet of things", "nhúng", "embedded", "sensor"],
    # Systems
    "Operating Systems":       ["operating system", "hệ điều hành"],
    "Computer Architecture":   ["computer architecture", "kiến trúc máy tính"],
    "Algorithms":              ["algorithm", "thuật toán", "optimization", "tối ưu hóa"],
    "Knowledge-Based Systems": ["knowledge-based", "expert system", "ontology", "knowledge graph"],
}

RESEARCH_TOPIC_TAXONOMY: dict[str, list[str]] = {
    "Educational Technology":   ["educational technology", "công nghệ giáo dục", "edtech",
                                  "e-learning", "lms", "moodle", "học trực tuyến"],
    "Learning Analytics":       ["learning analytics", "phân tích học tập", "student assessment",
                                  "academic performance", "đánh giá học sinh"],
    "AI in Education":          ["ai in education", "ai-powered learning", "intelligent tutoring",
                                  "adaptive learning", "trí tuệ nhân tạo trong giáo dục"],
    "Mobile Learning":          ["mobile learning", "học di động", "m-learning"],
    "Metaverse / XR":           ["metaverse", "virtual reality", "augmented reality", "xr", "vr", "ar"],
    "Digital Transformation":   ["digital transformation", "chuyển đổi số", "digitalization"],
    "Human-Computer Interaction": ["hci", "human-computer", "user interface", "ux", "ui",
                                    "giao diện người dùng", "eye tracking"],
    "Expert Finding":           ["expert finding", "expertise", "skill matching", "tìm kiếm chuyên gia"],
    "Healthcare IT":            ["healthcare", "y tế", "medical", "clinical", "bệnh viện"],
    "Finance / FinTech":        ["finance", "fintech", "stock", "chứng khoán", "tài chính"],
    "Smart Systems":            ["smart city", "smart home", "iot platform", "hệ thống thông minh"],
    "Continual Learning":       ["continual learning", "lifelong learning", "catastrophic forgetting",
                                  "incremental learning"],
}


# ══════════════════════════════════════════════════════════════
#  STEP 1 — Keyword / Taxonomy Matching
# ══════════════════════════════════════════════════════════════
def _taxonomy_match(text: str, taxonomy: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Returns {skill: [matched_keywords]} for each skill found in text.
    Conservative: only exact keyword matches.
    """
    text_lower = text.lower()
    matched: dict[str, list[str]] = {}
    for skill, keywords in taxonomy.items():
        hits = [kw for kw in keywords if kw in text_lower]
        if hits:
            matched[skill] = hits
    return matched


def _build_corpus(profile: dict) -> str:
    """Concatenate all raw text fields into one searchable string."""
    parts = [
        profile.get("bio_raw", ""),
        " ".join(profile.get("research_areas_raw", [])),
        " ".join(profile.get("teaching_raw", [])),
        " ".join(profile.get("publications_raw", [])),
        " ".join(profile.get("work_history_raw", [])),
    ]
    return "\n".join(p for p in parts if p)


# ══════════════════════════════════════════════════════════════
#  STEP 2 — Publication Evidence + Time Decay
# ══════════════════════════════════════════════════════════════
def _extract_year(pub_text: str) -> Optional[int]:
    m = re.search(r"\b(20\d{2}|199\d)\b", pub_text)
    return int(m.group()) if m else None


def _time_weight(year: Optional[int]) -> float:
    if year is None:
        return 0.5   # unknown year → neutral weight
    delta = max(0, CURRENT_YEAR - year)
    return round(math.exp(-DECAY_LAMBDA * delta), 3)


def _evidence_from_publications(
    publications: list[str],
    skill: str,
    keywords: list[str],
) -> list[dict]:
    """Find publications that mention the skill keywords → evidence list."""
    evidence = []
    kw_lower = [k.lower() for k in keywords]
    for pub in publications:
        pub_lower = pub.lower()
        if any(kw in pub_lower for kw in kw_lower):
            year  = _extract_year(pub)
            title = pub[:120].strip()
            evidence.append({
                "source":  "publication",
                "title":   title,
                "year":    year,
                "weight":  _time_weight(year),
            })
    return evidence


def _evidence_from_teaching(teaching: list[str], keywords: list[str]) -> list[dict]:
    kw_lower = [k.lower() for k in keywords]
    hits = [t for t in teaching if any(kw in t.lower() for kw in kw_lower)]
    return [{"source": "teaching", "subject": h, "weight": 0.6} for h in hits]


def _evidence_from_research_areas(areas: list[str], keywords: list[str]) -> list[dict]:
    kw_lower = [k.lower() for k in keywords]
    hits = [a for a in areas if any(kw in a.lower() for kw in kw_lower)]
    return [{"source": "research_area", "area": h, "weight": 0.8} for h in hits]


# ══════════════════════════════════════════════════════════════
#  STEP 3 — Skill Strength Score
# ══════════════════════════════════════════════════════════════
def _skill_strength(evidence: list[dict]) -> float:
    """
    skill_strength = weighted sum of evidence sources (normalised 0–1)
    Weights: research_area 40%, publication 30% (×time_weight), teaching 10%
    """
    if not evidence:
        return 0.0

    score = 0.0
    for e in evidence:
        src = e.get("source", "")
        w   = e.get("weight", 0.5)
        if src == "research_area":
            score += 0.4 * w
        elif src == "publication":
            score += 0.3 * w
        elif src == "teaching":
            score += 0.1 * w

    # Clip to [0, 1] and round
    return round(min(score, 1.0), 3)


# ══════════════════════════════════════════════════════════════
#  MAIN EXTRACTOR
# ══════════════════════════════════════════════════════════════
def extract_profile(raw: dict) -> dict:
    """
    Given a raw profile dict, return a full Research Profile with:
    - technical_skills: [{skill, confidence, strength, evidence}]
    - research_topics:  [{topic, confidence, strength, evidence}]
    """
    corpus = _build_corpus(raw)
    pubs   = raw.get("publications_raw", [])
    teach  = raw.get("teaching_raw", [])
    areas  = raw.get("research_areas_raw", [])

    # ── Technical skills ──────────────────────────────────────
    skill_matches = _taxonomy_match(corpus, SKILL_TAXONOMY)
    tech_skills = []
    for skill, kws in skill_matches.items():
        evidence = (
            _evidence_from_publications(pubs, skill, kws) +
            _evidence_from_teaching(teach, kws) +
            _evidence_from_research_areas(areas, kws)
        )
        strength   = _skill_strength(evidence)
        confidence = min(0.5 + len(evidence) * 0.08, 0.99)
        tech_skills.append({
            "skill":      skill,
            "confidence": round(confidence, 2),
            "strength":   strength,
            "evidence":   evidence[:5],   # keep top 5
        })
    tech_skills.sort(key=lambda x: -x["strength"])

    # ── Research topics ───────────────────────────────────────
    topic_matches = _taxonomy_match(corpus, RESEARCH_TOPIC_TAXONOMY)
    research_topics = []
    for topic, kws in topic_matches.items():
        evidence = (
            _evidence_from_publications(pubs, topic, kws) +
            _evidence_from_research_areas(areas, kws)
        )
        strength   = _skill_strength(evidence)
        confidence = min(0.5 + len(evidence) * 0.08, 0.99)
        research_topics.append({
            "topic":      topic,
            "confidence": round(confidence, 2),
            "strength":   strength,
            "evidence":   evidence[:5],
        })
    research_topics.sort(key=lambda x: -x["strength"])

    return {
        "advisor_id":        raw.get("slug", ""),
        "name":              raw.get("name", ""),
        "title":             raw.get("title", ""),
        "department":        raw.get("department", ""),
        "profile_url":       raw.get("profile_url", ""),
        "email":             raw.get("email"),
        "education_raw":     raw.get("education_raw", []),
        "work_history_raw":  raw.get("work_history_raw", []),
        "teaching_raw":      raw.get("teaching_raw", []),
        "research_areas_raw":raw.get("research_areas_raw", []),
        "publication_count": len(pubs),
        "technical_skills":  tech_skills,
        "research_topics":   research_topics,
        "extracted_at":      datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def run(slug_filter: Optional[str] = None) -> None:
    import pandas as pd

    raw_files = sorted(PROFILES_DIR.glob("*.json"))
    if not raw_files:
        logger.error("No raw profiles found. Run lecturer_detail_crawler.py first.")
        sys.exit(1)

    if slug_filter:
        raw_files = [f for f in raw_files if slug_filter.lower() in f.stem.lower()]

    logger.info(f"Extracting skills from {len(raw_files)} profiles...")
    all_profiles = []

    for f in raw_files:
        raw = json.loads(f.read_text(encoding="utf-8"))
        if "error" in raw:
            logger.warning(f"Skip (crawl error): {f.stem}")
            continue
        profile = extract_profile(raw)
        all_profiles.append(profile)
        logger.info(
            f"  {profile['title']} {profile['name']}: "
            f"{len(profile['technical_skills'])} skills, "
            f"{len(profile['research_topics'])} topics"
        )

    # Save JSON
    OUTPUT_JSON.write_text(
        json.dumps(all_profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"\nResearch profiles → {OUTPUT_JSON}")

    # Save flat CSV for RL pipeline
    rows = []
    for p in all_profiles:
        rows.append({
            "advisor_id":       p["advisor_id"],
            "name":             p["name"],
            "title":            p["title"],
            "email":            p.get("email", ""),
            "publication_count":p["publication_count"],
            "technical_skills": ", ".join(s["skill"] for s in p["technical_skills"]),
            "top_skills_str":   ", ".join(
                f"{s['skill']}({s['strength']:.2f})"
                for s in p["technical_skills"][:8]
            ),
            "research_topics":  ", ".join(t["topic"] for t in p["research_topics"]),
            "top_topics_str":   ", ".join(
                f"{t['topic']}({t['strength']:.2f})"
                for t in p["research_topics"][:5]
            ),
            "teaching_subjects":"; ".join(p.get("teaching_raw", [])),
            "research_areas":   "; ".join(p.get("research_areas_raw", [])),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Skills CSV → {OUTPUT_CSV}")

    # Print summary
    print(f"\n{'='*65}")
    print(f"ADVISOR RESEARCH PROFILES — {len(all_profiles)} advisors")
    print(f"{'='*65}")
    for p in all_profiles:
        print(f"\n► {p['title']} {p['name']}")
        if p["technical_skills"]:
            top = p["technical_skills"][:5]
            skill_str = ", ".join(f"{s['skill']}({s['strength']:.2f})" for s in top)
            print(f"  Skills : {skill_str}")
        if p["research_topics"]:
            top = p["research_topics"][:3]
            topic_str = ", ".join(t["topic"] for t in top)
            print(f"  Topics : {topic_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", type=str, default=None)
    args = parser.parse_args()
    run(slug_filter=args.slug)
