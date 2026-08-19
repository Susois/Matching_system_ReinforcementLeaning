"""
gemini_extractor.py
───────────────────
Call Gemini API to extract structured thesis information from raw text.

The prompt instructs Gemini to return a JSON object whose keys map
1-to-1 to the THESIS_CSV_COLUMNS defined in configs/settings.py.
Using JSON output (instead of line-by-line KEY: value) is more robust
and avoids regex-based parsing fragility.
"""

import json
import logging
import re
import time
from typing import Optional

from google import genai
from google.genai import types as genai_types

from configs.settings import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_DELAY,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS,
    PDF_TEXT_LIMIT,
    THESIS_CSV_COLUMNS,
)

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────
_SYSTEM_PROMPT = """\
Bạn là trợ lý phân tích luận văn tốt nghiệp. Nhiệm vụ: đọc văn bản luận văn và
trả về đúng một JSON object với các trường bên dưới.

Quy tắc bắt buộc:
- Chỉ trả về JSON, không có text thêm trước/sau.
- Nếu không tìm thấy thông tin → giá trị là null.
- Liệt kê công nghệ dưới dạng chuỗi ngăn cách bởi dấu phẩy, ví dụ "PyTorch, TensorFlow".
- Lĩnh vực chính (field_category) chọn đúng một trong:
  "Phát triển Web", "Phát triển App (Mobile/Desktop)",
  "Hệ thống tích hợp đa nền tảng", "AI / Học sâu / GenAI",
  "Khoa học dữ liệu / Big Data", "Game / Đồ họa",
  "An toàn thông tin / Cloud", "IoT / Hệ thống nhúng", "Nghiên cứu khoa học".
- thesis_grade: chỉ điểm số (vd "8.5") nếu có, nếu không → null.
- student_id: mã số sinh viên (dãy số), ví dụ "11221234".

JSON schema (tất cả giá trị là string hoặc null):
{
  "student_id": ...,
  "student_name": ...,
  "major": ...,
  "completion_year": ...,
  "thesis_title": ...,
  "field_category": ...,
  "advisor_name": ...,
  "thesis_grade": ...,
  "web_languages": ...,
  "frontend_frameworks": ...,
  "backend_frameworks": ...,
  "database_cache": ...,
  "web_api_tech": ...,
  "app_languages": ...,
  "app_frameworks": ...,
  "app_db_backend": ...,
  "mobile_client_tech": ...,
  "architecture": ...,
  "ai_frameworks": ...,
  "ai_problems": ...,
  "data_tools": ...,
  "data_models": ...,
  "game_engine": ...,
  "game_type": ...,
  "specialty_field": ...,
  "tools_environment": ...,
  "hardware": ...,
  "iot_protocol": ...,
  "research_methods": ...,
  "research_output": ...
}
"""


def _build_prompt(text: str) -> str:
    truncated = text[:PDF_TEXT_LIMIT]
    return f"{_SYSTEM_PROMPT}\n\nVĂN BẢN LUẬN VĂN:\n{truncated}"


def _clean_json_response(raw: str) -> str:
    """Strip markdown code fences that Gemini sometimes adds."""
    raw = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _parse_response(raw_text: str, source_file: str) -> dict:
    """Parse Gemini JSON response into a flat dict."""
    cleaned = _clean_json_response(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for {source_file}: {e}. Raw: {cleaned[:300]}")
        data = {}

    # Ensure all schema columns present; fill missing with None
    result: dict = {}
    for col in THESIS_CSV_COLUMNS:
        if col in ("source_file", "extraction_status", "extraction_notes"):
            continue
        val = data.get(col)
        # Normalise empty / placeholder strings to None
        if isinstance(val, str) and val.strip().lower() in {
            "", "null", "none", "không áp dụng", "không có", "n/a", "na", "không tìm thấy"
        }:
            val = None
        result[col] = val

    return result


class GeminiExtractor:
    """Thin wrapper around Gemini SDK (google.genai) for thesis extraction."""

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Create a .env file with GOOGLE_API_KEY=<your-key>."
            )
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = GEMINI_MODEL
        logger.info(f"GeminiExtractor ready — model: {self.model_name}")

    def extract(self, text: str, source_file: str) -> dict:
        """
        Send text to Gemini and return a dict with extracted fields.
        On failure returns a dict with extraction_status='failed'.
        """
        if not text or not text.strip():
            logger.warning(f"Empty text for {source_file}; skipping Gemini call.")
            return self._failed_record(source_file, "empty_text")

        prompt = _build_prompt(text)

        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                logger.info(f"Gemini call [{attempt}/{GEMINI_MAX_RETRIES}]: {source_file}")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=GEMINI_TEMPERATURE,
                        max_output_tokens=GEMINI_MAX_TOKENS,
                    ),
                )
                raw = response.text or ""
                if not raw.strip():
                    raise ValueError("Gemini returned empty response.")

                record = _parse_response(raw, source_file)
                record["source_file"]       = source_file
                record["extraction_status"] = "success"
                record["extraction_notes"]  = None
                logger.info(
                    f"Extracted OK: {source_file} → "
                    f"advisor='{record.get('advisor_name')}', "
                    f"title='{str(record.get('thesis_title',''))[:60]}'"
                )
                return record

            except Exception as e:
                logger.warning(f"Attempt {attempt} failed for {source_file}: {e}")
                if attempt < GEMINI_MAX_RETRIES:
                    time.sleep(GEMINI_RETRY_DELAY * attempt)

        return self._failed_record(source_file, "max_retries_exceeded")

    @staticmethod
    def _failed_record(source_file: str, reason: str) -> dict:
        record = {col: None for col in THESIS_CSV_COLUMNS}
        record["source_file"]       = source_file
        record["extraction_status"] = "failed"
        record["extraction_notes"]  = reason
        return record
