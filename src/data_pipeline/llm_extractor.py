"""
llm_extractor.py
────────────────
Gọi LLM để trích xuất thông tin có cấu trúc từ text khóa luận.

Hỗ trợ 2 provider (chọn qua LLM_PROVIDER trong .env):
  - "openai_compat"  →  9router (auto model rotation)
  - "gemini"         →  Google Gemini

Model rotation (chỉ với openai_compat):
  - Thử lần lượt từng model trong NINEROUTER_MODEL_LIST
  - Khi model hiện tại bị 429 / rate-limit / quota / not-found → tự động
    chuyển sang model tiếp theo
  - Sau khi hết vòng danh sách → chờ 60s rồi bắt đầu lại từ đầu
  - Trạng thái model hiện tại được lưu trong class (persist giữa các lần
    gọi trong cùng một run)

Cấu hình trong .env:
    LLM_PROVIDER=openai_compat
    NINEROUTER_BASE_URL=http://localhost:20128/v1
    NINEROUTER_API_KEY=sk-...
    NINEROUTER_MODEL_LIST=kr/claude-haiku-4.5,ag/gemini-3-flash,...  (optional)
"""

import json
import logging
import re
import time
from typing import Optional

from configs.settings import (
    LLM_PROVIDER,
    NINEROUTER_BASE_URL,
    NINEROUTER_API_KEY,
    NINEROUTER_MODEL_LIST,
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

# ── Prompt (dùng chung) ───────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a data extraction assistant. I will give you the text of a Vietnamese university thesis. \
Please extract specific information and return it as a single JSON object.

Return ONLY valid JSON with these exact keys (use null if information is not found):
student_id, student_name, major, completion_year, thesis_title, field_category,
advisor_name, thesis_grade, web_languages, frontend_frameworks, backend_frameworks,
database_cache, web_api_tech, app_languages, app_frameworks, app_db_backend,
mobile_client_tech, architecture, ai_frameworks, ai_problems, data_tools,
data_models, game_engine, game_type, specialty_field, tools_environment,
hardware, iot_protocol, research_methods, research_output

Guidelines:
- field_category must be exactly one of: "Phát triển Web", "Phát triển App (Mobile/Desktop)", \
"Hệ thống tích hợp đa nền tảng", "AI / Học sâu / GenAI", "Khoa học dữ liệu / Big Data", \
"Game / Đồ họa", "An toàn thông tin / Cloud", "IoT / Hệ thống nhúng", "Nghiên cứu khoa học"
- List technologies as comma-separated strings e.g. "PyTorch, TensorFlow, YOLO"
- thesis_grade: numeric grade only e.g. "8.5", null if not found
- student_id: numeric student ID e.g. "11221234"
- No markdown, no explanation, just the JSON object\
"""

_EMPTY_VALS = {
    "", "null", "none", "không áp dụng", "không có",
    "n/a", "na", "không tìm thấy",
}

# Patterns nhận dạng lỗi rate-limit / quota / model không khả dụng
_LIMIT_PATTERNS = re.compile(
    r"429|rate.?limit|quota|resource.?exhaust|too.?many|"
    r"context.?length|model.?not.?found|no.?active.?cred|"
    r"not.?supported|overloaded|capacity",
    re.IGNORECASE,
)

# Patterns nhận dạng model từ chối / refusal (không phải JSON)
_REFUSAL_PATTERNS = re.compile(
    r"i can'?t|i cannot|i'm unable|i am unable|"
    r"i won'?t|i will not|i refuse|"
    r"override.*instruction|safety|policy|harmful|"
    r"attempting to|this request",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────
def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _parse_response(raw: str, source_file: str) -> dict:
    cleaned = _clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error [{source_file}]: {e} | raw: {cleaned[:200]}")
        data = {}

    result: dict = {}
    for col in THESIS_CSV_COLUMNS:
        if col in ("source_file", "extraction_status", "extraction_notes"):
            continue
        val = data.get(col)
        if isinstance(val, str) and val.strip().lower() in _EMPTY_VALS:
            val = None
        result[col] = val
    return result


def _is_limit_error(err: str) -> bool:
    return bool(_LIMIT_PATTERNS.search(err))


def _parse_retry_after(err: str, default: float = 30.0) -> float:
    """Lấy thời gian chờ từ error message nếu có (vd 'retry in 19s')."""
    m = re.search(r"retry.{0,10}in\s+(\d+(?:\.\d+)?)\s*s", err, re.IGNORECASE)
    return float(m.group(1)) + 2 if m else default


# ── OpenAI-compatible backend với model rotation ──────────────
class _OpenAICompatBackend:
    """
    Gọi 9router qua OpenAI SDK.
    Tự động rotate qua NINEROUTER_MODEL_LIST khi gặp limit/lỗi.
    """

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Chạy: pip install openai")

        if not NINEROUTER_API_KEY:
            raise EnvironmentError("NINEROUTER_API_KEY chưa được set trong .env")

        from openai import OpenAI
        self.client    = OpenAI(base_url=NINEROUTER_BASE_URL, api_key=NINEROUTER_API_KEY)
        self.models    = list(NINEROUTER_MODEL_LIST)   # copy để không mutate config
        self._idx      = 0                             # model hiện tại
        self._failures: dict[str, int] = {}            # model → số lần lỗi liên tiếp

        logger.info(
            f"9router backend | {len(self.models)} models | "
            f"start: {self.models[0]} | url: {NINEROUTER_BASE_URL}"
        )
        logger.info(f"Model list: {self.models}")

    @property
    def current_model(self) -> str:
        return self.models[self._idx % len(self.models)]

    def _next_model(self, reason: str) -> None:
        """Chuyển sang model tiếp theo và log lý do."""
        old = self.current_model
        self._failures[old] = self._failures.get(old, 0) + 1
        self._idx += 1

        if self._idx >= len(self.models):
            # Đã hết vòng — reset và chờ
            self._idx = 0
            wait = 15
            logger.warning(
                f"Đã thử hết {len(self.models)} model. "
                f"Chờ {wait}s rồi bắt đầu lại từ {self.current_model}..."
            )
            time.sleep(wait)
        else:
            logger.warning(
                f"Model [{old}] bị limit ({reason[:15]}). "
                f"Chuyển sang [{self.current_model}]"
            )

    def call(self, text: str, source_file: str) -> str:
        """
        Gọi LLM với auto-rotate. Thử tối đa len(models) * 2 lần.
        Raise Exception nếu tất cả model đều fail.
        """
        user_msg = f"VĂN BẢN LUẬN VĂN:\n{text[:PDF_TEXT_LIMIT]}"
        max_attempts = len(self.models) * 2
        last_err = ""

        for attempt in range(max_attempts):
            model = self.current_model
            try:
                logger.info(
                    f"  [{attempt+1}/{max_attempts}] model={model} | {source_file}"
                )
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=GEMINI_TEMPERATURE,
                    max_tokens=GEMINI_MAX_TOKENS,
                )
                raw = resp.choices[0].message.content or ""
                if not raw.strip():
                    raise ValueError("Empty response from model")

                # Detect refusal — model từ chối xử lý → rotate sang model khác
                if _REFUSAL_PATTERNS.search(raw[:300]):
                    raise ValueError(f"Model refused: {raw[:120]}")

                # Reset failure count cho model này vì thành công
                self._failures[model] = 0
                return raw

            except Exception as e:
                last_err = str(e)
                if _is_limit_error(last_err) or "refused" in last_err.lower() or "Model refused" in last_err:
                    wait = _parse_retry_after(last_err, default=GEMINI_RETRY_DELAY)
                    # Refusal không cần chờ lâu, chỉ cần rotate
                    if "refused" in last_err.lower() or "Model refused" in last_err:
                        wait = 1
                    logger.warning(
                        f"  Limit/refused [{model}]: {last_err[:80]} "
                        f"| chờ {wait:.0f}s rồi rotate..."
                    )
                    time.sleep(wait)
                    self._next_model(last_err)
                else:
                    # Lỗi khác (network, parse, v.v.) — thử lại cùng model 1 lần
                    logger.warning(f"  Error [{model}]: {last_err[:80]}")
                    if attempt % 2 == 1:   # sau 2 lần cùng model → rotate
                        self._next_model(last_err)
                    else:
                        time.sleep(GEMINI_RETRY_DELAY)

        raise RuntimeError(
            f"Tất cả {len(self.models)} model đều fail cho {source_file}. "
            f"Lỗi cuối: {last_err[:200]}"
        )


# ── Gemini backend ────────────────────────────────────────────
class _GeminiBackend:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise EnvironmentError("GOOGLE_API_KEY chưa được set trong .env")
        from google import genai
        from google.genai import types as genai_types
        self._types = genai_types
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model  = GEMINI_MODEL
        logger.info(f"Gemini backend | model: {self.model}")

    def call(self, text: str, source_file: str) -> str:
        prompt = f"{_SYSTEM_PROMPT}\n\nVĂN BẢN LUẬN VĂN:\n{text[:PDF_TEXT_LIMIT]}"
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_TOKENS,
            ),
        )
        return resp.text or ""


# ── Public LLMExtractor ───────────────────────────────────────
class LLMExtractor:
    """
    Provider-agnostic extractor.
    - LLM_PROVIDER=openai_compat  →  9router với auto model rotation
    - LLM_PROVIDER=gemini         →  Google Gemini
    """

    def __init__(self):
        if LLM_PROVIDER == "gemini":
            self._backend = _GeminiBackend()
        else:
            self._backend = _OpenAICompatBackend()

    def extract(self, text: str, source_file: str) -> dict:
        if not text or not text.strip():
            logger.warning(f"Empty text: {source_file}")
            return self._failed(source_file, "empty_text")

        try:
            raw    = self._backend.call(text, source_file)
            record = _parse_response(raw, source_file)
            record["source_file"]       = source_file
            record["extraction_status"] = "success"
            record["extraction_notes"]  = None
            logger.info(
                f"OK: {source_file} → "
                f"advisor='{record.get('advisor_name')}' | "
                f"'{str(record.get('thesis_title', ''))[:55]}'"
            )
            return record

        except Exception as e:
            logger.error(f"FAILED {source_file}: {e}")
            return self._failed(source_file, str(e)[:200])

    @staticmethod
    def _failed(source_file: str, reason: str) -> dict:
        rec = {col: None for col in THESIS_CSV_COLUMNS}
        rec["source_file"]       = source_file
        rec["extraction_status"] = "failed"
        rec["extraction_notes"]  = reason
        return rec

    def model_status(self) -> str:
        """Log trạng thái model rotation (dùng để debug)."""
        if isinstance(self._backend, _OpenAICompatBackend):
            b = self._backend
            return (
                f"Current: {b.current_model} | "
                f"Index: {b._idx}/{len(b.models)} | "
                f"Failures: {b._failures}"
            )
        return f"Gemini: {GEMINI_MODEL}"
