"""Configuration constants and environment helpers for review-mcp."""

from __future__ import annotations

import logging
import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s",
    stream=sys.stderr,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("ReviewMCP")

EXCLUDE_PATTERNS = [
    "*.lock",
    "*.json",
    "*.svg",
    "*.png",
    "*.jpg",
    "*.woff",
    "*.woff2",
    "uv.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
]

DEFAULT_REVIEW_MODEL = "glm-5.3-flash"
DEFAULT_GLM_5_2_REASONING_EFFORT = "max"
# Request-level effort values shared across the reasoning-capable GLM 4.5+ family.
GLM_5_2_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}
DEFAULT_MODEL_API_TIMEOUT_SECONDS = 900.0
LIST_MODELS_TIMEOUT_SECONDS = 15.0
# Character counts are not model token counts. Keep this name as a compatibility
# export for callers that imported it, but do not apply a model-derived default.
# The provider's completion usage is the authoritative context measurement.
DEFAULT_MAX_REVIEW_CONTEXT_CHARS: int | None = None
DEFAULT_MAX_TOOL_RESULT_CHARS = 20000
DEFAULT_READ_FILE_LINES = 200
MAX_READ_FILE_LINES = 400
MAX_READ_LINE_CHARS = 2000
DEFAULT_TREE_ENTRIES = 200
MAX_TREE_ENTRIES = 500
DEFAULT_SEARCH_RESULTS = 20
MAX_SEARCH_RESULTS = 100
REPOSITORY_SEARCH_TIMEOUT_SECONDS = 10.0
MAX_CONTEXT_FILES_PER_DIRECTORY = 50
CONTEXT_DIRECTORY_SUFFIXES = {".md", ".txt", ".yaml", ".yml", ".json"}
OPENSPEC_ROOT_FILE_ORDER = {
    "proposal.md": 0,
    "design.md": 1,
    "tasks.md": 2,
}


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_glm_5_2_reasoning_effort() -> str:
    """Return the configured GLM-5.2 reasoning effort."""
    raw_value = os.getenv(
        "AI_REASONING_EFFORT",
        os.getenv("ZHIPU_REASONING_EFFORT", DEFAULT_GLM_5_2_REASONING_EFFORT),
    )
    value = raw_value.strip().lower()
    if value not in GLM_5_2_REASONING_EFFORTS:
        logger.warning(
            "Invalid reasoning effort %r; defaulting to %s",
            raw_value,
            DEFAULT_GLM_5_2_REASONING_EFFORT,
        )
        return DEFAULT_GLM_5_2_REASONING_EFFORT
    return value


_GLM_MODEL_VERSION_RE = re.compile(r"^glm-(\d+)(?:\.(\d+))?", re.IGNORECASE)
_MIN_REASONING_GLM_VERSION = (4, 5)


def _supports_reasoning(model_id: str) -> bool:
    """Return True when the model accepts reasoning-effort request controls."""
    match = _GLM_MODEL_VERSION_RE.match(model_id.strip())
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) >= _MIN_REASONING_GLM_VERSION


def _get_reasoning_effort(override: str | None = None) -> str:
    """Return the effective reasoning effort for reasoning-capable GLM models.

    An explicit per-call override wins when provided; otherwise the
    AI_REASONING_EFFORT / ZHIPU_REASONING_EFFORT environment settings apply.
    An invalid explicit override raises ValueError so the caller is told to
    pick a valid one; invalid environment values warn and fall back to the
    default.
    """
    if override is not None and override.strip():
        value = override.strip().lower()
        if value not in GLM_5_2_REASONING_EFFORTS:
            raise ValueError(
                f"Invalid reasoning_effort {override!r}. "
                "Pick one of: none, minimal, low, medium, high, xhigh, max."
            )
        return value
    return _get_glm_5_2_reasoning_effort()


def _get_review_model(override: str | None = None) -> str:
    """Return the effective review model id: per-call override, then env, then default."""
    if override:
        return override
    return os.getenv("AI_MODEL") or os.getenv("ZHIPU_MODEL") or DEFAULT_REVIEW_MODEL


def _get_model_api_timeout_seconds() -> float:
    """Return the timeout for each model API request."""
    raw_value = os.getenv(
        "AI_API_TIMEOUT_SECONDS",
        str(DEFAULT_MODEL_API_TIMEOUT_SECONDS),
    )
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid AI_API_TIMEOUT_SECONDS=%r; defaulting to %s",
            raw_value,
            DEFAULT_MODEL_API_TIMEOUT_SECONDS,
        )
        return DEFAULT_MODEL_API_TIMEOUT_SECONDS

    if value < 1:
        logger.warning(
            "AI_API_TIMEOUT_SECONDS must be >= 1; defaulting to %s",
            DEFAULT_MODEL_API_TIMEOUT_SECONDS,
        )
        return DEFAULT_MODEL_API_TIMEOUT_SECONDS
    return value


def _get_positive_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    """Return a positive integer environment setting with a safe fallback."""
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r; defaulting to %s", name, raw_value, default)
        return default
    if value < minimum:
        logger.warning("%s must be >= %s; defaulting to %s", name, minimum, default)
        return default
    return value


def _get_optional_positive_int_env(
    name: str,
    *,
    minimum: int = 1,
) -> int | None:
    """Return an explicit positive integer setting, or ``None`` when unset."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r; ignoring optional setting", name, raw_value)
        return None
    if value < minimum:
        logger.warning(
            "%s must be >= %s; ignoring optional setting", name, minimum
        )
        return None
    return value


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))
