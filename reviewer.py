"""Legacy import facade for the modular review implementation.

New code should import from :mod:`review_mcp`. Existing callers can continue to
import functions and configuration constants from ``reviewer``.
"""

# ruff: noqa: F401 - imports intentionally preserve the legacy module surface.

from review_mcp.config import (
    CONTEXT_DIRECTORY_SUFFIXES,
    DEFAULT_GLM_5_2_REASONING_EFFORT,
    DEFAULT_MAX_REVIEW_CONTEXT_CHARS,
    DEFAULT_MAX_TOOL_RESULT_CHARS,
    DEFAULT_MODEL_API_TIMEOUT_SECONDS,
    DEFAULT_READ_FILE_LINES,
    DEFAULT_REVIEW_MODEL,
    GLM_5_2_CONTEXT_WINDOW_TOKENS,
    GLM_5_2_MAX_OUTPUT_TOKENS,
    DEFAULT_SEARCH_RESULTS,
    DEFAULT_TREE_ENTRIES,
    EXCLUDE_PATTERNS,
    MAX_ALLOWED_ITERATIONS,
    MAX_CONTEXT_FILES_PER_DIRECTORY,
    MAX_READ_FILE_LINES,
    MAX_READ_LINE_CHARS,
    MAX_SEARCH_RESULTS,
    MAX_TREE_ENTRIES,
    OPENSPEC_ROOT_FILE_ORDER,
    REPOSITORY_SEARCH_TIMEOUT_SECONDS,
    _bounded_int,
    _env_flag,
    _get_glm_5_2_reasoning_effort,
    _get_max_iterations,
    _get_model_api_timeout_seconds,
    _get_optional_positive_int_env,
    _get_positive_int_env,
    logger,
)
from review_mcp.context import (
    _context_path_sort_key,
    _looks_like_openspec_change_dir,
    expand_context_entry,
    get_scoped_diff,
    normalize_file_uri_path,
    parse_file_links,
    read_context_file_with_links,
    read_context_files,
)
from review_mcp.repository import (
    _normalize_scope_files,
    _path_for_git,
    _repository_files,
    _repository_relative_path,
    _resolve_repository_path,
    _resolve_user_path,
    _run_git_command,
    get_changed_files,
    get_git_diff,
    list_repository_tree,
    read_repository_file,
    search_repository,
)
from review_mcp.review import (
    ReviewStatusCallback,
    _looks_like_unfinished_pseudo_tool_response,
    _notify_status,
    run_agentic_review,
)
from review_mcp.review_tools import (
    REVIEWER_TOOLS,
    _append_review_trace,
    _content_digest,
    _execute_tool,
    _message_content_to_text,
    _message_size_chars,
    _messages_size_chars,
    _normalized_tool_call_key,
    _truncate_content,
)
from review_mcp.transport import _create_streaming_chat_completion, _make_client
