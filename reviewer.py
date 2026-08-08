"""
Code review logic using Zhipu GLM via an OpenAI-compatible API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote

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

MAX_ALLOWED_ITERATIONS = 50
DEFAULT_REVIEW_MODEL = "glm-5.2"
DEFAULT_MODEL_API_TIMEOUT_SECONDS = 900.0
# GLM-5.2 becomes unreliable well before its advertised context window when a
# tool loop accumulates large source excerpts. Keep the default working set
# small enough that a final response is still practical after several reads.
DEFAULT_MAX_REVIEW_CONTEXT_CHARS = 45000
DEFAULT_MAX_TOOL_RESULT_CHARS = 8000
DEFAULT_READ_FILE_LINES = 120
MAX_READ_FILE_LINES = 200
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

ReviewStatusCallback = Callable[[str], None]


def _notify_status(
    status_callback: ReviewStatusCallback | None,
    message: str,
) -> None:
    """Send a best-effort status update to the MCP host."""
    if status_callback is None:
        return
    try:
        status_callback(message)
    except Exception:
        logger.debug("Status callback failed", exc_info=True)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_max_iterations() -> int:
    """Return a safe iteration limit from the environment."""
    raw_value = os.getenv("MAX_REVIEW_ITERATIONS", "20")
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid MAX_REVIEW_ITERATIONS=%r; defaulting to 20", raw_value)
        return 20

    if value < 1:
        logger.warning("MAX_REVIEW_ITERATIONS must be >= 1; defaulting to 20")
        return 20
    if value > MAX_ALLOWED_ITERATIONS:
        logger.warning(
            "MAX_REVIEW_ITERATIONS=%s is too high; capping at %s",
            value,
            MAX_ALLOWED_ITERATIONS,
        )
        return MAX_ALLOWED_ITERATIONS
    return value


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


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _run_git_command(
    working_dir: str | Path,
    args: list[str],
    *,
    check: bool = True,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(working_dir),
        capture_output=True,
        text=True,
        check=check,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        timeout=timeout,
    )


def _resolve_user_path(path: str | Path, working_dir: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(working_dir) / candidate


def _path_for_git(path: str | Path, working_dir: str | Path) -> str:
    resolved = _resolve_user_path(path, working_dir)
    try:
        return resolved.relative_to(Path(working_dir)).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_repository_path(path: str | Path, working_dir: str | Path) -> Path:
    """Resolve a model-requested path and keep it inside the repository."""
    root = Path(working_dir).resolve()
    candidate = _resolve_user_path(path or ".", root).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the repository: {path}") from exc
    return candidate


def _repository_relative_path(path: Path, working_dir: str | Path) -> str:
    relative = path.relative_to(Path(working_dir).resolve()).as_posix()
    return relative or "."


def _repository_files(working_dir: str | Path) -> list[str]:
    """List tracked and untracked, non-ignored repository files."""
    try:
        result = _run_git_command(
            working_dir,
            ["ls-files", "--cached", "--others", "--exclude-standard"],
        )
        return sorted({line for line in result.stdout.splitlines() if line})
    except (subprocess.CalledProcessError, FileNotFoundError):
        root = Path(working_dir).resolve()
        files: list[str] = []
        for candidate in root.rglob("*"):
            relative = candidate.relative_to(root)
            if any(
                part == ".git" or part.startswith(".venv") for part in relative.parts
            ):
                continue
            if candidate.is_file():
                files.append(relative.as_posix())
        return sorted(files)


def list_repository_tree(
    working_dir: str | Path,
    path: str = ".",
    *,
    max_depth: int = 3,
    max_entries: int = DEFAULT_TREE_ENTRIES,
) -> str:
    """Return a bounded tree of repository files visible to Git."""
    try:
        requested = _resolve_repository_path(path, working_dir)
    except ValueError as exc:
        return f"Error: {exc}"
    if not requested.exists():
        return f"Error: Repository path does not exist: {path}"
    if not requested.is_dir():
        return f"Error: Repository path is not a directory: {path}"

    max_depth = _bounded_int(max_depth, 3, 1, 6)
    max_entries = _bounded_int(
        max_entries,
        DEFAULT_TREE_ENTRIES,
        1,
        MAX_TREE_ENTRIES,
    )
    prefix = _repository_relative_path(requested, working_dir)
    prefix_with_slash = "" if prefix == "." else f"{prefix}/"
    entries: set[str] = set()

    for filepath in _repository_files(working_dir):
        if prefix_with_slash and not filepath.startswith(prefix_with_slash):
            continue
        scoped_path = filepath[len(prefix_with_slash) :]
        parts = Path(scoped_path).parts
        if not parts:
            continue
        visible_parts = parts[:max_depth]
        for depth in range(1, len(visible_parts) + 1):
            entry = "/".join(visible_parts[:depth])
            if depth < len(parts):
                entry += "/"
            entries.add(entry)

    ordered = sorted(entries, key=lambda item: (item.count("/"), item))
    truncated = len(ordered) > max_entries
    ordered = ordered[:max_entries]
    lines = [prefix]
    for entry in ordered:
        depth = entry.rstrip("/").count("/")
        lines.append(
            f"{'  ' * depth}{entry.split('/')[-1] or entry.split('/')[-2]}/"
            if entry.endswith("/")
            else f"{'  ' * depth}{entry.split('/')[-1]}"
        )
    if truncated:
        lines.append(
            f"... [TRUNCATED after {max_entries} entries; narrow path or depth]"
        )
    return "\n".join(lines)


def read_repository_file(
    filepath: str,
    working_dir: str | Path,
    *,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read a bounded, line-numbered excerpt from one repository file."""
    try:
        resolved = _resolve_repository_path(filepath, working_dir)
    except ValueError as exc:
        return f"Error: {exc}"
    if not resolved.is_file():
        return f"Error: Repository file does not exist or is not readable: {filepath}"

    try:
        start_line = max(1, int(start_line))
        requested_end = (
            int(end_line)
            if end_line is not None
            else start_line + DEFAULT_READ_FILE_LINES - 1
        )
    except (TypeError, ValueError):
        return "Error: start_line and end_line must be integers."
    requested_end = max(start_line, requested_end)
    bounded_end = min(requested_end, start_line + MAX_READ_FILE_LINES - 1)

    relative = _repository_relative_path(resolved, working_dir)
    selected_lines: list[tuple[int, str]] = []
    lines_seen = 0
    more_available = False
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                lines_seen = line_number
                if line_number < start_line:
                    continue
                if line_number > bounded_end:
                    more_available = True
                    break
                content = line.rstrip("\n\r")
                if len(content) > MAX_READ_LINE_CHARS:
                    content = content[:MAX_READ_LINE_CHARS] + "... [LINE TRUNCATED]"
                selected_lines.append((line_number, content))
    except UnicodeDecodeError:
        return f"Error: Repository file is not UTF-8 text: {filepath}"
    except OSError as exc:
        return f"Error reading repository file '{filepath}': {exc}"

    if not selected_lines:
        return (
            f"{relative} has {lines_seen} line(s); start_line {start_line} is past EOF."
        )
    actual_end = selected_lines[-1][0]
    width = len(str(actual_end))
    excerpt = "\n".join(
        f"{line_number:>{width}} | {line}" for line_number, line in selected_lines
    )
    notes: list[str] = []
    if requested_end > bounded_end:
        notes.append(f"requested range capped at {MAX_READ_FILE_LINES} lines")
    if more_available:
        notes.append(f"more content available after line {actual_end}")
    suffix = f"\n... [{'; '.join(notes)}]" if notes else ""
    return (
        f"--- FILE: {relative} (lines {start_line}-{actual_end}) ---\n{excerpt}{suffix}"
    )


def search_repository(
    query: str,
    working_dir: str | Path,
    *,
    path: str = ".",
    file_glob: str | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    max_results: int = DEFAULT_SEARCH_RESULTS,
) -> str:
    """Search repository text with ripgrep and return bounded matching lines."""
    if not query:
        return "Error: search query must not be empty."
    try:
        requested = _resolve_repository_path(path, working_dir)
    except ValueError as exc:
        return f"Error: {exc}"
    if not requested.exists():
        return f"Error: Repository path does not exist: {path}"

    max_results = _bounded_int(
        max_results,
        DEFAULT_SEARCH_RESULTS,
        1,
        MAX_SEARCH_RESULTS,
    )
    relative_path = _repository_relative_path(requested, working_dir)
    args = ["rg", "--line-number", "--column", "--no-heading", "--color=never"]
    if not regex:
        args.append("--fixed-strings")
    if not case_sensitive:
        args.append("--ignore-case")
    if file_glob:
        args.extend(["--glob", file_glob])
    args.extend(["--", query, relative_path])

    try:
        process = subprocess.Popen(
            args,
            cwd=str(Path(working_dir).resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        return (
            "Error: ripgrep (rg) is required for repository search but was not found."
        )

    matches: list[str] = []
    errors: list[str] = []
    output_limit_reached = False
    assert process.stdout is not None
    assert process.stderr is not None

    def collect_matches() -> None:
        nonlocal output_limit_reached
        for line in process.stdout:
            matches.append(line.rstrip("\n")[:1000])
            if len(matches) >= max_results:
                output_limit_reached = True
                process.terminate()
                break

    def collect_errors() -> None:
        for line in process.stderr:
            if len(errors) < 20:
                errors.append(line.rstrip("\n")[:1000])

    stdout_thread = threading.Thread(target=collect_matches, daemon=True)
    stderr_thread = threading.Thread(target=collect_errors, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join(REPOSITORY_SEARCH_TIMEOUT_SECONDS)
    timed_out = stdout_thread.is_alive()
    if timed_out:
        process.kill()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)
    for stream in (process.stdout, process.stderr):
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    if timed_out:
        return (
            "Error: Repository search timed out after "
            f"{REPOSITORY_SEARCH_TIMEOUT_SECONDS:g} seconds. Narrow the query or path."
        )
    stderr = "\n".join(errors)

    if not matches:
        if process.returncode not in {0, 1, -15} and stderr.strip():
            return f"Error searching repository: {stderr.strip()}"
        return "No matches found."
    suffix = (
        f"\n... [TRUNCATED after {max_results} matches; narrow the query or path]"
        if output_limit_reached
        else ""
    )
    return "\n".join(matches) + suffix


def _looks_like_openspec_change_dir(path: Path) -> bool:
    """Detect an explicitly provided OpenSpec change directory."""
    if not path.is_dir():
        return False

    has_openspec_layout = (path / "proposal.md").exists() or (
        path / "tasks.md"
    ).exists()
    has_specs_dir = (path / "specs").is_dir()
    return has_openspec_layout or has_specs_dir


def _context_path_sort_key(path: Path, root: Path) -> tuple[int, str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    if len(relative.parts) == 1:
        priority = OPENSPEC_ROOT_FILE_ORDER.get(relative.name, 10)
    elif relative.parts[0] == "specs":
        priority = 20
    else:
        priority = 30
    return priority, relative.as_posix()


def expand_context_entry(path: str | Path, working_dir: str | Path) -> list[Path]:
    """Expand a context file entry into one or more readable context files."""
    resolved = _resolve_user_path(path, working_dir)
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        return []
    if not _looks_like_openspec_change_dir(resolved):
        return []

    files = [
        candidate
        for candidate in resolved.rglob("*")
        if candidate.is_file()
        and not any(
            part.startswith(".") for part in candidate.relative_to(resolved).parts
        )
        and candidate.suffix.lower() in CONTEXT_DIRECTORY_SUFFIXES
    ]
    return sorted(
        files, key=lambda candidate: _context_path_sort_key(candidate, resolved)
    )[:MAX_CONTEXT_FILES_PER_DIRECTORY]


def _make_client():
    from openai import OpenAI

    api_key = os.getenv("AI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError(
            "AI_API_KEY environment variable is not set. "
            "ZHIPU_API_KEY is also accepted for backward compatibility."
        )

    base_url = os.getenv("ZHIPU_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=_get_model_api_timeout_seconds(),
    )


def _normalize_scope_files(
    scope_files: list[str] | None,
    working_dir: str | Path,
) -> list[str] | None:
    """Normalize scope files into stable git-friendly paths."""
    if not scope_files:
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for filepath in scope_files:
        candidate = _path_for_git(filepath, working_dir)
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def get_git_diff(
    working_dir: str | Path,
    target: str = "staged",
    scope_files: list[str] | None = None,
) -> str:
    """Get git diff output for the requested target."""
    try:
        if target == "all":
            staged = get_git_diff(working_dir, "staged", scope_files=scope_files)
            unstaged = get_git_diff(working_dir, "unstaged", scope_files=scope_files)
            sections = []
            if staged.strip():
                sections.append(f"# Staged changes\n{staged}")
            if unstaged.strip():
                sections.append(f"# Unstaged changes\n{unstaged}")
            return "\n\n".join(sections)
        if target == "staged":
            args = ["diff", "--staged"]
        elif target == "unstaged":
            args = ["diff"]
        else:
            args = ["diff", target]

        normalized_scope = _normalize_scope_files(scope_files, working_dir)
        if normalized_scope:
            args.append("--")
            args.extend(normalized_scope)
        elif EXCLUDE_PATTERNS:
            args.append("--")
            args.extend(f":!{pattern}" for pattern in EXCLUDE_PATTERNS)

        return _run_git_command(working_dir, args).stdout
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        return f"Error running git diff: {stderr or exc}"
    except FileNotFoundError:
        return "Error: git is not installed or not in PATH."


def get_changed_files(
    working_dir: str | Path,
    scope_files: list[str] | None = None,
) -> list[str]:
    """Get the list of changed files (staged + unstaged)."""
    try:
        staged = _run_git_command(working_dir, ["diff", "--staged", "--name-only"])
        unstaged = _run_git_command(working_dir, ["diff", "--name-only"])
        files = set(
            staged.stdout.strip().splitlines() + unstaged.stdout.strip().splitlines()
        )
        changed_files = sorted(file for file in files if file)

        normalized_scope = _normalize_scope_files(scope_files, working_dir)
        if normalized_scope:
            scope_set = set(normalized_scope)
            changed_files = [
                filepath
                for filepath in changed_files
                if _path_for_git(filepath, working_dir) in scope_set
            ]

        return changed_files
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def read_context_files(
    filepaths: list[str] | str | None, working_dir: str | Path
) -> str:
    """Read multiple context files and format them for the prompt."""
    if not filepaths:
        return ""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    filepaths = [filepath for filepath in filepaths if filepath and filepath.strip()]
    if not filepaths:
        return ""

    context_chunks: list[str] = []
    for filepath in filepaths:
        resolved_paths = expand_context_entry(filepath, working_dir)
        if not resolved_paths:
            resolved = _resolve_user_path(filepath, working_dir)
            context_chunks.append(
                f"\n\n(Note: Context file '{resolved}' not found or not readable)\n"
            )
            continue

        if len(resolved_paths) > 1:
            source = _resolve_user_path(filepath, working_dir)
            context_chunks.append(f"\n\n--- CONTEXT DIRECTORY: {source} ---")

        for resolved in resolved_paths:
            try:
                content = resolved.read_text(encoding="utf-8")
                if len(content) > 50000:
                    content = content[:50000] + "\n\n... [TRUNCATED] ..."
                context_chunks.append(f"\n\n--- FILE: {resolved} ---\n{content}")
            except FileNotFoundError:
                context_chunks.append(
                    f"\n\n(Note: Context file '{resolved}' not found)\n"
                )
            except Exception as exc:
                context_chunks.append(f"\n\n(Error reading '{resolved}': {exc})\n")
    return "".join(context_chunks)


def read_context_file_with_links(
    filepath: str | Path,
    working_dir: str | Path,
    diff_target: str = "staged",
) -> tuple[str, list[str]]:
    """Read one context file and resolve render_diffs/file links."""
    resolved = _resolve_user_path(filepath, working_dir)
    try:
        content = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"(Context file not found: {resolved})", []
    except Exception as exc:
        return f"(Error reading context file '{resolved}': {exc})", []

    diff_files, read_files = parse_file_links(content)

    def replace_render_diff(match: re.Match[str]) -> str:
        full_path = normalize_file_uri_path(unquote(match.group(1)))
        file_diff = get_scoped_diff([full_path], working_dir, diff_target)
        if file_diff:
            return f"```diff\n{file_diff}\n```"
        return f"(No changes in {full_path})"

    processed = re.sub(
        r"render_diffs\s*\(\s*file:///([^)]+)\s*\)",
        replace_render_diff,
        content,
    )

    linked_context = ""
    for read_path in read_files[:5]:
        try:
            linked_file = _resolve_user_path(read_path, working_dir)
            file_content = linked_file.read_text(encoding="utf-8")
            if len(file_content) > 10000:
                file_content = file_content[:10000] + "\n... [TRUNCATED]"
            linked_context += f"\n\n--- LINKED FILE: {linked_file} ---\n{file_content}"
        except Exception:
            pass

    return processed + linked_context, diff_files


def normalize_file_uri_path(path: str) -> str:
    """Normalize a path extracted from a file:// URI."""
    if len(path) > 1 and path[1] == ":":
        return path
    if not path.startswith("/"):
        return "/" + path
    return path


def parse_file_links(content: str) -> tuple[list[str], list[str]]:
    """Parse markdown content for render_diffs() and file:// links."""
    diff_files: list[str] = []
    read_files: list[str] = []

    render_diff_pattern = r"render_diffs\s*\(\s*file:///([^)]+)\s*\)"
    for match in re.finditer(render_diff_pattern, content):
        diff_files.append(normalize_file_uri_path(unquote(match.group(1))))

    link_pattern = r"\[([^\]]*)\]\(file:///([^)]+)\)"
    for match in re.finditer(link_pattern, content):
        full_path = normalize_file_uri_path(unquote(match.group(2)))
        if full_path not in diff_files:
            read_files.append(full_path)

    return diff_files, read_files


def get_scoped_diff(
    files: list[str],
    working_dir: str | Path,
    target: str = "staged",
) -> str:
    """Get git diff for specific files only."""
    if not files:
        return ""

    diffs: list[str] = []
    for filepath in files:
        try:
            normalized = _path_for_git(filepath, working_dir)
            if target == "staged":
                args = ["diff", "--staged", "--", normalized]
            elif target == "unstaged":
                args = ["diff", "--", normalized]
            else:
                args = ["diff", target, "--", normalized]

            result = _run_git_command(working_dir, args)
            if result.stdout.strip():
                diffs.append(f"# Diff for: {filepath}\n{result.stdout}")
        except subprocess.CalledProcessError:
            diffs.append(f"# No diff available for: {filepath}")
        except Exception as exc:
            diffs.append(f"# Error getting diff for {filepath}: {exc}")

    return "\n\n".join(diffs)


def _append_review_trace(review: str, trace: list[str], enabled: bool) -> str:
    if not enabled:
        return review
    if not trace:
        return f"{review}\n\n---\n## Review Trace\n- No trace events recorded."
    trace_lines = "\n".join(f"- {item}" for item in trace)
    return f"{review}\n\n---\n## Review Trace\n{trace_lines}"


def _message_content_to_text(content: object) -> str:
    """Normalize provider message content into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _message_size_chars(message: dict | object) -> int:
    """Estimate request size using message content and tool-call arguments."""
    if isinstance(message, dict):
        size = len(str(message.get("content", "") or ""))
        tool_calls = message.get("tool_calls", []) or []
    else:
        size = len(_message_content_to_text(getattr(message, "content", "")))
        tool_calls = getattr(message, "tool_calls", []) or []

    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        if function is not None:
            size += len(str(getattr(function, "name", "") or ""))
            size += len(str(getattr(function, "arguments", "") or ""))
    return size


def _messages_size_chars(messages: list[dict | object]) -> int:
    return sum(_message_size_chars(message) for message in messages)


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _truncate_content(content: str, limit: int, label: str) -> tuple[str, bool]:
    """Truncate text to a hard character limit while preserving a clear marker."""
    if len(content) <= limit:
        return content, False
    marker = f"\n... [{label} TRUNCATED at {limit} chars]"
    keep = max(0, limit - len(marker))
    return content[:keep] + marker, True


def _normalized_tool_call_key(
    name: str,
    arguments: dict,
    working_dir: str | Path,
) -> str:
    """Normalize semantically equivalent tool requests for deduplication."""
    normalized = dict(arguments) if isinstance(arguments, dict) else {}
    if name == "get_uncommitted_changes":
        normalized = {"target": normalized.get("target", "all")}
    elif name == "list_changed_files":
        normalized = {}
    elif name == "read_file":
        path = normalized.get("path", "")
        try:
            path = _repository_relative_path(
                _resolve_repository_path(path, working_dir), working_dir
            )
        except ValueError:
            pass
        try:
            start_line = int(normalized.get("start_line", 1))
        except (TypeError, ValueError):
            start_line = 1
        end_line = normalized.get("end_line")
        if end_line is None:
            end_line = start_line + DEFAULT_READ_FILE_LINES - 1
        normalized = {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
        }
    return f"{name}:{json.dumps(normalized, sort_keys=True, default=str)}"


def _execute_tool(
    name: str,
    arguments: dict,
    *,
    working_dir: str | Path,
) -> str:
    """Execute a model tool and return its result as a string."""
    if not isinstance(arguments, dict):
        arguments = {}

    if name == "get_uncommitted_changes":
        return get_git_diff(working_dir, arguments.get("target", "all"))
    if name == "read_files":
        paths = arguments.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list):
            return "Error: read_files paths must be a list of repository file paths."
        excerpts = [read_repository_file(path, working_dir) for path in paths[:10]]
        return "\n\n".join(excerpts)
    if name == "read_file":
        return read_repository_file(
            arguments.get("path", ""),
            working_dir,
            start_line=arguments.get("start_line", 1),
            end_line=arguments.get("end_line"),
        )
    if name == "list_changed_files":
        files = get_changed_files(working_dir)
        return "\n".join(files) if files else "No changed files found."
    if name == "list_repository_tree":
        return list_repository_tree(
            working_dir,
            arguments.get("path", "."),
            max_depth=arguments.get("max_depth", 3),
            max_entries=arguments.get("max_entries", DEFAULT_TREE_ENTRIES),
        )
    if name == "search_repository":
        return search_repository(
            arguments.get("query", ""),
            working_dir,
            path=arguments.get("path", "."),
            file_glob=arguments.get("file_glob"),
            regex=arguments.get("regex", False),
            case_sensitive=arguments.get("case_sensitive", False),
            max_results=arguments.get("max_results", DEFAULT_SEARCH_RESULTS),
        )
    return f"Unknown tool: {name}"


REVIEWER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_uncommitted_changes",
            "description": "Get the repository-wide git diff for uncommitted changes. This is never limited to focus_files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "'all' (default) for staged and unstaged changes, 'staged', 'unstaged', or a git ref like 'HEAD~1'",
                        "default": "all",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": "Read bounded excerpts from up to 10 repository files. Each excerpt starts at line 1 and is capped; use read_file for targeted ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Repository-relative file paths to preview",
                    }
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a targeted, line-numbered range from one repository file. Results are capped at 200 lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the repository root",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to return (1-based, default 1)",
                        "default": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to return (inclusive); defaults to 120 lines after start_line (hard cap 200)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_changed_files",
            "description": "List all repository files with staged or unstaged changes. This is never limited to focus_files.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_repository_tree",
            "description": "List a bounded repository tree before choosing files to inspect. Narrow path or depth for large repositories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative directory to list (default '.')",
                        "default": ".",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth from path (default 3, maximum 6)",
                        "default": 3,
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum returned entries (default 200, maximum 500)",
                        "default": 200,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repository",
            "description": "Search repository text with ripgrep and return bounded matching lines. Use this to locate relevant symbols before reading precise ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text or regex to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Repository-relative file or directory to search (default '.')",
                        "default": ".",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Optional ripgrep glob such as '*.py'",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Interpret query as a regular expression (default false)",
                        "default": False,
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Use case-sensitive matching (default false)",
                        "default": False,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum matching lines (default 20, maximum 100)",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def run_agentic_review(
    working_dir: str,
    diff_target: str = "staged",
    context_files: list[str] | None = None,
    focus_files: list[str] | None = None,
    task_description: str = "",
    include_trace: bool | None = None,
    status_callback: ReviewStatusCallback | None = None,
) -> str:
    """
    Run an agentic review where GLM decides what information to gather.
    """
    trace_enabled = (
        _env_flag("REVIEW_MCP_INCLUDE_TRACE")
        if include_trace is None
        else include_trace
    )
    trace: list[str] = []

    repo_dir = Path(working_dir).resolve()
    trace.append(f"Workspace: {repo_dir}")
    trace.append(f"Diff target: {diff_target}")
    if not repo_dir.exists():
        return _append_review_trace(
            f"Error: The directory '{repo_dir}' does not exist.", trace, trace_enabled
        )
    if not repo_dir.is_dir():
        return _append_review_trace(
            f"Error: The path '{repo_dir}' is not a directory.", trace, trace_enabled
        )

    try:
        client = _make_client()
    except ValueError as exc:
        return _append_review_trace(f"Error: {exc}", trace, trace_enabled)

    context_files_to_read = context_files or []
    context_file_content = ""
    all_diff_files: list[str] = []
    loaded_context_files: list[str] = []

    logger.info("Loading context files...")
    _notify_status(status_callback, "Loading review context files.")
    for context_entry in context_files_to_read:
        resolved_context_entry = _resolve_user_path(context_entry, repo_dir)
        expanded_context_files = expand_context_entry(context_entry, repo_dir)
        if not expanded_context_files:
            logger.warning(
                "  ! No readable context files found for %s", resolved_context_entry
            )
            continue

        if resolved_context_entry.is_dir():
            context_file_content += (
                f"\n\n--- CONTEXT DIRECTORY: {resolved_context_entry} ---"
            )

        for resolved_context_file in expanded_context_files:
            try:
                logger.info("  ✓ Loaded %s", resolved_context_file)
                loaded_context_files.append(str(resolved_context_file))

                processed_content, diff_files = read_context_file_with_links(
                    resolved_context_file,
                    repo_dir,
                    diff_target,
                )
                all_diff_files.extend(diff_files)
                context_file_content += f"\n\n--- CONTEXT FILE: {resolved_context_file} ---\n{processed_content}"
            except Exception as exc:
                logger.error("  ✗ Error loading %s: %s", resolved_context_file, exc)

    trace.append(f"Context entries requested: {len(context_files_to_read)}")
    trace.append(f"Context files loaded: {len(loaded_context_files)}")

    if focus_files:
        files_to_diff = focus_files
        logger.info("Focusing on %s specified files", len(files_to_diff))
        _notify_status(
            status_callback,
            f"Preparing a scoped diff for {len(files_to_diff)} focus file(s).",
        )
    elif all_diff_files:
        files_to_diff = all_diff_files
        logger.info("Found %s files in context files", len(files_to_diff))
        _notify_status(
            status_callback,
            f"Preparing diffs for {len(files_to_diff)} file(s) referenced by context.",
        )
    else:
        files_to_diff = None
        logger.info("No specific files - reviewer will decide")
        _notify_status(
            status_callback,
            "No scoped diff found; the model will inspect changed files with reviewer tools.",
        )

    if files_to_diff:
        diff_content = get_git_diff(
            repo_dir,
            diff_target,
            scope_files=files_to_diff,
        )
        changed_files = files_to_diff
    else:
        diff_content = ""
        changed_files = []
    trace.append(f"Files selected for initial diff: {len(changed_files)}")
    if diff_content:
        trace.append(f"Initial diff size: {len(diff_content)} chars")
    else:
        trace.append("Initial diff size: 0 chars")

    system_prompt = """You are a Senior Code Reviewer with access to tools.

Your job is to review code changes. You have access to these tools:
- get_uncommitted_changes: Get repository-wide diffs, never limited to focus_files
- list_changed_files: List all staged and unstaged changed files
- list_repository_tree: Inspect a bounded repository tree
- search_repository: Locate relevant text with bounded ripgrep results
- read_file: Read a targeted line range from one repository file
- read_files: Preview bounded excerpts from several repository files

The user will provide you with context about what to review. Use your tools
to gather only the additional information you need. Inspect the tree or search before
reading unfamiliar files, request narrow line ranges, and do not request content that
is already present in the prompt or a previous tool result. Do not read a whole large
file in consecutive chunks: use search to locate the relevant functions, then read
only those ranges. Return the review once you have enough evidence.

REVIEW FOCUS:
1. Does the code match the stated intent (if provided)?
2. Are there logic errors, bugs, or security risks?
3. Any missed requirements?
4. Does it follow best practices?

Be concise but thorough. Ignore minor style issues."""

    if focus_files:
        system_prompt += (
            "\n\nA focus_files scope is active. Keep the review centered on those files. "
            "The initial diff is scoped, but repository discovery tools intentionally see "
            "the whole repository. Only inspect additional files when directly relevant."
        )

    max_context_chars = _get_positive_int_env(
        "MAX_REVIEW_CONTEXT_CHARS",
        DEFAULT_MAX_REVIEW_CONTEXT_CHARS,
        minimum=10000,
    )
    max_tool_result_chars = _get_positive_int_env(
        "MAX_REVIEW_TOOL_RESULT_CHARS",
        DEFAULT_MAX_TOOL_RESULT_CHARS,
        minimum=1000,
    )
    trace.append(f"Max review context: {max_context_chars} chars")
    trace.append(f"Max tool result: {max_tool_result_chars} chars")

    sections: list[str] = []
    if task_description:
        bounded_task, task_truncated = _truncate_content(
            task_description,
            10000,
            "TASK DESCRIPTION",
        )
        sections.append(f"## Task Description\n{bounded_task}")
        if task_truncated:
            trace.append("Task description truncated for initial context budget")
    if focus_files:
        sections.append(f"## Focus Files\n{chr(10).join(focus_files)}")
    if changed_files:
        sections.append(f"## Files to Review\n{chr(10).join(changed_files)}")

    fixed_size = len(system_prompt) + sum(len(section) for section in sections) + 5000
    variable_budget = max(1000, max_context_chars - fixed_size)
    context_budget = 0
    diff_budget = 0
    if context_file_content.strip() and diff_content.strip():
        context_budget = min(len(context_file_content), variable_budget // 3)
        diff_budget = min(len(diff_content), variable_budget - context_budget)
        unused = variable_budget - context_budget - diff_budget
        context_budget += min(unused, len(context_file_content) - context_budget)
        unused = variable_budget - context_budget - diff_budget
        diff_budget += min(unused, len(diff_content) - diff_budget)
    elif context_file_content.strip():
        context_budget = variable_budget
    elif diff_content.strip():
        diff_budget = variable_budget

    initial_supplied_contents: list[str] = []
    if context_budget:
        bounded_context, context_truncated = _truncate_content(
            context_file_content,
            context_budget,
            "INITIAL CONTEXT",
        )
        sections.append(f"## Context Files\n{bounded_context}")
        initial_supplied_contents.append(bounded_context)
        if context_truncated:
            trace.append("Initial context files truncated to fit review context budget")
    if diff_budget:
        bounded_diff, diff_truncated = _truncate_content(
            diff_content,
            diff_budget,
            "INITIAL DIFF",
        )
        sections.append(f"## Git Diff ({diff_target})\n```diff\n{bounded_diff}\n```")
        initial_supplied_contents.append(bounded_diff)
        if diff_truncated:
            trace.append("Initial diff truncated to fit review context budget")

    if sections:
        user_message = "Please review the following:\n\n" + "\n\n".join(sections)
        user_message += "\n\n---\nProvide a thorough code review. Use your tools if you need more information."
    else:
        user_message = "Please review the current code changes. Use list_changed_files and get_uncommitted_changes to see what's been modified."

    user_message_limit = max(500, max_context_chars - len(system_prompt))
    user_message, initial_prompt_truncated = _truncate_content(
        user_message,
        user_message_limit,
        "INITIAL PROMPT",
    )
    if initial_prompt_truncated:
        trace.append(
            "Assembled initial prompt hard-truncated to review context ceiling"
        )

    messages: list[dict | object] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    model_name = (
        os.getenv("AI_MODEL") or os.getenv("ZHIPU_MODEL") or DEFAULT_REVIEW_MODEL
    )
    max_iterations = _get_max_iterations()
    trace.append(f"Max review iterations: {max_iterations}")
    logger.info("Starting review...")
    logger.info("Found %s changed files", len(changed_files))
    logger.info("Loaded %s context file paths", len(loaded_context_files))
    _notify_status(
        status_callback,
        f"Starting model review with {len(changed_files)} changed file(s) and {len(loaded_context_files)} context file(s).",
    )

    empty_final_retry_sent = False
    seen_tool_calls: set[str] = set()
    seen_result_digests = {
        _content_digest(content)
        for content in [
            *initial_supplied_contents,
            context_file_content,
            diff_content,
        ]
        if content.strip()
    }
    force_final_response = False
    for iteration in range(max_iterations):
        logger.info("Iteration %s: Calling GLM...", iteration + 1)
        total_chars = _messages_size_chars(messages)
        logger.info("  Payload size: %s chars, %s messages", total_chars, len(messages))
        trace.append(
            f"Iteration {iteration + 1}: payload {total_chars} chars across {len(messages)} messages"
        )
        _notify_status(
            status_callback,
            f"Iteration {iteration + 1}/{max_iterations}: calling {model_name} with {len(messages)} message(s), about {total_chars} chars.",
        )

        backoff = 1
        response = None
        for retry in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=REVIEWER_TOOLS,
                    tool_choice="none" if force_final_response else "auto",
                    temperature=0.2,
                )
                break
            except Exception as exc:
                retryable = any(
                    code in str(exc) for code in ["500", "502", "503", "429"]
                )
                if retry < 2 and retryable:
                    logger.warning(
                        "  Retry %s due to %s. Waiting %ss...", retry + 1, exc, backoff
                    )
                    _notify_status(
                        status_callback,
                        f"Iteration {iteration + 1}/{max_iterations}: retryable model error; retrying in {backoff}s.",
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                error_msg = f"Error calling model '{model_name}' API at iteration {iteration + 1}: {exc}"
                logger.error(error_msg)
                _notify_status(status_callback, error_msg)
                if "502" in str(exc) or "500" in str(exc):
                    error_msg += "\n\nTIP: This often happens if the context is too large or the payload is complex. Try reducing the number of focus_files or context_files."
                return _append_review_trace(error_msg, trace, trace_enabled)

        if response is None:
            _notify_status(status_callback, "Model call did not return a response.")
            return _append_review_trace(
                "Error: Model call did not return a response.",
                trace,
                trace_enabled,
            )

        message = response.choices[0].message
        if not message.tool_calls:
            logger.info("Review complete!")
            content = _message_content_to_text(message.content)
            if content.strip():
                trace.append(f"Review complete after {iteration + 1} iteration(s)")
                _notify_status(
                    status_callback, "Review complete; returning the final response."
                )
                return _append_review_trace(content, trace, trace_enabled)

            if not empty_final_retry_sent and iteration + 1 < max_iterations:
                empty_final_retry_sent = True
                _notify_status(
                    status_callback,
                    "Model returned an empty final response; asking for the review text.",
                )
                trace.append(
                    f"Iteration {iteration + 1}: model returned an empty final response; requesting review text"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You returned an empty response. Provide the final code review now. "
                            "If there are no reviewable diffs, say that explicitly and summarize "
                            "what diffs and changed files were checked."
                        ),
                    }
                )
                continue

            trace.append(
                f"Review completed with empty content after {iteration + 1} iteration(s)"
            )
            return _append_review_trace("No review generated.", trace, trace_enabled)

        messages.append(message)
        logger.info("GLM requested %s tool(s)", len(message.tool_calls))
        trace.append(
            f"Iteration {iteration + 1}: model requested {len(message.tool_calls)} tool call(s)"
        )
        _notify_status(
            status_callback,
            f"Iteration {iteration + 1}/{max_iterations}: model requested {len(message.tool_calls)} reviewer tool(s).",
        )

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            try:
                func_args = json.loads(tool_call.function.arguments)
                if not isinstance(func_args, dict):
                    logger.warning(
                        "Invalid argument type for %s: %s",
                        func_name,
                        type(func_args).__name__,
                    )
                    func_args = {}
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid JSON arguments for %s: %r",
                    func_name,
                    tool_call.function.arguments,
                )
                func_args = {}

            logger.info("  → %s(%s)", func_name, func_args)
            _notify_status(status_callback, f"Running reviewer tool: {func_name}.")
            tool_key = _normalized_tool_call_key(func_name, func_args, repo_dir)
            if tool_key in seen_tool_calls:
                result = "This exact tool request was already completed. Use the previous result or request a narrower, different range."
                trace.append(f"Tool call deduplicated: {func_name}")
            elif force_final_response:
                result = "The review context budget is exhausted. Produce the final review from the information already gathered."
            else:
                seen_tool_calls.add(tool_key)
                try:
                    raw_result = _execute_tool(
                        func_name,
                        func_args,
                        working_dir=repo_dir,
                    )
                except Exception as exc:
                    logger.error("Reviewer tool %s failed: %s", func_name, exc)
                    raw_result = f"Error running reviewer tool '{func_name}': {exc}"
                    trace.append(f"Tool error contained: {func_name}")
                result_digest = _content_digest(raw_result)
                if raw_result.strip() and result_digest in seen_result_digests:
                    result = "This content was already included in the initial prompt or a previous tool result. Use the existing copy."
                    trace.append(f"Tool result deduplicated: {func_name}")
                else:
                    seen_result_digests.add(result_digest)
                    remaining_context = max_context_chars - _messages_size_chars(
                        messages
                    )
                    result_limit = min(
                        max_tool_result_chars, max(0, remaining_context - 200)
                    )
                    if result_limit < 200:
                        result = "The review context budget is exhausted. Produce the final review from the information already gathered."
                        force_final_response = True
                        trace.append(
                            "Review context budget exhausted; forcing final response"
                        )
                    else:
                        result, result_truncated = _truncate_content(
                            raw_result,
                            result_limit,
                            f"{func_name} RESULT",
                        )
                        if result_truncated:
                            trace.append(
                                f"Tool result truncated: {func_name} from {len(raw_result)} to {len(result)} chars"
                            )
                        if result_limit < max_tool_result_chars:
                            force_final_response = True
                            trace.append(
                                "Review context budget reached; forcing final response"
                            )
            trace.append(f"Tool call: {func_name} -> {len(result)} chars")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    _notify_status(
        status_callback, "Review stopped after reaching the maximum model iterations."
    )
    return _append_review_trace(
        "Error: Maximum iterations reached without completing review.",
        trace,
        trace_enabled,
    )
