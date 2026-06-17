"""
Code review logic using Zhipu GLM via an OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
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


def _looks_like_openspec_change_dir(path: Path) -> bool:
    """Detect an explicitly provided OpenSpec change directory."""
    if not path.is_dir():
        return False

    has_openspec_layout = (path / "proposal.md").exists() or (path / "tasks.md").exists()
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
        and not any(part.startswith(".") for part in candidate.relative_to(resolved).parts)
        and candidate.suffix.lower() in CONTEXT_DIRECTORY_SUFFIXES
    ]
    return sorted(files, key=lambda candidate: _context_path_sort_key(candidate, resolved))[
        :MAX_CONTEXT_FILES_PER_DIRECTORY
    ]


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
        timeout=120.0,
    )


def get_git_diff(working_dir: str | Path, target: str = "staged") -> str:
    """Get git diff output for the requested target."""
    try:
        if target == "staged":
            args = ["diff", "--staged"]
        elif target == "unstaged":
            args = ["diff"]
        else:
            args = ["diff", target]

        if EXCLUDE_PATTERNS:
            args.append("--")
            args.extend(f":!{pattern}" for pattern in EXCLUDE_PATTERNS)

        return _run_git_command(working_dir, args).stdout
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        return f"Error running git diff: {stderr or exc}"
    except FileNotFoundError:
        return "Error: git is not installed or not in PATH."


def get_changed_files(working_dir: str | Path) -> list[str]:
    """Get the list of changed files (staged + unstaged)."""
    try:
        staged = _run_git_command(working_dir, ["diff", "--staged", "--name-only"])
        unstaged = _run_git_command(working_dir, ["diff", "--name-only"])
        files = set(staged.stdout.strip().splitlines() + unstaged.stdout.strip().splitlines())
        return sorted(file for file in files if file)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def read_context_files(filepaths: list[str] | str | None, working_dir: str | Path) -> str:
    """Read multiple context files and format them for the prompt."""
    if not filepaths:
        return ""
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    context_chunks: list[str] = []
    for filepath in filepaths:
        resolved_paths = expand_context_entry(filepath, working_dir)
        if not resolved_paths:
            resolved = _resolve_user_path(filepath, working_dir)
            context_chunks.append(f"\n\n(Note: Context file '{resolved}' not found or not readable)\n")
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
                context_chunks.append(f"\n\n(Note: Context file '{resolved}' not found)\n")
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
        return get_git_diff(working_dir, arguments.get("target", "staged"))
    if name == "read_files":
        return read_context_files(arguments.get("paths", []), working_dir)
    if name == "read_file":
        return read_context_files([arguments.get("path", "")], working_dir)
    if name == "list_changed_files":
        files = get_changed_files(working_dir)
        return "\n".join(files) if files else "No changed files found."
    return f"Unknown tool: {name}"


REVIEWER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_uncommitted_changes",
            "description": "Get git diff output for uncommitted changes. Use this to see what code has been modified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "'staged' for staged changes, 'unstaged' for working tree changes, or a git ref like 'HEAD~1'",
                        "default": "staged",
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
            "description": "Read the contents of one or more files. OpenSpec change directories are also accepted and expanded into context files. Use this to read source files, documentation, or any relevant context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths or OpenSpec change directories to read (absolute or relative to cwd)",
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
            "description": "Read the contents of one file. Prefer read_files when reading more than one file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read (absolute or relative to cwd)",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_changed_files",
            "description": "List all files with uncommitted changes (staged + unstaged). Use this to know which files have been modified.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
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
) -> str:
    """
    Run an agentic review where GLM decides what information to gather.
    """
    trace_enabled = _env_flag("REVIEW_MCP_INCLUDE_TRACE") if include_trace is None else include_trace
    trace: list[str] = []

    repo_dir = Path(working_dir).resolve()
    trace.append(f"Workspace: {repo_dir}")
    trace.append(f"Diff target: {diff_target}")
    if not repo_dir.exists():
        return _append_review_trace(f"Error: The directory '{repo_dir}' does not exist.", trace, trace_enabled)
    if not repo_dir.is_dir():
        return _append_review_trace(f"Error: The path '{repo_dir}' is not a directory.", trace, trace_enabled)

    try:
        client = _make_client()
    except ValueError as exc:
        return _append_review_trace(f"Error: {exc}", trace, trace_enabled)

    context_files_to_read = context_files or []
    context_file_content = ""
    all_diff_files: list[str] = []
    loaded_context_files: list[str] = []

    logger.info("Loading context files...")
    for context_entry in context_files_to_read:
        resolved_context_entry = _resolve_user_path(context_entry, repo_dir)
        expanded_context_files = expand_context_entry(context_entry, repo_dir)
        if not expanded_context_files:
            logger.warning("  ! No readable context files found for %s", resolved_context_entry)
            continue

        if resolved_context_entry.is_dir():
            context_file_content += f"\n\n--- CONTEXT DIRECTORY: {resolved_context_entry} ---"

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
                context_file_content += (
                    f"\n\n--- CONTEXT FILE: {resolved_context_file} ---\n{processed_content}"
                )
            except Exception as exc:
                logger.error("  ✗ Error loading %s: %s", resolved_context_file, exc)

    trace.append(f"Context entries requested: {len(context_files_to_read)}")
    trace.append(f"Context files loaded: {len(loaded_context_files)}")

    if focus_files:
        files_to_diff = focus_files
        logger.info("Focusing on %s specified files", len(files_to_diff))
    elif all_diff_files:
        files_to_diff = all_diff_files
        logger.info("Found %s files in context files", len(files_to_diff))
    else:
        files_to_diff = None
        logger.info("No specific files - reviewer will decide")

    if files_to_diff:
        diff_content = get_scoped_diff(files_to_diff, repo_dir, diff_target)
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
- get_uncommitted_changes: Get git diffs (staged, unstaged, or vs specific refs)
- read_files: Read multiple source files at once (efficient - use this to batch file reads)
- list_changed_files: See which files have been modified

The user will provide you with context about what to review. Use your tools
to gather any additional information you need. Be efficient - batch file reads together.

REVIEW FOCUS:
1. Does the code match the stated intent (if provided)?
2. Are there logic errors, bugs, or security risks?
3. Any missed requirements?
4. Does it follow best practices?

Be concise but thorough. Ignore minor style issues."""

    sections: list[str] = []
    if task_description:
        sections.append(f"## Task Description\n{task_description}")
    if context_file_content.strip():
        sections.append(f"## Context Files\n{context_file_content}")
    if changed_files:
        sections.append(f"## Files to Review\n{chr(10).join(changed_files)}")
    if diff_content.strip():
        sections.append(f"## Git Diff ({diff_target})\n```diff\n{diff_content}\n```")

    if sections:
        user_message = "Please review the following:\n\n" + "\n\n".join(sections)
        user_message += "\n\n---\nProvide a thorough code review. Use your tools if you need more information."
    else:
        user_message = "Please review the current code changes. Use list_changed_files and get_uncommitted_changes to see what's been modified."

    messages: list[dict | object] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    max_iterations = _get_max_iterations()
    trace.append(f"Max review iterations: {max_iterations}")
    logger.info("Starting review...")
    logger.info("Found %s changed files", len(changed_files))
    logger.info("Loaded %s context file paths", len(loaded_context_files))

    empty_final_retry_sent = False
    for iteration in range(max_iterations):
        logger.info("Iteration %s: Calling GLM...", iteration + 1)
        total_chars = 0
        for message in messages:
            if isinstance(message, dict):
                total_chars += len(message.get("content", "") or "")
            else:
                total_chars += len(getattr(message, "content", "") or "")
        logger.info("  Payload size: %s chars, %s messages", total_chars, len(messages))
        trace.append(
            f"Iteration {iteration + 1}: payload {total_chars} chars across {len(messages)} messages"
        )

        backoff = 1
        response = None
        for retry in range(3):
            try:
                response = client.chat.completions.create(
                    model=os.getenv("AI_MODEL") or os.getenv("ZHIPU_MODEL") or "GLM-4.7",
                    messages=messages,
                    tools=REVIEWER_TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                )
                break
            except Exception as exc:
                retryable = any(code in str(exc) for code in ["500", "502", "503", "429"])
                if retry < 2 and retryable:
                    logger.warning("  Retry %s due to %s. Waiting %ss...", retry + 1, exc, backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                error_msg = f"Error calling GLM-4.7 API at iteration {iteration + 1}: {exc}"
                logger.error(error_msg)
                if "502" in str(exc) or "500" in str(exc):
                    error_msg += "\n\nTIP: This often happens if the context is too large or the payload is complex. Try reducing the number of focus_files or context_files."
                return _append_review_trace(error_msg, trace, trace_enabled)

        if response is None:
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
                return _append_review_trace(content, trace, trace_enabled)

            if not empty_final_retry_sent and iteration + 1 < max_iterations:
                empty_final_retry_sent = True
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

            trace.append(f"Review completed with empty content after {iteration + 1} iteration(s)")
            return _append_review_trace("No review generated.", trace, trace_enabled)

        messages.append(message)
        logger.info("GLM requested %s tool(s)", len(message.tool_calls))
        trace.append(f"Iteration {iteration + 1}: model requested {len(message.tool_calls)} tool call(s)")

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
            result = _execute_tool(func_name, func_args, working_dir=repo_dir)
            trace.append(f"Tool call: {func_name} -> {len(result)} chars")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return _append_review_trace(
        "Error: Maximum iterations reached without completing review.",
        trace,
        trace_enabled,
    )
