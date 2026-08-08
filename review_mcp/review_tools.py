"""Message, context-budget, and tool-dispatch helpers for agentic reviews."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from review_mcp import repository
from review_mcp.config import (
    DEFAULT_READ_FILE_LINES,
    DEFAULT_SEARCH_RESULTS,
    DEFAULT_TREE_ENTRIES,
)
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
    """Estimate request size using content, reasoning, and tool-call arguments."""
    if isinstance(message, dict):
        size = len(str(message.get("content", "") or ""))
        size += len(str(message.get("reasoning_content", "") or ""))
        tool_calls = message.get("tool_calls", []) or []
    else:
        size = len(_message_content_to_text(getattr(message, "content", "")))
        size += len(str(getattr(message, "reasoning_content", "") or ""))
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
            path = repository._repository_relative_path(
                repository._resolve_repository_path(path, working_dir), working_dir
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
        return repository.get_git_diff(working_dir, arguments.get("target", "all"))
    if name == "read_files":
        paths = arguments.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list):
            return "Error: read_files paths must be a list of repository file paths."
        excerpts = [
            repository.read_repository_file(path, working_dir) for path in paths[:10]
        ]
        return "\n\n".join(excerpts)
    if name == "read_file":
        return repository.read_repository_file(
            arguments.get("path", ""),
            working_dir,
            start_line=arguments.get("start_line", 1),
            end_line=arguments.get("end_line"),
        )
    if name == "list_changed_files":
        files = repository.get_changed_files(working_dir)
        return "\n".join(files) if files else "No changed files found."
    if name == "list_repository_tree":
        return repository.list_repository_tree(
            working_dir,
            arguments.get("path", "."),
            max_depth=arguments.get("max_depth", 3),
            max_entries=arguments.get("max_entries", DEFAULT_TREE_ENTRIES),
        )
    if name == "search_repository":
        return repository.search_repository(
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
            "description": "Read a targeted, line-numbered range from one repository file. Results are capped at 400 lines.",
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
                        "description": "Last line to return (inclusive); defaults to 200 lines after start_line",
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
