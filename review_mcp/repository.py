"""Repository, path, search, and git helpers for review-mcp."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from .config import (
    DEFAULT_READ_FILE_LINES,
    DEFAULT_SEARCH_RESULTS,
    DEFAULT_TREE_ENTRIES,
    EXCLUDE_PATTERNS,
    MAX_READ_FILE_LINES,
    MAX_READ_LINE_CHARS,
    MAX_SEARCH_RESULTS,
    MAX_TREE_ENTRIES,
    REPOSITORY_SEARCH_TIMEOUT_SECONDS,
    _bounded_int,
)


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

