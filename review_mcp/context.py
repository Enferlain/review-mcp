"""Context-file expansion and linked diff helpers for review-mcp."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from .config import (
    CONTEXT_DIRECTORY_SUFFIXES,
    MAX_CONTEXT_FILES_PER_DIRECTORY,
    OPENSPEC_ROOT_FILE_ORDER,
)
from .repository import (
    _path_for_git,
    _resolve_user_path,
    _run_git_command,
)


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

