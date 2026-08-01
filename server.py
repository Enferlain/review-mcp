"""
MCP server entrypoint for agentic code review.
"""

import argparse
import os
import sys
import time
from typing import Awaitable, Callable

import anyio
from mcp.server.mcpserver import Context, MCPServer

DEFAULT_REVIEW_TOOL_TIMEOUT_SECONDS = 1800


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse local CLI options without interfering with the MCP host."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--workspace-dir",
        dest="workspace_dir",
        default=None,
        help="Optional default workspace directory (git repository root)",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def _get_tool_timeout_seconds() -> int:
    """Return a safe tool timeout that stays ahead of host-level deadlines."""
    raw_value = os.getenv("REVIEW_TOOL_TIMEOUT_SECONDS", str(DEFAULT_REVIEW_TOOL_TIMEOUT_SECONDS))
    try:
        value = int(raw_value)
    except ValueError:
        print(
            f"[ReviewMCP] Invalid REVIEW_TOOL_TIMEOUT_SECONDS={raw_value!r}; "
            f"defaulting to {DEFAULT_REVIEW_TOOL_TIMEOUT_SECONDS}",
            file=sys.stderr,
        )
        return DEFAULT_REVIEW_TOOL_TIMEOUT_SECONDS

    if value < 1:
        print(
            "[ReviewMCP] REVIEW_TOOL_TIMEOUT_SECONDS must be >= 1; "
            f"defaulting to {DEFAULT_REVIEW_TOOL_TIMEOUT_SECONDS}",
            file=sys.stderr,
        )
        return DEFAULT_REVIEW_TOOL_TIMEOUT_SECONDS
    return value


async def _send_status_safely(
    status_reporter: Callable[[str, float | None, float | None], Awaitable[None]],
    message: str,
    progress: float | None = None,
    total: float | None = None,
) -> None:
    """Best-effort MCP status reporting that never fails the review."""
    try:
        await status_reporter(message, progress, total)
    except Exception as exc:
        print(f"[ReviewMCP] Could not send status update: {exc}", file=sys.stderr)


async def _run_review_with_timeout(
    review_call: Callable[[], str],
    timeout_seconds: float,
    status_reporter: Callable[[str, float | None, float | None], Awaitable[None]] | None = None,
    status_interval_seconds: float = 30.0,
) -> str:
    """Run the blocking review call with status heartbeats and a user-facing timeout."""
    timeout_seconds = float(timeout_seconds)
    result: str | None = None
    finished = anyio.Event()

    async def run_review() -> None:
        nonlocal result
        try:
            result = await anyio.to_thread.run_sync(
                review_call,
                abandon_on_cancel=True,
            )
        finally:
            finished.set()

    async def report_heartbeat() -> None:
        if status_reporter is None:
            return

        started = time.monotonic()
        while True:
            with anyio.move_on_after(status_interval_seconds) as scope:
                await finished.wait()
            if not scope.cancel_called:
                return

            elapsed = time.monotonic() - started
            await _send_status_safely(
                status_reporter,
                f"Review still running after {int(elapsed)}s; waiting for the model or reviewer tools to return.",
                min(elapsed, timeout_seconds),
                timeout_seconds,
            )

    try:
        with anyio.fail_after(timeout_seconds):
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(run_review)
                task_group.start_soon(report_heartbeat)
    except TimeoutError:
        timeout_label = int(timeout_seconds) if timeout_seconds.is_integer() else timeout_seconds
        return (
            f"Error: review timed out after {timeout_label}s before the MCP host limit. "
            "Try narrowing focus_files, adding targeted context_files, or reducing "
            "MAX_REVIEW_ITERATIONS / REVIEW_TOOL_TIMEOUT_SECONDS."
        )

    return result or "No review generated."


def create_mcp(workspace_dir: str | None = None) -> MCPServer:
    """Create the MCP server with an optional default workspace."""
    default_workspace_dir = os.path.abspath(workspace_dir) if workspace_dir else None
    mcp = MCPServer(name="Code Review MCP")

    @mcp.tool()
    async def review_with_context(
        diff_target: str = "staged",
        context_files: list[str] | None = None,
        focus_files: list[str] | None = None,
        task_description: str = "",
        working_directory: str | None = None,
        include_trace: bool | None = None,
        ctx: Context = None,
    ) -> str:
        """
        Review code changes against project context using GLM.

        Args:
            diff_target: 'staged', 'unstaged', or a git ref like 'HEAD~1'
            context_files: Additional files or OpenSpec change folders to read
            focus_files: Specific files to focus the review on
            task_description: Optional task description for reviewer intent
            working_directory: Git repository root to review. Required unless the server was started with --workspace-dir
            include_trace: Include a compact diagnostic trace in the returned review

        Returns:
            The generated code review.
        """
        selected_workspace = working_directory or default_workspace_dir
        if not selected_workspace:
            return (
                "Error: No workspace directory was provided. "
                "Pass the current repository path as working_directory when calling review_with_context, "
                "or start review-mcp with --workspace-dir for a fixed default."
            )

        effective_dir = os.path.abspath(selected_workspace)
        print(f"[ReviewMCP] Effective workspace: {effective_dir}", file=sys.stderr)

        # Import lazily so MCP startup stays fast and reliable in editors.
        import reviewer

        timeout_seconds = float(_get_tool_timeout_seconds())
        started_at = time.monotonic()

        async def report_status(
            message: str,
            progress: float | None = None,
            total: float | None = None,
        ) -> None:
            print(f"[ReviewMCP] {message}", file=sys.stderr)
            if ctx is None:
                return

            progress_value = progress
            if progress_value is None:
                progress_value = min(time.monotonic() - started_at, timeout_seconds)
            await ctx.report_progress(
                progress_value,
                total=total or timeout_seconds,
                message=message,
            )

        def report_status_from_thread(message: str) -> None:
            anyio.from_thread.run(report_status, message, None, None)

        await report_status(
            f"Starting review in {effective_dir}.",
            0,
            timeout_seconds,
        )
        return await _run_review_with_timeout(
            lambda: reviewer.run_agentic_review(
                working_dir=effective_dir,
                diff_target=diff_target,
                context_files=context_files,
                focus_files=focus_files,
                task_description=task_description,
                include_trace=include_trace,
                status_callback=report_status_from_thread,
            ),
            timeout_seconds,
            report_status,
        )

    return mcp


def main() -> None:
    """Run the MCP server."""
    args = parse_args(sys.argv[1:])
    workspace_dir = os.path.abspath(args.workspace_dir) if args.workspace_dir else None
    if workspace_dir:
        print(f"[ReviewMCP] Using default workspace: {workspace_dir}", file=sys.stderr)
    else:
        print("[ReviewMCP] No default workspace configured", file=sys.stderr)
    create_mcp(workspace_dir).run()


if __name__ == "__main__":
    main()
