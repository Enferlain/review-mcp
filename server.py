"""
MCP server entrypoint for agentic code review.
"""

from __future__ import annotations

import argparse
import os
import sys

import anyio
from mcp.server.fastmcp import FastMCP


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


def create_mcp(workspace_dir: str | None = None) -> FastMCP:
    """Create the MCP server with an optional default workspace."""
    default_workspace_dir = os.path.abspath(workspace_dir) if workspace_dir else None
    mcp = FastMCP(name="Code Review MCP")

    @mcp.tool()
    async def review_with_context(
        diff_target: str = "staged",
        context_files: list[str] | None = None,
        focus_files: list[str] | None = None,
        task_description: str = "",
        working_directory: str | None = None,
        include_trace: bool | None = None,
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

        return await anyio.to_thread.run_sync(
            lambda: reviewer.run_agentic_review(
                working_dir=effective_dir,
                diff_target=diff_target,
                context_files=context_files,
                focus_files=focus_files,
                task_description=task_description,
                include_trace=include_trace,
            )
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
