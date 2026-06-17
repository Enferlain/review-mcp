# Code Review MCP

AI-powered code review using Zhipu GLM. The server gathers git diffs and optional source context, then asks the model for a focused review.

## Installation

Install dependencies:

```bash
git clone https://github.com/Enferlain/antigravity-review-mcp.git
cd antigravity-review-mcp
cp .env.example .env
# Optional: edit .env and add your API key
uv sync
```

Then add it to your MCP client.

For local development, use `uv run` from your clone:

```json
{
  "mcpServers": {
    "review-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/antigravity-review-mcp",
        "run",
        "review-mcp"
      ]
    }
  }
}
```

For day-to-day use from Git, `uvx` avoids hardcoding a local install path:

```json
{
  "mcpServers": {
    "review-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/Enferlain/antigravity-review-mcp",
        "review-mcp"
      ]
    }
  }
}
```

The MCP caller should pass the repository being reviewed as `working_directory` on each `review_with_context` call. If your MCP client cannot pass a per-call workspace, you can still start the server with a fixed fallback:

```json
"args": [
  "--from",
  "git+https://github.com/Enferlain/antigravity-review-mcp",
  "review-mcp",
  "--workspace-dir",
  "/absolute/path/to/the-repo-you-want-reviewed"
]
```

## Configuration

Environment variables (in `.env`):

- `AI_API_KEY` (required): Your API key
- `ZHIPU_API_KEY` (optional): Backward-compatible fallback key name
- `ZHIPU_BASE_URL` (optional): Override API endpoint
- `AI_MODEL` / `ZHIPU_MODEL` (optional): Override the review model (default: `GLM-4.7`)
- `MAX_REVIEW_ITERATIONS` (optional): Max tool-calling iterations (default: 20, capped at 50)
- `REVIEW_MCP_INCLUDE_TRACE` (optional): Append diagnostic trace details to review responses (`true`/`false`)

## Usage

The MCP exposes **1 tool**: `review_with_context`

Parameters:

- `diff_target`: 'staged' (default), 'unstaged', or a git ref like 'HEAD~1'
- `context_files`: Additional files or OpenSpec change folders to include as context
- `focus_files`: Specific files to focus the review on
- `task_description`: Description of what you're trying to accomplish
- `working_directory`: Git repository root to review (required unless the server was started with `--workspace-dir`)
- `include_trace`: Include a compact diagnostic trace in the returned review (optional, defaults to `REVIEW_MCP_INCLUDE_TRACE`)

When called, it automatically:

1. Reads any files listed in `context_files`
2. Expands explicitly provided OpenSpec change folders into their context files
3. Resolves `render_diffs()` and `file:///` links inside those context files
4. Includes an initial scoped diff when `focus_files` or context-file `render_diffs()` links identify files
5. Lets GLM gather additional diffs or files as needed using its tools
6. Returns the final review

OpenSpec change folders are included only when the MCP caller passes the folder path in `context_files`, for example:

```json
{
  "context_files": [
    "/home/imi/Projects/sd-scripts/openspec/changes/resource-intelligence-system"
  ]
}
```

If MCP calls feel opaque, set `include_trace` to `true` for a single call or set `REVIEW_MCP_INCLUDE_TRACE=true` in the environment. The returned review will include a compact trace with the workspace, diff target, context-file count, payload sizes, model iterations, and tool calls.

Trace output is meant for debugging review behavior and can be disabled again once the setup is behaving as expected.

## Codex / VS Code Notes

This server now starts cleanly under MCP hosts because it avoids doing heavy work at import time. A few setup notes still matter:

1. Prefer passing `working_directory` per tool call so one MCP config can review any repo.
2. If your MCP client cannot inject the current repo automatically, set `--workspace-dir` in the config as a fixed fallback.
3. Prefer setting `AI_API_KEY` as a system/user environment variable instead of storing it in MCP config.
4. The tool-level `working_directory` argument still overrides the configured workspace when your agent provides it.

Example Windows fallback path:

```json
"args": [
  "--directory",
  "D:/Projects/antigravity-review-mcp",
  "run",
  "review-mcp",
  "--workspace-dir",
  "D:/Projects/myrepo"
]
```

Example prompt to your AI assistant:

> "Review my staged changes"

## Security Note

The reviewer agent can read any file accessible from the working directory. This is by design for comprehensive reviews, but be aware of this when using in sensitive environments.
