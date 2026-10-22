# Code Review MCP

AI-powered code review using Zhipu GLM. The server gathers git diffs and optional source context, then asks the model for a focused review.

## Installation

Install dependencies:

```bash
git clone https://github.com/Enferlain/review-mcp.git
cd review-mcp
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
        "/absolute/path/to/review-mcp",
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
        "git+https://github.com/Enferlain/review-mcp",
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
  "git+https://github.com/Enferlain/review-mcp",
  "review-mcp",
  "--workspace-dir",
  "/absolute/path/to/the-repo-you-want-reviewed"
]
```

## Configuration

Environment variables (in `.env`):

- `AI_API_KEY` (required): Your API key
- `ZHIPU_API_KEY` (optional): Backward-compatible fallback key name
- `GLM_API_KEY` (optional): Alternate fallback key name
- `ZHIPU_BASE_URL` (optional): Override API endpoint
- `AI_MODEL` / `ZHIPU_MODEL` (optional): Override the review model (default: `glm-5.3-flash`); can also be set per call via the `model` parameter — use the `list_models` tool to see what the endpoint offers
- `AI_REASONING_EFFORT` / `ZHIPU_REASONING_EFFORT` (optional): Reasoning effort (default: `max`), sent for GLM models with reasoning controls (the `glm-4.5`+ family); also settable per call via the `reasoning_effort` parameter, where `none` disables thinking. Note: these request parameters are now sent for the whole `glm-4.5`+ family, not just `glm-5.2`.
- `AI_API_TIMEOUT_SECONDS` (optional): Timeout for each model API request (default: 900)
- `MAX_REVIEW_CONTEXT_CHARS` (optional): Explicit character-based safety ceiling for the accumulated request. Unset by default; character counts are not used as a proxy for model tokens. The completion API's `usage.prompt_tokens` is the authoritative context measurement, and the provider enforces its own model window.
- `MAX_REVIEW_TOOL_RESULT_CHARS` (optional): Per-tool result ceiling before truncation (default: 20000)
- `MAX_REVIEW_MODEL_CALLS` (optional): Review-turn budget (default: 64; invalid values or values below 2 fall back to 64), including one reserved tool-free final synthesis call. Transient failures can cause up to three HTTP attempts per turn. This bounds long tool loops independently of the selected model's context window. Provider context-limit errors still end the review with a failure; this budget does not estimate tokens.
- `REVIEW_TOOL_TIMEOUT_SECONDS` (optional): End the MCP tool call before the host-level timeout (default: 1800)
- `REVIEW_MCP_INCLUDE_TRACE` (optional): Append diagnostic trace details to returned and saved reviews (`true`/`false`, default: false).

## Usage

The MCP exposes **2 tools**: `review_with_context` and `list_models`

### `review_with_context`

Parameters:

- `diff_target`: 'staged' (default), 'unstaged', or a git ref like 'HEAD~1'
- `context_files`: Additional files or OpenSpec change folders to include as context
- `focus_files`: Specific files to focus the review on
- `task_description`: Description of what you're trying to accomplish
- `working_directory`: Git repository root to review (required unless the server was started with `--workspace-dir`)
- `include_trace`: Append a diagnostic trace to the returned and saved review when true (optional; defaults to `REVIEW_MCP_INCLUDE_TRACE`, then false). An explicit false disables tracing even when the environment enables it.
- `model`: Model id for this review, e.g. `glm-5.3` (optional; defaults to `AI_MODEL`/`ZHIPU_MODEL`, then `glm-5.3-flash`)
- `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` (optional; defaults to `AI_REASONING_EFFORT`/`ZHIPU_REASONING_EFFORT`, then `max`; only sent for reasoning-capable GLM models, `none` disables thinking; invalid values return an error listing the options)

Progress pairs ongoing and finished work: `Reading …` / `Read …`, `Searching …` / `Searched …`, and `Inspecting …` / `Inspected …`. Searches include their result counts. Each review turn reports the messages sent, then the provider-reported input and output token counts when available. These are per-turn counts; the input includes conversation history. Retries, shortened results, and finalization are also reported.

One review is a tool-use conversation. The model requests repository tools; this server executes them and sends their results back so the model can decide what to inspect next or produce its findings. Each exchange is a **review turn**, not a separate review. A single completion request cannot execute local repository tools by itself. Progress reports observable actions, not the model's private reasoning.

When called, it automatically:

1. Reads any files listed in `context_files`
2. Expands explicitly provided OpenSpec change folders into their context files
3. Resolves `render_diffs()` and `file:///` links inside those context files
4. Includes an initial scoped diff when `focus_files` or context-file `render_diffs()` links identify files
5. Lets GLM inspect a bounded repository tree, search with ripgrep, and read targeted line ranges
6. Lets GLM request repository-wide staged and unstaged diffs even when `focus_files` is active
7. Deduplicates repeated requests/results, records provider-reported token usage, and enforces the optional character safety ceiling plus per-tool result limits
8. Returns the final review

OpenSpec change folders are included only when the MCP caller passes the folder path in `context_files`, for example:

```json
{
  "context_files": [
    "/home/imi/Projects/sd-scripts/openspec/changes/resource-intelligence-system"
  ]
}
```

CLI and MCP responses contain the review followed by `Review artifacts: <directory>`. Diagnostic traces are appended to the review only when enabled with `include_trace=true`, `--trace`, or `REVIEW_MCP_INCLUDE_TRACE=true`. Tracing is off by default. The saved `review.md` includes the trace when enabled; no separate trace file is created.

Trace output is meant for debugging review behavior and can be disabled again once the setup is behaving as expected.

### `list_models`

Lists the model ids currently available from the configured provider, queried live from the endpoint's models listing (using the configured API key and `ZHIPU_BASE_URL`) rather than a hardcoded snapshot. Returns one model id per line with the configured default model marked, e.g.:

```text
glm-4.5
glm-4.6
glm-5.2
glm-5.3  <- current default
```

If the configured default model is not offered by the endpoint, a trailing `Configured default: <model> (not offered by this endpoint)` line notes it instead. The listing uses a short request timeout (15s) and returns an `Error: ...` message when no API key is configured or the provider cannot be reached.

## CLI mode (streaming progress)

Some MCP clients (Codex CLI at the time of writing) neither request MCP progress notifications nor render them, so a long `review_with_context` call shows no sign of life until it finishes. The same review can instead run as a subprocess whose output streams live over the host's exec channel:

```bash
uvx --from git+https://github.com/Enferlain/review-mcp.git@main review-mcp run <workspace> \
  [--diff-target staged] [--context PATH ...] [--focus PATH ...] [--task TEXT] \
  [--model ID] [--reasoning-effort max] [--trace] [--timeout 1800] \
  [--heartbeat 120] [--progress plain|ndjson|none]
```

- The review text is printed to **stdout** (the invoking agent captures it as command output); exit code 0 on success, 1 on a failed/timed-out review, 130 on Ctrl-C.
- Progress goes to **stderr** as concise `[  12.3s] action` lines by default. `--progress ndjson` emits JSON records for automation (`schema_version`, `type` = `status` | `heartbeat` | `result`, `ts`, `elapsed`, `message`; result records add `ok` and `chars`). Consumers should ignore unknown keys and types. Heartbeats default to 120 seconds of silence; `--heartbeat 0` disables them. `--progress none` silences stderr.
  `chars` counts the saved review text, excluding the artifacts path footer and the newline added when printing it.
- `review-mcp models` prints the same live model listing as the `list_models` tool.

Every review started through the CLI or MCP tool automatically keeps a private
`review-mcp-run-*` directory in the system temporary directory. Its path is
reported in startup progress (or supplied as `artifacts_dir` and `result_file`
in CLI NDJSON records). The directory contains:

- `review.md`: the completed review or error response, atomically saved before
  stdout is written, so losing the terminal connection cannot lose a saved result.
  The artifacts path footer is added to the returned response, not to this file.
- `progress.ndjson`: progress records written as they happen, even when terminal
  progress is disabled with `--progress none`.
- `status.json`: request options, process ID, outcome, and whether a result exists.

Files are retained after exit, Ctrl-C, or timeout. A hard kill can leave the last
status as `running` or a partial last progress record; an unfinished review has
no final result or resume support. If an MCP request is canceled while its worker
continues, a result that finishes later is still saved. Temporary files may be removed by
the OS. With CLI `--progress none`, no path is printed; locate runs under the system
temporary directory by their `review-mcp-run-*` prefix and request metadata.

Review and provider diagnostics use the selected progress format too. `--progress none` suppresses them, including on interruption. Timeout values must be finite and positive; heartbeat values must be finite and nonnegative (0 disables heartbeats).

Example progress (one review, with repository tools between analysis turns):

```text
[    0.0s] Preparing review of HEAD^.
[    0.1s] Review turn 1: sending 2 messages to glm-5.3 (effort: max).
[    4.0s] Review turn 1: received response; 1,234 input tokens, 120 output tokens.
[    4.0s] Reading server.py, lines 120–180
[    4.1s] Read server.py, lines 120–180
[    4.1s] Searching review_mcp for 'run_agentic_review'
[    4.2s] Searched review_mcp for 'run_agentic_review': 3 matching line(s).
[    4.3s] Review turn 2: sending 5 messages to glm-5.3.
[   11.0s] review finished
```

Running the bare `review-mcp` command with no arguments still starts the MCP server, so existing host registrations are unaffected.

## Codex / VS Code Notes

This server now starts cleanly under MCP hosts because it avoids doing heavy work at import time. A few setup notes still matter:

1. Prefer passing `working_directory` per tool call so one MCP config can review any repo.
2. If your MCP client cannot inject the current repo automatically, set `--workspace-dir` in the config as a fixed fallback.
3. Prefer setting `AI_API_KEY` (or the `GLM_API_KEY` alias) as a system/user environment variable instead of storing it in MCP config.
4. The tool-level `working_directory` argument still overrides the configured workspace when your agent provides it.

Example Windows fallback path:

```json
"args": [
  "--directory",
  "D:/Projects/review-mcp",
  "run",
  "review-mcp",
  "--workspace-dir",
  "D:/Projects/myrepo"
]
```

Example prompt to your AI assistant:

> "Review my staged changes"

## Security Note

Model-requested tree, search, and file-read tools are confined to the selected repository root, including protection against `..` and symlink escapes. Explicit caller-provided `context_files` may still reference readable files outside the repository, so use those deliberately in sensitive environments.
