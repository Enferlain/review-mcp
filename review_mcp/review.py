"""Agentic code-review orchestration for review-mcp."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

from review_mcp import repository
from review_mcp.config import (
    DEFAULT_MAX_TOOL_RESULT_CHARS,
    DEFAULT_REVIEW_MODEL,
    GLM_5_2_CONTEXT_WINDOW_TOKENS,
    GLM_5_2_MAX_OUTPUT_TOKENS,
    _env_flag,
    _get_glm_5_2_reasoning_effort,
    _get_max_iterations,
    _get_optional_positive_int_env,
    _get_positive_int_env,
    logger,
)
from review_mcp.context import expand_context_entry, read_context_file_with_links
from review_mcp.review_tools import (
    REVIEWER_TOOLS,
    _append_review_trace,
    _content_digest,
    _execute_tool,
    _message_content_to_text,
    _messages_size_chars,
    _normalized_tool_call_key,
    _truncate_content,
)
from review_mcp.transport import _create_streaming_chat_completion, _make_client


ReviewStatusCallback = Callable[[str], None]


_PSEUDO_TOOL_TAG_RE = re.compile(
    r"<\s*/?\s*(?:tool_calls?|function_call)\b|<\|\s*(?:tool_calls?|function_call)\s*\|>",
    re.IGNORECASE,
)
_PSEUDO_TOOL_INTENT_RE = re.compile(
    r"\b(?:let\s+me|i\s*(?:['’]ll|will|need\s+to)|"
    r"(?:first|next),?\s+i\s*(?:['’]ll|will|need\s+to)?)\s+"
    r"(?:call|read|inspect|search|check|look(?:\s+at)?|open|run)\b",
    re.IGNORECASE,
)
_PSEUDO_TOOL_RESOLUTION_RE = re.compile(
    r"\b(?:handled correctly|looks correct|is correct|no (?:issue|problem)|"
    r"completed|resolved)\b",
    re.IGNORECASE,
)


def _looks_like_unfinished_pseudo_tool_response(content: str) -> bool:
    """Detect model text that is clearly an unfinished tool request.

    Structured tool calls are handled separately by the review loop. This
    conservative check only catches provider-style markup or explicit
    process narration (for example, ``Let me read the file``); ordinary
    review prose is left untouched.
    """
    normalized = content.strip()
    if not normalized:
        return False
    if _PSEUDO_TOOL_TAG_RE.search(normalized):
        return True
    intent_matches = list(_PSEUDO_TOOL_INTENT_RE.finditer(normalized))
    if not intent_matches:
        return False
    if len(intent_matches) >= 3:
        return True
    if len(normalized) >= 500:
        return False
    return not bool(_PSEUDO_TOOL_RESOLUTION_RE.search(normalized))


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
        resolved_context_entry = repository._resolve_user_path(context_entry, repo_dir)
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
        diff_content = repository.get_git_diff(
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
is already present in the prompt or a previous tool result.

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

    # This is an optional operational safety valve only. It is deliberately
    # not derived from the model's token window: characters are not tokens.
    max_context_chars = _get_optional_positive_int_env(
        "MAX_REVIEW_CONTEXT_CHARS",
        minimum=10000,
    )
    max_tool_result_chars = _get_positive_int_env(
        "MAX_REVIEW_TOOL_RESULT_CHARS",
        DEFAULT_MAX_TOOL_RESULT_CHARS,
        minimum=1000,
    )
    if max_context_chars is None:
        trace.append(
            "Max review context: provider token window (no character safety override)"
        )
    else:
        trace.append(f"Max review context safety override: {max_context_chars} chars")
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

    context_budget: int | None = None
    diff_budget: int | None = None
    if max_context_chars is not None:
        fixed_size = (
            len(system_prompt) + sum(len(section) for section in sections) + 5000
        )
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
    if context_file_content.strip():
        if max_context_chars is None:
            # No default character cap: include all caller-supplied context.
            # The provider's token usage is the authoritative measurement.
            bounded_context = context_file_content
            context_truncated = False
        else:
            bounded_context, context_truncated = _truncate_content(
                context_file_content,
                context_budget or 0,
                "INITIAL CONTEXT",
            )
        sections.append(f"## Context Files\n{bounded_context}")
        initial_supplied_contents.append(bounded_context)
        if context_truncated:
            trace.append("Initial context files truncated to fit review context budget")
    if diff_content.strip():
        if max_context_chars is None:
            bounded_diff = diff_content
            diff_truncated = False
        else:
            bounded_diff, diff_truncated = _truncate_content(
                diff_content,
                diff_budget or 0,
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

    if max_context_chars is not None:
        user_message_limit = max(500, max_context_chars - len(system_prompt))
        user_message, initial_prompt_truncated = _truncate_content(
            user_message,
            user_message_limit,
            "INITIAL PROMPT",
        )
        if initial_prompt_truncated:
            trace.append(
                "Assembled initial prompt hard-truncated to explicit character safety override"
            )

    messages: list[dict | object] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    model_name = (
        os.getenv("AI_MODEL") or os.getenv("ZHIPU_MODEL") or DEFAULT_REVIEW_MODEL
    )
    model_context_window_tokens = (
        GLM_5_2_CONTEXT_WINDOW_TOKENS
        if model_name.strip().lower() == "glm-5.2"
        else None
    )
    glm_5_2_reasoning_effort = (
        _get_glm_5_2_reasoning_effort()
        if model_name.strip().lower() == "glm-5.2"
        else None
    )
    if glm_5_2_reasoning_effort:
        trace.append(f"Reasoning effort: {glm_5_2_reasoning_effort}")
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
    tool_schema_chars = len(json.dumps(REVIEWER_TOOLS, ensure_ascii=False))
    force_final_response = False
    for iteration in range(max_iterations):
        final_iteration = iteration + 1 == max_iterations
        logger.info("Iteration %s: Calling GLM...", iteration + 1)
        total_chars = _messages_size_chars(messages) + tool_schema_chars
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
                completion_kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.2,
                }
                if glm_5_2_reasoning_effort:
                    thinking_type = (
                        "disabled"
                        if glm_5_2_reasoning_effort == "none"
                        else "enabled"
                    )
                    completion_kwargs.update(
                        reasoning_effort=glm_5_2_reasoning_effort,
                        extra_body={
                            "thinking": {
                                "type": thinking_type,
                                "clear_thinking": False,
                            }
                        },
                    )
                if force_final_response or final_iteration:
                    completion_kwargs.update(
                        tools=REVIEWER_TOOLS,
                        tool_choice="none",
                    )
                else:
                    completion_kwargs.update(
                        tools=REVIEWER_TOOLS,
                        tool_choice="auto",
                    )
                response = _create_streaming_chat_completion(
                    client,
                    **completion_kwargs,
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
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if isinstance(prompt_tokens, int):
            usage_trace = f"Iteration {iteration + 1}: API usage {prompt_tokens} prompt tokens"
            if isinstance(completion_tokens, int):
                usage_trace += f", {completion_tokens} completion tokens"
            trace.append(usage_trace)
            if model_context_window_tokens is not None:
                remaining_input_tokens = (
                    model_context_window_tokens
                    - prompt_tokens
                    - GLM_5_2_MAX_OUTPUT_TOKENS
                )
                trace.append(
                    f"Iteration {iteration + 1}: {max(0, remaining_input_tokens)} input tokens remain after reserving the model output allowance"
                )
                if remaining_input_tokens <= 0 and not final_iteration:
                    force_final_response = True
                    trace.append(
                        "Model token budget reached; forcing final response on the next iteration"
                    )
        if not message.tool_calls:
            content = _message_content_to_text(message.content)
            if content.strip():
                if _looks_like_unfinished_pseudo_tool_response(content):
                    trace.append(
                        f"Iteration {iteration + 1}: model returned an unfinished pseudo-tool response"
                    )
                    if final_iteration:
                        error_msg = (
                            "Error: Model returned an unfinished tool request instead of a final code review. "
                            "Retry the review so the model can return a structured tool call or completed review text."
                        )
                        logger.error(error_msg)
                        _notify_status(status_callback, error_msg)
                        return _append_review_trace(error_msg, trace, trace_enabled)

                    # Keep the assistant turn in context and let the next
                    # iteration recover without modifying the review prompt.
                    messages.append(message)
                    _notify_status(
                        status_callback,
                        "Model returned an unfinished tool request; continuing the review.",
                    )
                    continue

                logger.info("Review complete!")
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
                result = "The model context budget has been reached. Produce the final review from the information already gathered."
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
                    if max_context_chars is None:
                        result_limit = max_tool_result_chars
                    else:
                        remaining_context = max_context_chars - _messages_size_chars(
                            messages
                        ) - tool_schema_chars
                        result_limit = min(
                            max_tool_result_chars, max(0, remaining_context - 200)
                        )
                    if result_limit < 200:
                        result = "The review context safety override is exhausted. Produce the final review from the information already gathered."
                        force_final_response = True
                        trace.append(
                            "Explicit character safety override exhausted; forcing final response"
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
                        if result_truncated and result_limit < max_tool_result_chars:
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
