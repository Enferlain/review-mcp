"""One-shot CLI mode: run a review as a subprocess with streaming progress.

Usage:
    review-mcp run [workspace] [--diff-target ...] [--context ...] [--focus ...]
                   [--task ...] [--model ...] [--reasoning-effort ...]
                   [--trace] [--timeout N] [--heartbeat N]
                   [--progress ndjson|plain|none]
    review-mcp models

The review text goes to stdout so the invoking agent captures it as command
output. Concise progress actions go to stderr; ``--progress ndjson`` provides
one JSON record per line for automated consumers.

NDJSON record contract (version 1): every line carries ``schema_version``, ``type``
(``status`` | ``heartbeat`` | ``result``), ``ts`` (unix seconds), ``elapsed``
(seconds since start), and ``message``; ``result`` records additionally carry
``ok`` (bool) and ``chars`` (review text length, excluding the trailing newline). Consumers must ignore
unknown keys and record types.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import threading
import time
from contextlib import contextmanager

PROGRESS_SCHEMA_VERSION = 1


def _seconds(value: str, *, allow_zero: bool = False) -> float:
    parsed = float(value)
    if (
        not math.isfinite(parsed)
        or parsed > threading.TIMEOUT_MAX
        or (parsed < 0 if allow_zero else parsed <= 0)
    ):
        bound = "nonnegative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(
            f"must be finite, {bound}, and <= {threading.TIMEOUT_MAX:g}"
        )
    return parsed


@contextmanager
def _route_review_logs(emit, worker, finished):
    """Route application and transport diagnostics through the progress channel."""

    class ProgressHandler(logging.Handler):
        def emit(self, record):
            emit(record.getMessage())

    handler = ProgressHandler()
    saved = []
    for name in ("ReviewMCP", "openai", "httpx", "httpcore"):
        logger = logging.getLogger(name)
        saved.append((logger, logger.handlers[:], logger.level, logger.propagate))
        logger.handlers = [handler]
        logger.propagate = False
        # Keep the transport libraries quiet except for warnings/errors.
        logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        # A timed-out/interrupted daemon may still log after this function
        # returns. Mute only that worker on the restored handlers until its
        # review function exits; other threads retain their normal logging.
        if not finished.is_set():

            class LateWorkerFilter(logging.Filter):
                def filter(self, record):
                    return record.thread != worker.ident or finished.is_set()

            late_filter = LateWorkerFilter()
            restored_handlers = set(logging.getLogger().handlers)
            for _, handlers, _, _ in saved:
                restored_handlers.update(handlers)
            for restored in restored_handlers:
                restored.addFilter(late_filter)

            def remove_late_filter():
                finished.wait()
                for restored in restored_handlers:
                    restored.removeFilter(late_filter)

            threading.Thread(
                target=remove_late_filter, daemon=True, name="review-log-cleanup"
            ).start()
        for logger, handlers, level, propagate in saved:
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate


def _emit_record(
    stream_lock: threading.Lock,
    kind: str,
    message: str,
    started: float,
    *,
    ok: bool | None = None,
    chars: int | None = None,
    extra: dict | None = None,
    record_sink=None,
    style: str = "ndjson",
) -> None:
    record: dict = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "type": kind,
        "ts": time.time(),
        "elapsed": round(time.monotonic() - started, 3),
        "message": message,
    }
    if ok is not None:
        record["ok"] = ok
    if chars is not None:
        record["chars"] = chars
    if extra:
        record.update(extra)
    with stream_lock:
        if record_sink is not None:
            record_sink(record)
        if style == "plain":
            message = " ".join(message.split())
            sys.stderr.write(f"[{record['elapsed']:7.1f}s] {message}\n")
        elif style == "ndjson":
            sys.stderr.write(json.dumps(record, ensure_ascii=False) + "\n")
        if style != "none":
            sys.stderr.flush()


def _make_status_emitter(
    style: str,
    started: float,
    stream_lock: threading.Lock,
    record_sink=None,
) -> tuple:
    """Build a run_agentic_review status callback plus heartbeat support."""

    class _HeartbeatState:
        def __init__(self) -> None:
            self.last_event = time.monotonic()
            self.last_message = "starting"
            self.stopped = threading.Event()
            self.lock = threading.Lock()

    state = _HeartbeatState()

    def emit(message: str) -> None:
        if state.stopped.is_set():
            return
        _emit_record(
            stream_lock,
            "status",
            message,
            started,
            record_sink=record_sink,
            style=style,
        )
        with state.lock:
            state.last_event = time.monotonic()
            state.last_message = message

    def heartbeat_loop(interval: float) -> None:
        while not state.stopped.wait(interval):
            with state.lock:
                idle = time.monotonic() - state.last_event
                message = state.last_message
            if idle < interval:
                continue
            _emit_record(
                stream_lock,
                "heartbeat",
                f"still running: {message}",
                started,
                extra={"idle": round(idle, 3)},
                record_sink=record_sink,
                style=style,
            )

    return emit, heartbeat_loop, state


def format_model_list(model_ids: list[str], default_model: str) -> str:
    """Render provider model ids, marking the configured default.

    Variant suffixes like ``[1m]`` select a context size on some endpoints but
    are not part of the listed id, so defaults match on the base id.
    """

    def _base_model_id(model_id: str) -> str:
        return re.sub(r"\[[^\]]*\]$", "", model_id.strip().lower())

    default_base = _base_model_id(default_model)
    lines = [
        f"{model_id}  <- current default"
        if _base_model_id(model_id) == default_base
        else model_id
        for model_id in model_ids
    ]
    if not any("<-" in line for line in lines):
        lines.append(
            f"Configured default: {default_model} (not offered by this endpoint)"
        )
    return "\n".join(lines)


def models_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="review-mcp models")
    parser.parse_args(argv)

    from review_mcp.config import _get_review_model
    from review_mcp.transport import _list_available_models

    try:
        model_ids = _list_available_models()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: Could not list models from the provider: {exc}", file=sys.stderr)
        return 1

    if not model_ids:
        print("Error: The provider returned no models.", file=sys.stderr)
        return 1
    print(format_model_list(model_ids, _get_review_model()))
    return 0


def run_main(argv: list[str] | None = None) -> int:
    from server import _get_tool_timeout_seconds

    parser = argparse.ArgumentParser(
        prog="review-mcp run",
        description="Run one code review as a subprocess with streaming progress on stderr.",
    )
    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Git repository to review (default: current directory)",
    )
    parser.add_argument(
        "--diff-target",
        default="staged",
        help="'staged', 'unstaged', or a git ref like 'HEAD~1'",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=None,
        dest="context_files",
        help="Additional context file or OpenSpec folder (repeatable)",
    )
    parser.add_argument(
        "--focus",
        action="append",
        default=None,
        dest="focus_files",
        help="File to focus the review on (repeatable)",
    )
    parser.add_argument(
        "--task",
        default="",
        dest="task_description",
        help="Task description for reviewer intent",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id for this review (default: AI_MODEL env / glm-5.3-flash)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort for reasoning-capable GLM models (default: high)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=None,
        help="Include the diagnostic trace in the review",
    )
    parser.add_argument(
        "--timeout",
        type=_seconds,
        default=None,
        help="Give up after this many seconds (default: REVIEW_TOOL_TIMEOUT_SECONDS or 1800)",
    )
    parser.add_argument(
        "--heartbeat",
        type=lambda value: _seconds(value, allow_zero=True),
        default=120.0,
        help="Emit a heartbeat after this many idle seconds; 0 disables (default: 120)",
    )
    parser.add_argument(
        "--progress",
        choices=["ndjson", "plain", "none"],
        default="plain",
        help="Progress style on stderr (default: plain)",
    )
    args = parser.parse_args(argv)

    from review_mcp.review import run_agentic_review
    from review_mcp.artifacts import ReviewArtifacts

    started = time.monotonic()
    stream_lock = threading.Lock()
    try:
        artifacts = ReviewArtifacts(vars(args).copy())
    except OSError as exc:
        error = f"Error: Could not create saved review files: {exc}"
        _emit_record(
            stream_lock,
            "result",
            error,
            started,
            ok=False,
            chars=len(error),
            style=args.progress,
        )
        print(error)
        return 1

    persistence_errors: list[OSError] = []

    def save_record(record: dict) -> None:
        try:
            artifacts.append_progress(record)
        except OSError as exc:
            persistence_errors.append(exc)

    emit, heartbeat_loop, heartbeat_state = _make_status_emitter(
        args.progress,
        started,
        stream_lock,
        record_sink=save_record,
    )
    _emit_record(
        stream_lock,
        "status",
        f"Saved review files: {artifacts.directory}",
        started,
        extra={
            "artifacts_dir": str(artifacts.directory),
            "result_file": str(artifacts.result_path),
        },
        record_sink=save_record,
        style=args.progress,
    )
    with heartbeat_state.lock:
        heartbeat_state.last_message = f"Saved review files: {artifacts.directory}"
        heartbeat_state.last_event = time.monotonic()
    timeout_seconds = (
        args.timeout
        if args.timeout is not None
        else _get_tool_timeout_seconds(warn=emit)
    )

    def finish(message: str, *, ok: bool, chars: int) -> bool:
        interrupted = message == "interrupted"
        timed_out = message == "review timed out"
        if persistence_errors:
            ok = False
            message += f"; could not save all review files: {persistence_errors[0]}"
        state = (
            "interrupted"
            if interrupted
            else ("timed_out" if timed_out else ("completed" if ok else "failed"))
        )
        try:
            artifacts.finish(
                state,
                exit_code=130 if interrupted else (0 if ok else 1),
                message=message,
            )
        except OSError as exc:
            persistence_errors.append(exc)
            ok = False
            message += f"; could not save review status: {exc}"
        _emit_record(
            stream_lock,
            "result",
            message,
            started,
            ok=ok,
            chars=chars,
            extra={
                "artifacts_dir": str(artifacts.directory),
                "result_file": str(artifacts.result_path),
            },
            record_sink=save_record,
            style=args.progress,
        )
        return not persistence_errors

    heartbeat_thread: threading.Thread | None = None
    if args.heartbeat > 0:
        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            args=(args.heartbeat,),
            daemon=True,
            name="review-heartbeat",
        )
        heartbeat_thread.start()

    from review_mcp.review_tools import ReviewResult

    result_holder: list[ReviewResult] = []
    review_finished = threading.Event()

    def review_work() -> None:
        try:
            try:
                result = run_agentic_review(
                    working_dir=args.workspace,
                    diff_target=args.diff_target,
                    context_files=args.context_files,
                    focus_files=args.focus_files,
                    task_description=args.task_description,
                    include_trace=args.trace,
                    status_callback=emit,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                )
            except BaseException as exc:
                result = ReviewResult(f"Error: review crashed: {exc!r}", ok=False)
            try:
                artifacts.save_result(result, ok=result.ok)
            except OSError as exc:
                persistence_errors.append(exc)
                result = ReviewResult(
                    f"Error: Could not save review output: {exc}\n\n{result}", ok=False
                )
            result_holder.append(result)
        finally:
            heartbeat_state.stopped.set()
            review_finished.set()

    worker = threading.Thread(target=review_work, daemon=True, name="review-run")
    interrupted = False
    with _route_review_logs(emit, worker, review_finished):
        worker.start()
        try:
            worker.join(timeout_seconds)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            heartbeat_state.stopped.set()
            if heartbeat_thread is not None:
                try:
                    heartbeat_thread.join()
                except KeyboardInterrupt:
                    interrupted = True

        if interrupted:
            finish("interrupted", ok=False, chars=0)
            return 130

        timed_out = worker.is_alive()
        if timed_out:
            result = ReviewResult(
                f"Error: review timed out after {timeout_seconds:g}s in CLI mode. "
                "Try narrowing --focus, adding --context, or raising --timeout.",
                ok=False,
            )
        else:
            result = (
                result_holder[0]
                if result_holder
                else ReviewResult("No review generated.", ok=False)
            )

        ok = result.ok
        saved = finish(
            "review timed out"
            if timed_out
            else ("review finished" if ok else "review failed"),
            ok=ok,
            chars=len(result),
        )
        print(result)
        return 0 if ok and saved else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: review-mcp {run,models} ... (no arguments starts the MCP server)",
            file=sys.stderr,
        )
        return 2
    if argv[0] == "run":
        return run_main(argv[1:])
    if argv[0] == "models":
        return models_main(argv[1:])
    print(
        f"Error: unknown review-mcp mode {argv[0]!r}; expected 'run' or 'models'.",
        file=sys.stderr,
    )
    return 2
