"""Tests for the one-shot CLI mode (review-mcp run / models)."""

from __future__ import annotations

import json
import time

import pytest

import review_mcp.cli as cli
from review_mcp.review import run_agentic_review
from review_mcp.review_tools import ReviewResult


@pytest.fixture(autouse=True)
def artifact_runs(monkeypatch, tmp_path):
    """Isolate saved files and let timed-out test workers finish their cleanup."""
    import threading
    import review_mcp.artifacts as artifacts

    monkeypatch.setattr(artifacts.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    original = artifacts.ReviewArtifacts
    original_join = threading.Thread.join
    runs = []

    def create(request):
        run = original(request)
        runs.append(run)
        return run

    monkeypatch.setattr(artifacts, "ReviewArtifacts", create)
    yield runs
    for thread in threading.enumerate():
        if thread.name in {"review-run", "review-log-cleanup", "review-heartbeat"}:
            original_join(thread, timeout=3)


@pytest.fixture()
def fake_review(monkeypatch):
    """Replace run_agentic_review with a controllable fake."""
    holder = {"calls": [], "fn": None}

    def install(
        result="## Review\nLooks good.",
        status=("step one", "step two"),
        delay=0.0,
        ok=True,
    ):
        def fake(**kwargs):
            holder["calls"].append(kwargs)
            for message in status:
                if kwargs.get("status_callback"):
                    kwargs["status_callback"](message)
            if delay:
                time.sleep(delay)
            return ReviewResult(result, ok=ok)

        monkeypatch.setattr("review_mcp.review.run_agentic_review", fake)
        holder["fn"] = fake
        return holder

    return install


def _parse_records(err_text):
    lines = [line for line in err_text.splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_run_emits_ndjson_progress_and_review(capsys, fake_review):
    fake_review(result="## Review\nLooks good.")
    code = cli.run_main(["--progress", "ndjson", "--heartbeat", "0"])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == "## Review\nLooks good.\n"

    records = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
    statuses = [r for r in records if r["type"] == "status"]
    results = [r for r in records if r["type"] == "result"]
    assert "artifacts_dir" in statuses[0]
    assert [r["message"] for r in statuses[1:]] == ["step one", "step two"]
    assert len(results) == 1
    assert results[0]["ok"] is True
    assert results[0]["chars"] == len("## Review\nLooks good.")
    for record in records:
        assert record["schema_version"] == 1
        assert "ts" in record
        assert 0 <= record["elapsed"] < 60


def test_run_forwards_arguments(fake_review, capsys, tmp_path):
    holder = fake_review()
    cli.run_main(
        [
            str(tmp_path),
            "--diff-target",
            "HEAD~1",
            "--context",
            "docs/spec.md",
            "--context",
            "openspec/changes/x",
            "--focus",
            "server.py",
            "--task",
            "ship it",
            "--model",
            "glm-5.2",
            "--reasoning-effort",
            "low",
            "--trace",
            "--heartbeat",
            "0",
            "--progress",
            "none",
        ]
    )
    (call,) = holder["calls"]
    assert call["working_dir"] == str(tmp_path)
    assert call["diff_target"] == "HEAD~1"
    assert call["context_files"] == ["docs/spec.md", "openspec/changes/x"]
    assert call["focus_files"] == ["server.py"]
    assert call["task_description"] == "ship it"
    assert call["model"] == "glm-5.2"
    assert call["reasoning_effort"] == "low"
    assert call["include_trace"] is True
    capsys.readouterr()  # drain


def test_run_error_result_exits_nonzero(capsys, fake_review):
    fake_review(result="Error: model exploded.", ok=False)
    code = cli.run_main(["--heartbeat", "0", "--progress", "ndjson"])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out.startswith("Error: model exploded.")
    results = [r for r in _parse_records(captured.err) if r["type"] == "result"]
    assert results[0]["ok"] is False


def test_run_timeout_returns_error_and_exits_nonzero(capsys, fake_review):
    fake_review(delay=0.3)
    code = cli.run_main(
        ["--timeout", "0.05", "--heartbeat", "0", "--progress", "ndjson"]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out.startswith("Error: review timed out")
    results = [r for r in _parse_records(captured.err) if r["type"] == "result"]
    assert results[0]["ok"] is False
    assert results[0]["message"] == "review timed out"


def test_run_heartbeat_emitted_during_silence(capsys, fake_review):
    fake_review(status=(), delay=0.2)
    code = cli.run_main(["--heartbeat", "0.05", "--progress", "ndjson"])
    assert code == 0
    records = _parse_records(capsys.readouterr().err)
    heartbeats = [r for r in records if r["type"] == "heartbeat"]
    assert heartbeats, "expected at least one heartbeat during the silent review"
    assert heartbeats[0]["idle"] >= 0.05
    assert "Saved review files:" in heartbeats[0]["message"]


def test_run_plain_progress_format(capsys, fake_review):
    fake_review()
    code = cli.run_main(["--progress", "plain", "--heartbeat", "0"])
    assert code == 0
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.strip()]
    assert all(line.startswith("[") for line in lines if "s]" in line)
    assert not any(line.startswith("{") for line in lines)


def test_run_progress_none_is_silent(capsys, fake_review):
    fake_review()
    code = cli.run_main(["--progress", "none", "--heartbeat", "0"])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.err == ""


def test_run_defaults_workspace_to_cwd(fake_review, capsys):
    holder = fake_review()
    cli.run_main(["--heartbeat", "0", "--progress", "none"])
    (call,) = holder["calls"]
    assert call["working_dir"] == "."
    capsys.readouterr()


def test_format_model_list_marks_default_and_suffix_variants():
    lines = cli.format_model_list(
        ["glm-5.3", "glm-5.3[1m]", "glm-5.2"], "glm-5.3"
    ).splitlines()
    assert "glm-5.3  <- current default" in lines
    assert "glm-5.3[1m]  <- current default" in lines
    assert "glm-5.2" in lines


def test_format_model_list_flags_missing_default():
    lines = cli.format_model_list(["glm-5.2"], "glm-9.9").splitlines()
    assert lines[-1] == "Configured default: glm-9.9 (not offered by this endpoint)"


def test_main_dispatch(monkeypatch, fake_review):
    fake_review()
    assert cli.main(["run", "--heartbeat", "0", "--progress", "none"]) == 0
    assert cli.main(["bogus"]) == 2
    assert cli.main([]) == 2


def test_run_agentic_review_signature_includes_model_and_effort():
    import inspect

    parameters = inspect.signature(run_agentic_review).parameters
    assert "model" in parameters
    assert "reasoning_effort" in parameters


@pytest.mark.parametrize("failure", ["provider", "empty"])
@pytest.mark.parametrize("trace", [False, True])
def test_real_review_failures_have_unsuccessful_outcomes(
    monkeypatch, capsys, tmp_path, failure, trace
):
    from types import SimpleNamespace
    import review_mcp.review as review

    monkeypatch.setattr(review, "_make_client", lambda: object())

    def complete(*args, **kwargs):
        if failure == "provider":
            raise RuntimeError("400 bad request")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    monkeypatch.setattr(review, "_create_streaming_chat_completion", complete)
    code = cli.run_main(
        [
            str(tmp_path),
            "--heartbeat",
            "0",
            "--progress",
            "ndjson",
            *(["--trace"] if trace else []),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert _parse_records(captured.err)[-1]["ok"] is False
    assert (
        "Error calling model" if failure == "provider" else "No review generated."
    ) in captured.out


def test_successful_review_can_start_with_error_text(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace
    import review_mcp.review as review

    monkeypatch.setattr(review, "_make_client", lambda: object())
    monkeypatch.setattr(
        review,
        "_create_streaming_chat_completion",
        lambda *a, **kw: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Error: missing validation in the diff.",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        ),
    )
    assert (
        cli.run_main([str(tmp_path), "--heartbeat", "0", "--progress", "ndjson"]) == 0
    )
    assert _parse_records(capsys.readouterr().err)[-1]["ok"] is True


@pytest.mark.parametrize("style", ["ndjson", "plain", "none"])
def test_real_logging_obeys_progress_format_in_subprocess(style):
    import subprocess
    import sys

    script = (
        """
from unittest.mock import patch
from review_mcp.cli import run_main
with patch('review_mcp.review._make_client', return_value=object()), patch('review_mcp.review._create_streaming_chat_completion', side_effect=RuntimeError('400 bad request')):
    raise SystemExit(run_main(['.', '--heartbeat', '0', '--progress', %r]))
"""
        % style
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 1
    if style == "ndjson":
        records = _parse_records(result.stderr)
        assert records[-1]["ok"] is False
        assert any("400 bad request" in record["message"] for record in records)
    elif style == "none":
        assert result.stderr == ""
    else:
        assert result.stderr
        assert all(
            line.startswith("[") and "s]" in line for line in result.stderr.splitlines()
        )


@pytest.mark.parametrize(
    "option,values",
    [
        ("--timeout", ["0", "-1", "nan", "inf", "-inf", "1e300"]),
        ("--heartbeat", ["-1", "nan", "inf", "-inf", "1e300"]),
    ],
)
def test_invalid_timing_arguments_fail_before_starting_review(
    option, values, fake_review, capsys
):
    holder = fake_review()
    for value in values:
        with pytest.raises(SystemExit) as exc:
            cli.run_main([f"{option}={value}"])
        assert exc.value.code == 2
    assert holder["calls"] == []
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("style", ["ndjson", "plain", "none"])
def test_interrupt_respects_progress_format(monkeypatch, capsys, fake_review, style):
    import threading

    fake_review()
    original_join = threading.Thread.join

    def interrupt_worker(thread, *args, **kwargs):
        if thread.name == "review-run":
            original_join(thread, *args, **kwargs)
            raise KeyboardInterrupt
        return original_join(thread, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "join", interrupt_worker)
    assert cli.run_main(["--progress", style, "--heartbeat", "0"]) == 130
    captured = capsys.readouterr()
    if style == "none":
        assert captured.err == ""
    elif style == "plain":
        assert captured.err.splitlines()[-1].endswith("s] interrupted")
        assert not any(line.startswith("{") for line in captured.err.splitlines())
    else:
        assert _parse_records(captured.err)[-1]["message"] == "interrupted"
    assert captured.out == ""


@pytest.mark.parametrize("style", ["ndjson", "plain", "none"])
def test_invalid_timeout_environment_warning_obeys_progress_format(
    monkeypatch, capsys, fake_review, style
):
    fake_review()
    monkeypatch.setenv("REVIEW_TOOL_TIMEOUT_SECONDS", "invalid")
    assert cli.run_main(["--progress", style, "--heartbeat", "0"]) == 0
    captured = capsys.readouterr()
    if style == "ndjson":
        assert any(
            "Invalid REVIEW_TOOL_TIMEOUT_SECONDS" in r["message"]
            for r in _parse_records(captured.err)
        )
    elif style == "none":
        assert captured.err == ""
    else:
        assert all("s]" in line for line in captured.err.splitlines())


def test_cli_restores_logging_configuration(fake_review, capsys):
    import logging

    fake_review()
    loggers = [
        logging.getLogger(n) for n in ("ReviewMCP", "openai", "httpx", "httpcore")
    ]
    before = [(x.handlers[:], x.level, x.propagate) for x in loggers]
    assert cli.run_main(["--heartbeat", "0", "--progress", "ndjson"]) == 0
    assert [(x.handlers[:], x.level, x.propagate) for x in loggers] == before
    capsys.readouterr()


def test_real_review_outcome_serializes_as_mcp_text(monkeypatch, tmp_path):
    import anyio
    from mcp import Client
    from types import SimpleNamespace
    from server import create_mcp
    import review_mcp.review as review

    monkeypatch.setattr(review, "_make_client", lambda: object())
    monkeypatch.setattr(
        review,
        "_create_streaming_chat_completion",
        lambda *a, **kw: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="review text", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        ),
    )

    async def call():
        async with Client(create_mcp(str(tmp_path))) as client:
            result = await client.call_tool("review_with_context", {})
            assert result.structured_content["result"] == "review text"

    anyio.run(call)


@pytest.mark.parametrize("style", ["ndjson", "plain", "none"])
def test_second_interrupt_during_heartbeat_cleanup_is_handled(
    monkeypatch, capsys, fake_review, style
):
    import threading

    fake_review()
    original_join = threading.Thread.join

    def interrupt_after_join(thread, *args, **kwargs):
        original_join(thread, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(threading.Thread, "join", interrupt_after_join)
    assert cli.run_main(["--progress", style]) == 130
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    if style == "none":
        assert captured.err == ""
    elif style == "ndjson":
        assert _parse_records(captured.err)[-1]["message"] == "interrupted"
    else:
        assert captured.err.splitlines()[-1].endswith("s] interrupted")


@pytest.mark.parametrize("ok", [False, True])
def test_review_result_preserves_outcome_when_copied_or_pickled(ok):
    import copy
    import pickle

    result = ReviewResult("review text", ok=ok)
    for duplicate in (
        copy.copy(result),
        copy.deepcopy(result),
        pickle.loads(pickle.dumps(result)),
    ):
        assert duplicate == result
        assert duplicate.ok is ok


@pytest.mark.parametrize("style", ["ndjson", "plain", "none"])
def test_timeout_routes_worker_logs_until_stdout_is_written(monkeypatch, capsys, style):
    import builtins
    import logging
    import threading

    release = threading.Event()
    logged = threading.Event()

    def late_review(**kwargs):
        release.wait(2)
        logging.getLogger("ReviewMCP").error("late worker diagnostic")
        logged.set()
        return ReviewResult("late review", ok=True)

    def print_after_worker_logs(*args, **kwargs):
        release.set()
        assert logged.wait(2)
        builtins.print(*args, **kwargs)

    monkeypatch.setattr("review_mcp.review.run_agentic_review", late_review)
    monkeypatch.setattr(cli, "print", print_after_worker_logs, raising=False)
    assert (
        cli.run_main(["--progress", style, "--timeout", "0.01", "--heartbeat", "0"])
        == 1
    )
    captured = capsys.readouterr()
    assert "late worker diagnostic" not in captured.err
    if style == "none":
        assert captured.err == ""
    elif style == "ndjson":
        assert _parse_records(captured.err)[-1]["ok"] is False


@pytest.mark.parametrize("style", ["ndjson", "plain", "none"])
def test_timeout_mutes_late_worker_after_return_and_keeps_other_logs(
    monkeypatch, capsys, style
):
    import io
    import logging
    import threading

    release = threading.Event()
    logged = threading.Event()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("ReviewMCP")
    logger.addHandler(handler)
    root_filters_before = {h: h.filters[:] for h in logging.getLogger().handlers}

    def late_review(**kwargs):
        try:
            release.wait(2)
            logger.error("late review error after CLI return")
            logging.getLogger("httpx").warning(
                "late transport warning after CLI return"
            )
            return ReviewResult("late review", ok=True)
        finally:
            logged.set()

    monkeypatch.setattr("review_mcp.review.run_agentic_review", late_review)
    try:
        assert (
            cli.run_main(["--progress", style, "--timeout", "0.01", "--heartbeat", "0"])
            == 1
        )
        logger.error("unrelated main thread diagnostic")
        release.set()
        assert logged.wait(2)
        # Cleanup removes worker-specific filters after the review exits.
        deadline = time.monotonic() + 2
        while (
            handler.filters
            or any(h.filters != old for h, old in root_filters_before.items())
        ) and time.monotonic() < deadline:
            time.sleep(0.001)
        assert handler.filters == []
        assert all(h.filters == old for h, old in root_filters_before.items())
        assert stream.getvalue() == "unrelated main thread diagnostic\n"
        captured = capsys.readouterr()
        assert "late review error" not in captured.err
        assert "late transport warning" not in captured.err
    finally:
        release.set()
        logger.removeHandler(handler)


def test_default_progress_is_plain_and_suppresses_routine_logs(monkeypatch, capsys):
    import logging

    def review(**kwargs):
        logging.getLogger("ReviewMCP").info(
            "Payload size: 99999 chars across 20 messages"
        )
        kwargs["status_callback"]("Read server.py, lines 120–180")
        return ReviewResult("review text", ok=True)

    monkeypatch.setattr("review_mcp.review.run_agentic_review", review)
    assert cli.run_main(["--heartbeat", "0"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "review text\n"
    assert "Read server.py, lines 120–180" in captured.err
    assert "Payload size" not in captured.err
    assert not any(line.startswith("{") for line in captured.err.splitlines())


@pytest.mark.parametrize("style", ["plain", "ndjson", "none"])
def test_review_files_are_saved_in_every_progress_mode(
    fake_review, artifact_runs, capsys, style
):
    fake_review(result="Saved review text")
    assert cli.run_main(["--progress", style, "--heartbeat", "0"]) == 0
    run = artifact_runs[0]
    assert run.result_path.read_text() == "Saved review text\n"
    status = json.loads(run.status_path.read_text())
    assert status["state"] == "completed"
    assert status["result_available"] is True and status["result_ok"] is True
    assert status["exit_code"] == 0
    events = _parse_records(run.progress_path.read_text())
    assert events[0]["artifacts_dir"] == str(run.directory)
    assert [e["message"] for e in events if e["message"].startswith("step")] == [
        "step one",
        "step two",
    ]
    assert events[-1]["type"] == "result" and events[-1]["ok"] is True
    if __import__("os").name == "posix":
        assert run.directory.stat().st_mode & 0o777 == 0o700
    captured = capsys.readouterr()
    if style == "none":
        assert captured.err == ""
    else:
        assert str(run.directory) in captured.err


def test_result_is_saved_before_stdout_and_survives_broken_pipe(
    monkeypatch, fake_review, artifact_runs
):
    fake_review(result="Recover this review")

    def broken_stdout(*args, **kwargs):
        assert artifact_runs[0].result_path.read_text() == "Recover this review\n"
        raise BrokenPipeError("terminal disconnected")

    monkeypatch.setattr(cli, "print", broken_stdout, raising=False)
    with pytest.raises(BrokenPipeError):
        cli.run_main(["--progress", "none", "--heartbeat", "0"])
    assert (
        json.loads(artifact_runs[0].status_path.read_text())["result_available"] is True
    )


def test_timeout_keeps_progress_and_does_not_replace_late_review(
    monkeypatch, artifact_runs
):
    import threading

    release = threading.Event()

    def review(**kwargs):
        kwargs["status_callback"]("Reading example.py")
        release.wait(3)
        return ReviewResult("Late completed review", ok=True)

    monkeypatch.setattr("review_mcp.review.run_agentic_review", review)
    try:
        assert (
            cli.run_main(
                ["--timeout", "0.03", "--progress", "none", "--heartbeat", "0"]
            )
            == 1
        )
        run = artifact_runs[0]
        assert json.loads(run.status_path.read_text())["state"] == "timed_out"
        assert any(
            e["message"] == "Reading example.py"
            for e in _parse_records(run.progress_path.read_text())
        )
        assert not run.result_path.exists()
    finally:
        release.set()
    for worker in threading.enumerate():
        if worker.name == "review-run":
            worker.join(timeout=3)
    assert run.result_path.read_text() == "Late completed review\n"
    status = json.loads(run.status_path.read_text())
    assert status["state"] == "timed_out" and status["exit_code"] == 1
    assert status["result_available"] is True


def test_cannot_create_artifacts_fails_before_review(monkeypatch, fake_review, capsys):
    holder = fake_review()

    def fail(request):
        raise OSError("disk unavailable")

    monkeypatch.setattr("review_mcp.artifacts.ReviewArtifacts", fail)
    assert cli.run_main(["--progress", "ndjson"]) == 1
    assert holder["calls"] == []
    captured = capsys.readouterr()
    assert "Could not create saved review files" in captured.out
    assert _parse_records(captured.err)[-1]["ok"] is False


@pytest.mark.parametrize("stop", ["interrupt", "kill"])
def test_subprocess_interruption_preserves_saved_progress(tmp_path, stop):
    import subprocess
    import sys
    import os
    import signal

    script = """
import time
from unittest.mock import patch
from review_mcp.cli import run_main
from review_mcp.review_tools import ReviewResult
def review(**kwargs):
    kwargs['status_callback']('Reading interrupt-fixture.py')
    time.sleep(60)
    return ReviewResult('late', ok=True)
with patch('review_mcp.review.run_agentic_review', side_effect=review):
    raise SystemExit(run_main(['--progress', 'ndjson', '--heartbeat', '0']))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    events = []
    try:
        # The child flushes each event; once this event is visible its disk copy exists.
        while True:
            line = process.stderr.readline()
            assert line, "child exited before producing progress"
            record = json.loads(line)
            events.append(record)
            if record["message"] == "Reading interrupt-fixture.py":
                break
        directory = __import__("pathlib").Path(events[0]["artifacts_dir"])
        if stop == "interrupt":
            process.send_signal(signal.SIGINT)
        else:
            process.kill()
        stdout, stderr = process.communicate(timeout=10)
        status = json.loads((directory / "status.json").read_text())
        saved = _parse_records((directory / "progress.ndjson").read_text())
        assert any(e["message"] == "Reading interrupt-fixture.py" for e in saved)
        assert not (directory / "review.md").exists()
        if stop == "interrupt":
            assert process.returncode == 130
            assert status["state"] == "interrupted" and status["exit_code"] == 130
            assert saved[-1]["type"] == "result"
        else:
            assert process.returncode != 0
            assert status["state"] == "running" and status["result_available"] is False
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_result_write_failure_is_not_reported_as_success(
    monkeypatch, fake_review, artifact_runs, capsys
):
    fake_review()

    def fail(self, result, *, ok):
        raise OSError("result file is not writable")

    # artifact_runs wraps construction, but the class method is still on its returned instance.
    import review_mcp.artifacts as artifacts

    original_factory = artifacts.ReviewArtifacts

    def create(request):
        run = original_factory(request)
        monkeypatch.setattr(
            run, "save_result", lambda result, *, ok: fail(run, result, ok=ok)
        )
        return run

    monkeypatch.setattr(artifacts, "ReviewArtifacts", create)
    assert cli.run_main(["--progress", "ndjson", "--heartbeat", "0"]) == 1
    captured = capsys.readouterr()
    assert "Could not save review output" in captured.out
    assert _parse_records(captured.err)[-1]["ok"] is False
    status = json.loads(artifact_runs[0].status_path.read_text())
    assert status["state"] == "failed" and status["exit_code"] == 1
    assert status["result_available"] is False


def test_progress_write_failure_is_recorded_in_terminal_status(
    monkeypatch, fake_review, artifact_runs, capsys
):
    fake_review()
    import review_mcp.artifacts as artifacts

    original_factory = artifacts.ReviewArtifacts

    def fail(record):
        raise OSError("progress file is not writable")

    def create(request):
        run = original_factory(request)
        monkeypatch.setattr(run, "append_progress", fail)
        return run

    monkeypatch.setattr(artifacts, "ReviewArtifacts", create)
    assert cli.run_main(["--progress", "ndjson", "--heartbeat", "0"]) == 1
    captured = capsys.readouterr()
    assert _parse_records(captured.err)[-1]["ok"] is False
    status = json.loads(artifact_runs[0].status_path.read_text())
    assert status["state"] == "failed" and status["exit_code"] == 1
    assert status["result_available"] is True


def test_subprocess_kill_after_completion_preserves_review():
    import subprocess
    import sys
    from pathlib import Path

    script = """
import time
from unittest.mock import patch
from review_mcp.cli import run_main
from review_mcp.review_tools import ReviewResult
def disconnected_stdout(*args, **kwargs):
    time.sleep(60)
with patch('review_mcp.review.run_agentic_review', return_value=ReviewResult('Recovered completed review', ok=True)), patch('review_mcp.cli.print', disconnected_stdout, create=True):
    raise SystemExit(run_main(['--progress', 'ndjson', '--heartbeat', '0']))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    directory = None
    try:
        while True:
            line = process.stderr.readline()
            assert line, "child exited before saving a result"
            event = json.loads(line)
            if "artifacts_dir" in event:
                directory = Path(event["artifacts_dir"])
            if event["type"] == "result":
                break
        process.kill()
        stdout, _ = process.communicate(timeout=10)
        assert stdout == ""
        assert (directory / "review.md").read_text() == "Recovered completed review\n"
        status = json.loads((directory / "status.json").read_text())
        assert status["state"] == "completed" and status["result_available"] is True
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
