import os
import unittest
import time
from unittest import mock

import anyio
from mcp import Client

import server


class ServerWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        import review_mcp.artifacts as artifacts

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patch = mock.patch.object(artifacts.tempfile, "tempdir", temporary.name)
        patch.start()
        self.addCleanup(patch.stop)
        self.artifact_runs = []
        original = artifacts.ReviewArtifacts
        self.artifact_class = original

        def create(request):
            run = original(request)
            self.artifact_runs.append(run)
            return run

        patch = mock.patch.object(artifacts, "ReviewArtifacts", side_effect=create)
        patch.start()
        self.addCleanup(patch.stop)

    def test_parse_args_has_no_implicit_workspace_default(self) -> None:
        args = server.parse_args([])

        self.assertIsNone(args.workspace_dir)

    def test_review_requires_workspace_when_no_default_is_configured(self) -> None:
        async def call_review_without_workspace() -> str:
            async with Client(server.create_mcp()) as client:
                result = await client.call_tool("review_with_context", {})
                return result.structured_content["result"]

        result = anyio.run(call_review_without_workspace)

        self.assertIn("No workspace directory was provided", result)
        self.assertIn("working_directory", result)

    def test_tool_timeout_default_supports_long_reviews(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(server._get_tool_timeout_seconds(), 1800)

    def test_timeout_wrapper_returns_actionable_error(self) -> None:
        started = time.perf_counter()
        result = anyio.run(
            server._run_review_with_timeout,
            lambda: (time.sleep(0.2), "late result")[1],
            0.05,
        )
        elapsed = time.perf_counter() - started

        self.assertIn("review timed out after 0.05s", result)
        self.assertLess(elapsed, 0.2)

    def test_timeout_wrapper_reports_heartbeat_while_review_runs(self) -> None:
        statuses: list[tuple[str, float | None, float | None]] = []

        async def report_status(
            message: str,
            progress: float | None,
            total: float | None,
        ) -> None:
            statuses.append((message, progress, total))

        result = anyio.run(
            server._run_review_with_timeout,
            lambda: (time.sleep(0.08), "done")[1],
            1,
            report_status,
            0.02,
        )

        self.assertEqual(result, "done")
        self.assertTrue(any("Review still running" in status[0] for status in statuses))
        self.assertTrue(all(status[1] is not None for status in statuses))

    def test_idle_status_reports_last_action_only_after_inactivity(self) -> None:
        from types import SimpleNamespace
        clock = [0.0]
        statuses = []

        def review(**kwargs):
            kwargs["status_callback"]("Read server.py, lines 120–180")
            return "synthetic review"

        async def wrapper(review_call, timeout_seconds, status_reporter):
            result = await anyio.to_thread.run_sync(review_call)
            clock[0] = 90
            await status_reporter("heartbeat", 90, timeout_seconds)
            clock[0] = 130
            await status_reporter("heartbeat", 130, timeout_seconds)
            clock[0] = 260
            await status_reporter("heartbeat", 260, timeout_seconds)
            return result

        async def progress(value, total, message):
            statuses.append(message)

        async def call():
            async with Client(server.create_mcp(".")) as client:
                await client.call_tool("review_with_context", {}, progress_callback=progress)

        with (
            mock.patch.object(server, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0])),
            mock.patch("reviewer.run_agentic_review", side_effect=review),
            mock.patch("server._run_review_with_timeout", side_effect=wrapper),
        ):
            anyio.run(call)

        waiting = [message for message in statuses if message.startswith("Waiting")]
        self.assertEqual(waiting, [
            "Waiting 130s: Read server.py, lines 120–180",
            "Waiting 260s: Read server.py, lines 120–180",
        ])

    def test_mcp_review_saves_result_and_progress_before_returning(self) -> None:
        import json
        from review_mcp.review_tools import ReviewResult

        def review(**kwargs):
            kwargs["status_callback"]("Reading saved-fixture.py")
            return ReviewResult("Recoverable MCP review", ok=True)

        async def call():
            async with Client(server.create_mcp(".")) as client:
                result = await client.call_tool("review_with_context", {})
                self.assertEqual(result.structured_content["result"], "Recoverable MCP review")
                self.assertEqual(self.artifact_runs[0].result_path.read_text(), "Recoverable MCP review\n")

        with mock.patch("reviewer.run_agentic_review", side_effect=review):
            anyio.run(call)
        run = self.artifact_runs[0]
        status = json.loads(run.status_path.read_text())
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["request"]["entrypoint"], "mcp")
        self.assertTrue(status["result_available"])
        events = [json.loads(line) for line in run.progress_path.read_text().splitlines()]
        self.assertTrue(any(event["message"] == "Reading saved-fixture.py" for event in events))

    def test_mcp_persistence_failure_preserves_review_and_marks_failure(self) -> None:
        import json

        for method in ("append_progress", "save_result"):
            with self.subTest(method=method):
                async def call():
                    async with Client(server.create_mcp(".")) as client:
                        result = await client.call_tool("review_with_context", {})
                        return result.structured_content["result"]

                with (
                    mock.patch.object(self.artifact_class, method, side_effect=OSError("disk full")),
                    mock.patch("reviewer.run_agentic_review", return_value="Completed review text"),
                ):
                    result = anyio.run(call)
                self.assertIn("Completed review text", result)
                self.assertIn("Error: Could not save all review files: disk full", result)
                status = json.loads(self.artifact_runs[-1].status_path.read_text())
                self.assertEqual(status["state"], "failed")
                self.assertEqual(status["exit_code"], 1)

    def test_mcp_initial_notification_cancellation_is_preserved(self) -> None:
        import json
        from types import SimpleNamespace

        for fail_finish in (False, True):
            with self.subTest(fail_finish=fail_finish):
                async def call():
                    canceled = anyio.get_cancelled_exc_class()

                    async def notify(*args, **kwargs):
                        raise canceled()

                    tool = server.create_mcp(".")._tool_manager.get_tool("review_with_context")
                    with self.assertRaises(canceled):
                        await tool.fn(ctx=SimpleNamespace(report_progress=notify))

                with mock.patch("reviewer.run_agentic_review") as review:
                    if fail_finish:
                        with mock.patch.object(self.artifact_class, "finish", side_effect=OSError("disk full")):
                            anyio.run(call)
                    else:
                        anyio.run(call)
                    review.assert_not_called()
                if not fail_finish:
                    status = json.loads(self.artifact_runs[-1].status_path.read_text())
                    self.assertEqual(status["state"], "interrupted")
                    self.assertEqual(status["exit_code"], 130)

    def test_mcp_timeout_status_write_failure_returns_actionable_error(self) -> None:

        async def timed_out(*args):
            return "Error: review timed out"

        async def call():
            async with Client(server.create_mcp(".")) as client:
                result = await client.call_tool("review_with_context", {})
                return result.structured_content["result"]

        with (
            mock.patch("server._run_review_with_timeout", side_effect=timed_out),
            mock.patch.object(self.artifact_class, "finish", side_effect=OSError("disk full")),
        ):
            result = anyio.run(call)
        self.assertIn("review timed out", result)
        self.assertIn("Could not save all review files: disk full", result)

    def test_mcp_timeout_retains_late_worker_result(self) -> None:
        import json
        import threading
        from review_mcp.review_tools import ReviewResult
        release = threading.Event()

        def review(**kwargs):
            kwargs["status_callback"]("Reading timeout-fixture.py")
            release.wait(3)
            return ReviewResult("Late MCP review", ok=True)

        async def call():
            async with Client(server.create_mcp(".")) as client:
                result = await client.call_tool("review_with_context", {})
                self.assertIn("review timed out", result.structured_content["result"])

        with (
            mock.patch("reviewer.run_agentic_review", side_effect=review),
            mock.patch("server._get_tool_timeout_seconds", return_value=0.05),
        ):
            try:
                anyio.run(call)
                run = self.artifact_runs[0]
                self.assertEqual(json.loads(run.status_path.read_text())["state"], "timed_out")
            finally:
                release.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if json.loads(run.status_path.read_text())["result_available"]:
                    break
                time.sleep(0.005)
        self.assertEqual(run.result_path.read_text(), "Late MCP review\n")
        status = json.loads(run.status_path.read_text())
        self.assertEqual(status["state"], "timed_out")
        self.assertTrue(status["result_available"])

    def test_canceled_mcp_request_keeps_worker_result(self) -> None:
        import json
        import threading
        from review_mcp.review_tools import ReviewResult
        release = threading.Event()

        def review(**kwargs):
            kwargs["status_callback"]("Reading cancellation-fixture.py")
            release.wait(3)
            return ReviewResult("Recovered canceled MCP request", ok=True)

        async def call():
            ready = anyio.Event()
            done = anyio.Event()
            scopes = []

            async def progress(value, total, message):
                if message == "Reading cancellation-fixture.py":
                    ready.set()

            async with Client(server.create_mcp(".")) as client:
                async def request():
                    with anyio.CancelScope() as scope:
                        scopes.append(scope)
                        await client.call_tool("review_with_context", {}, progress_callback=progress)
                    done.set()

                async with anyio.create_task_group() as group:
                    group.start_soon(request)
                    with anyio.fail_after(3):
                        await ready.wait()
                        scopes[0].cancel()
                        await done.wait()
                        run = self.artifact_runs[0]
                        while json.loads(run.status_path.read_text())["state"] != "interrupted":
                            await anyio.sleep(0.005)
                        release.set()
                        while not json.loads(run.status_path.read_text())["result_available"]:
                            await anyio.sleep(0.005)

        with mock.patch("reviewer.run_agentic_review", side_effect=review):
            try:
                anyio.run(call)
            finally:
                release.set()
        run = self.artifact_runs[0]
        self.assertEqual(run.result_path.read_text(), "Recovered canceled MCP request\n")
        status = json.loads(run.status_path.read_text())
        self.assertEqual(status["state"], "interrupted")
        self.assertTrue(status["result_available"])

    def test_review_tool_schema_does_not_expose_context(self) -> None:
        async def list_review_tool():
            result = await server.create_mcp(".").list_tools()
            return next(tool for tool in result if tool.name == "review_with_context")

        tool = anyio.run(list_review_tool)

        self.assertNotIn("ctx", tool.input_schema["properties"])

    def test_review_tool_schema_includes_model_and_reasoning_effort(self) -> None:
        async def list_review_tool():
            result = await server.create_mcp(".").list_tools()
            return next(tool for tool in result if tool.name == "review_with_context")

        tool = anyio.run(list_review_tool)

        self.assertIn("model", tool.input_schema["properties"])
        self.assertIn("reasoning_effort", tool.input_schema["properties"])
        self.assertNotIn("model", tool.input_schema.get("required", []))
        self.assertNotIn("reasoning_effort", tool.input_schema.get("required", []))
        self.assertNotIn("ctx", tool.input_schema["properties"])

    def test_server_exposes_list_models_tool(self) -> None:
        async def list_tool_names():
            result = await server.create_mcp(".").list_tools()
            return {tool.name for tool in result}

        self.assertEqual(
            anyio.run(list_tool_names), {"review_with_context", "list_models"}
        )

    def test_list_models_marks_configured_default(self) -> None:
        async def call_list_models() -> str:
            async with Client(server.create_mcp(".")) as client:
                result = await client.call_tool("list_models", {})
                return result.structured_content["result"]

        with (
            mock.patch.dict(os.environ, {"AI_MODEL": "glm-5.3"}, clear=True),
            mock.patch(
                "reviewer._list_available_models",
                return_value=["glm-4.6", "glm-5.2", "glm-5.3"],
            ),
        ):
            result = anyio.run(call_list_models)

        self.assertIn("glm-4.6", result)
        self.assertIn("glm-5.2", result)
        self.assertIn("glm-5.3  <- current default", result)
        self.assertEqual(result.count("<-"), 1)

    def test_list_models_reports_missing_api_key(self) -> None:
        async def call_list_models() -> str:
            async with Client(server.create_mcp(".")) as client:
                result = await client.call_tool("list_models", {})
                return result.structured_content["result"]

        with mock.patch(
            "reviewer._list_available_models",
            side_effect=ValueError("AI_API_KEY environment variable is not set."),
        ):
            result = anyio.run(call_list_models)

        self.assertTrue(result.startswith("Error:"))
        self.assertIn("AI_API_KEY", result)

    def test_list_models_notes_default_not_offered(self) -> None:
        async def call_list_models() -> str:
            async with Client(server.create_mcp(".")) as client:
                result = await client.call_tool("list_models", {})
                return result.structured_content["result"]

        with (
            mock.patch.dict(os.environ, {"AI_MODEL": "glm-9"}, clear=True),
            mock.patch(
                "reviewer._list_available_models",
                return_value=["glm-5.2"],
            ),
        ):
            result = anyio.run(call_list_models)

        self.assertIn("glm-5.2", result)
        self.assertIn("Configured default: glm-9 (not offered by this endpoint)", result)

    def test_list_models_marks_base_id_for_variant_default(self) -> None:
        async def call_list_models() -> str:
            async with Client(server.create_mcp(".")) as client:
                result = await client.call_tool("list_models", {})
                return result.structured_content["result"]

        with (
            mock.patch.dict(
                os.environ, {"AI_MODEL": "glm-5.3-flash[1m]"}, clear=True
            ),
            mock.patch(
                "reviewer._list_available_models",
                return_value=["glm-5.2", "glm-5.3-flash"],
            ),
        ):
            result = anyio.run(call_list_models)

        self.assertIn("glm-5.3-flash  <- current default", result)
        self.assertNotIn("(not offered by this endpoint)", result)

    def test_review_with_context_returns_final_review(self) -> None:
        async def run_tool_flow() -> str:
            async with Client(server.create_mcp(".")) as client:
                result = await client.call_tool(
                    "review_with_context",
                    {"working_directory": "."},
                )
                return result.structured_content["result"]

        with mock.patch("reviewer.run_agentic_review", return_value="synthetic final review"):
            result = anyio.run(run_tool_flow)

        self.assertEqual(result, "synthetic final review")

    def test_review_with_context_threads_model_and_reasoning_effort(self) -> None:
        async def run_tool_flow() -> dict:
            async with Client(server.create_mcp(".")) as client:
                await client.call_tool(
                    "review_with_context",
                    {
                        "working_directory": ".",
                        "model": "glm-5.3",
                        "reasoning_effort": "low",
                    },
                )

        with mock.patch("reviewer.run_agentic_review", return_value="review") as review:
            anyio.run(run_tool_flow)

        self.assertEqual(review.call_args.kwargs["model"], "glm-5.3")
        self.assertEqual(review.call_args.kwargs["reasoning_effort"], "low")


if __name__ == "__main__":
    unittest.main()
