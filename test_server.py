import unittest
import time
from unittest import mock

import anyio
from mcp import Client

import server


class ServerWorkspaceTests(unittest.TestCase):
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

    def test_review_tool_schema_does_not_expose_context(self) -> None:
        async def list_review_tool():
            result = await server.create_mcp(".").list_tools()
            return next(tool for tool in result if tool.name == "review_with_context")

        tool = anyio.run(list_review_tool)

        self.assertNotIn("ctx", tool.input_schema["properties"])

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


if __name__ == "__main__":
    unittest.main()
