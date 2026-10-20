import os
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
