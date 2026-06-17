import unittest

import anyio

import server


class ServerWorkspaceTests(unittest.TestCase):
    def test_parse_args_has_no_implicit_workspace_default(self) -> None:
        args = server.parse_args([])

        self.assertIsNone(args.workspace_dir)

    def test_review_requires_workspace_when_no_default_is_configured(self) -> None:
        async def call_review_without_workspace() -> str:
            mcp = server.create_mcp()
            _, payload = await mcp.call_tool("review_with_context", {})
            return payload["result"]

        result = anyio.run(call_review_without_workspace)

        self.assertIn("No workspace directory was provided", result)
        self.assertIn("working_directory", result)


if __name__ == "__main__":
    unittest.main()
