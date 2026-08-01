import unittest

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


if __name__ == "__main__":
    unittest.main()
