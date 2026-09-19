import os
import unittest
from unittest import mock

from review_mcp.transport import _make_client


class MakeClientEnvTests(unittest.TestCase):
    def test_glm_api_key_is_accepted_as_fallback(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"GLM_API_KEY": "glm-test-key"},
                clear=True,
            ),
            mock.patch("openai.OpenAI") as openai_client,
        ):
            _make_client()

        self.assertEqual(
            openai_client.call_args.kwargs["api_key"],
            "glm-test-key",
        )

    def test_missing_api_key_error_lists_supported_aliases(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GLM_API_KEY"):
                _make_client()


if __name__ == "__main__":
    unittest.main()
