import os
import unittest
from types import SimpleNamespace
from unittest import mock

from review_mcp.transport import _list_available_models, _make_client


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


class ListAvailableModelsTests(unittest.TestCase):
    def test_returns_sorted_ids_with_short_timeout(self) -> None:
        fake_client = mock.Mock()
        fake_client.with_options.return_value.models.list.return_value = (
            SimpleNamespace(
                data=[SimpleNamespace(id="glm-5.3"), SimpleNamespace(id="glm-4.6")]
            )
        )

        with (
            mock.patch.dict(os.environ, {"GLM_API_KEY": "glm-test-key"}, clear=True),
            mock.patch("openai.OpenAI", return_value=fake_client),
        ):
            model_ids = _list_available_models()

        self.assertEqual(model_ids, ["glm-4.6", "glm-5.3"])
        fake_client.with_options.assert_called_once_with(timeout=15.0)

    def test_propagates_missing_key_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                _list_available_models()


if __name__ == "__main__":
    unittest.main()
