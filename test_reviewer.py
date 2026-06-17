import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import reviewer


class RunAgenticReviewEnvTests(unittest.TestCase):
    def test_invalid_max_review_iterations_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = mock.Mock()
            fake_message = mock.Mock()
            fake_message.tool_calls = None
            fake_message.content = "synthetic review"
            fake_response = mock.Mock()
            fake_response.choices = [mock.Mock(message=fake_message)]
            fake_client.chat.completions.create.return_value = fake_response

            with mock.patch.dict(os.environ, {"MAX_REVIEW_ITERATIONS": "not-a-number"}, clear=False):
                with mock.patch("reviewer._make_client", return_value=fake_client):
                    result = reviewer.run_agentic_review(working_dir=tmpdir)

        self.assertEqual(result, "synthetic review")
        fake_client.chat.completions.create.assert_called_once()

    def test_get_git_diff_builds_single_pathspec_separator(self) -> None:
        completed = mock.Mock(stdout="diff output")

        with mock.patch("reviewer._run_git_command", return_value=completed) as run_git:
            result = reviewer.get_git_diff(".", "staged")

        self.assertEqual(result, "diff output")
        args = run_git.call_args.args[1]
        self.assertEqual(args[:2], ["diff", "--staged"])
        self.assertEqual(args.count("--"), 1)
        self.assertGreater(len(args), 3)
        self.assertTrue(all(part.startswith(":!") for part in args[3:]))

    def test_max_review_iterations_is_capped(self) -> None:
        with mock.patch.dict(os.environ, {"MAX_REVIEW_ITERATIONS": "500"}, clear=False):
            self.assertEqual(reviewer._get_max_iterations(), reviewer.MAX_ALLOWED_ITERATIONS)

    def test_include_trace_appends_diagnostic_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = mock.Mock()
            fake_message = mock.Mock()
            fake_message.tool_calls = None
            fake_message.content = "synthetic review"
            fake_response = mock.Mock()
            fake_response.choices = [mock.Mock(message=fake_message)]
            fake_client.chat.completions.create.return_value = fake_response

            with mock.patch("reviewer._make_client", return_value=fake_client):
                result = reviewer.run_agentic_review(working_dir=tmpdir, include_trace=True)

        self.assertIn("synthetic review", result)
        self.assertIn("## Review Trace", result)
        self.assertIn(f"- Workspace: {tmpdir}", result)
        self.assertIn("- Diff target: staged", result)

    def test_openspec_change_directory_expands_to_context_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            change_dir = Path(tmpdir) / "openspec" / "changes" / "resource-intelligence-system"
            spec_dir = change_dir / "specs" / "resource-intelligence"
            spec_dir.mkdir(parents=True)
            (change_dir / "proposal.md").write_text("Proposal text", encoding="utf-8")
            (change_dir / "tasks.md").write_text("Tasks text", encoding="utf-8")
            (spec_dir / "spec.md").write_text("Spec text", encoding="utf-8")

            expanded = reviewer.expand_context_entry(str(change_dir), tmpdir)

        self.assertEqual(
            [path.name for path in expanded],
            ["proposal.md", "tasks.md", "spec.md"],
        )

    def test_run_agentic_review_includes_openspec_change_directory_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            change_dir = Path(tmpdir) / "openspec" / "changes" / "resource-intelligence-system"
            spec_dir = change_dir / "specs" / "resource-intelligence"
            spec_dir.mkdir(parents=True)
            (change_dir / "proposal.md").write_text("Proposal text", encoding="utf-8")
            (change_dir / "tasks.md").write_text("Tasks text", encoding="utf-8")
            (spec_dir / "spec.md").write_text("Spec text", encoding="utf-8")

            fake_client = mock.Mock()
            fake_message = mock.Mock()
            fake_message.tool_calls = None
            fake_message.content = "synthetic review"
            fake_response = mock.Mock()
            fake_response.choices = [mock.Mock(message=fake_message)]
            fake_client.chat.completions.create.return_value = fake_response

            with mock.patch("reviewer._make_client", return_value=fake_client):
                result = reviewer.run_agentic_review(
                    working_dir=tmpdir,
                    context_files=[str(change_dir)],
                    include_trace=True,
                )

        self.assertIn("synthetic review", result)
        self.assertIn("Context files loaded: 3", result)
        messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
        user_message = messages[1]["content"]
        self.assertIn("--- CONTEXT DIRECTORY:", user_message)
        self.assertIn("Proposal text", user_message)
        self.assertIn("Tasks text", user_message)
        self.assertIn("Spec text", user_message)


if __name__ == "__main__":
    unittest.main()
