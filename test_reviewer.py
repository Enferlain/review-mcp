import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import reviewer


def _tool_message(
    name: str, arguments: dict, call_id: str = "call-1"
) -> SimpleNamespace:
    return SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            )
        ],
    )


def _model_response(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RunAgenticReviewEnvTests(unittest.TestCase):
    def test_model_requests_use_streaming_transport(self) -> None:
        fake_message = SimpleNamespace(content="review", tool_calls=None)
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = _model_response(fake_message)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("reviewer._make_client", return_value=fake_client):
                self.assertEqual(
                    reviewer.run_agentic_review(working_dir=tmpdir),
                    "review",
                )

        self.assertIs(
            fake_client.chat.completions.create.call_args.kwargs["stream"],
            True,
        )

    def test_streaming_completion_reassembles_fragmented_tool_calls(self) -> None:
        def chunk(*, content=None, tool_calls=None, finish_reason=None):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=finish_reason,
                        delta=SimpleNamespace(
                            content=content,
                            tool_calls=tool_calls,
                        ),
                    )
                ]
            )

        def tool_delta(index, *, call_id=None, name=None, arguments=None):
            function = None
            if name is not None or arguments is not None:
                function = SimpleNamespace(name=name, arguments=arguments)
            return SimpleNamespace(index=index, id=call_id, function=function)

        stream = mock.Mock()
        stream.__iter__ = mock.Mock(
            return_value=iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(finish_reason=None, delta=None)
                        ]
                    ),
                    chunk(
                        content="working ",
                        tool_calls=[tool_delta(0, call_id="call-", name="read_")],
                    ),
                    chunk(
                        content="now",
                        tool_calls=[
                            tool_delta(0, call_id="1", name="file", arguments='{\"path\":')
                        ]
                    ),
                    chunk(
                        tool_calls=[tool_delta(0, arguments='\"reviewer.py\"}')],
                        finish_reason="tool_calls",
                    ),
                ]
            )
        )
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = stream

        response = reviewer._create_streaming_chat_completion(
            fake_client,
            model="test-model",
            messages=[],
        )

        message = response.choices[0].message
        self.assertEqual(message.content, "working now")
        self.assertEqual(message.tool_calls[0].id, "call-1")
        self.assertEqual(message.tool_calls[0].function.name, "read_file")
        self.assertEqual(
            message.tool_calls[0].function.arguments,
            '{\"path\":\"reviewer.py\"}',
        )
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")
        stream.close.assert_called_once_with()

    def test_streaming_completion_accepts_complete_compatible_response(self) -> None:
        complete_response = _model_response(
            SimpleNamespace(content="complete", tool_calls=None)
        )
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = complete_response

        response = reviewer._create_streaming_chat_completion(
            fake_client,
            model="test-model",
            messages=[],
        )

        self.assertIs(response, complete_response)
        self.assertIs(
            fake_client.chat.completions.create.call_args.kwargs["stream"],
            True,
        )

    def test_make_client_uses_900_second_api_timeout_by_default(self) -> None:
        with mock.patch.dict(os.environ, {"AI_API_KEY": "test-key"}, clear=True):
            with mock.patch("openai.OpenAI") as openai_client:
                reviewer._make_client()

        self.assertEqual(openai_client.call_args.kwargs["timeout"], 900.0)

    def test_model_api_timeout_can_be_overridden(self) -> None:
        with mock.patch.dict(
            os.environ, {"AI_API_TIMEOUT_SECONDS": "1200"}, clear=False
        ):
            self.assertEqual(reviewer._get_model_api_timeout_seconds(), 1200.0)

    def test_invalid_model_api_timeout_falls_back_to_default(self) -> None:
        with mock.patch.dict(
            os.environ, {"AI_API_TIMEOUT_SECONDS": "invalid"}, clear=False
        ):
            self.assertEqual(reviewer._get_model_api_timeout_seconds(), 900.0)

    def test_invalid_max_review_iterations_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = mock.Mock()
            fake_message = mock.Mock()
            fake_message.tool_calls = None
            fake_message.content = "synthetic review"
            fake_response = mock.Mock()
            fake_response.choices = [mock.Mock(message=fake_message)]
            fake_client.chat.completions.create.return_value = fake_response

            with mock.patch.dict(
                os.environ, {"MAX_REVIEW_ITERATIONS": "not-a-number"}, clear=False
            ):
                with mock.patch("reviewer._make_client", return_value=fake_client):
                    result = reviewer.run_agentic_review(working_dir=tmpdir)

        self.assertEqual(result, "synthetic review")
        fake_client.chat.completions.create.assert_called_once()

    def test_run_agentic_review_reports_status_milestones(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = mock.Mock()
            fake_message = mock.Mock()
            fake_message.tool_calls = None
            fake_message.content = "synthetic review"
            fake_response = mock.Mock()
            fake_response.choices = [mock.Mock(message=fake_message)]
            fake_client.chat.completions.create.return_value = fake_response
            statuses: list[str] = []

            with mock.patch.dict(
                os.environ, {"AI_MODEL": "", "ZHIPU_MODEL": ""}, clear=False
            ):
                with mock.patch("reviewer._make_client", return_value=fake_client):
                    result = reviewer.run_agentic_review(
                        working_dir=tmpdir,
                        status_callback=statuses.append,
                    )

        self.assertEqual(result, "synthetic review")
        self.assertEqual(
            fake_client.chat.completions.create.call_args.kwargs["model"],
            reviewer.DEFAULT_REVIEW_MODEL,
        )
        self.assertTrue(
            any("Loading review context files" in status for status in statuses)
        )
        self.assertTrue(
            any(
                f"calling {reviewer.DEFAULT_REVIEW_MODEL}" in status
                for status in statuses
            )
        )
        self.assertTrue(any("Review complete" in status for status in statuses))

    def test_default_review_model_is_glm_5_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = mock.Mock()
            fake_message = mock.Mock()
            fake_message.tool_calls = None
            fake_message.content = "synthetic review"
            fake_response = mock.Mock()
            fake_response.choices = [mock.Mock(message=fake_message)]
            fake_client.chat.completions.create.return_value = fake_response

            with mock.patch.dict(
                os.environ, {"AI_MODEL": "", "ZHIPU_MODEL": ""}, clear=False
            ):
                with mock.patch("reviewer._make_client", return_value=fake_client):
                    reviewer.run_agentic_review(working_dir=tmpdir)

        self.assertEqual(
            fake_client.chat.completions.create.call_args.kwargs["model"], "glm-5.2"
        )

    def test_legacy_zhipu_model_env_still_overrides_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = mock.Mock()
            fake_message = mock.Mock()
            fake_message.tool_calls = None
            fake_message.content = "synthetic review"
            fake_response = mock.Mock()
            fake_response.choices = [mock.Mock(message=fake_message)]
            fake_client.chat.completions.create.return_value = fake_response

            with mock.patch.dict(
                os.environ,
                {"AI_MODEL": "", "ZHIPU_MODEL": "legacy-env-model"},
                clear=False,
            ):
                with mock.patch("reviewer._make_client", return_value=fake_client):
                    reviewer.run_agentic_review(working_dir=tmpdir)

        self.assertEqual(
            fake_client.chat.completions.create.call_args.kwargs["model"],
            "legacy-env-model",
        )

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

    def test_get_git_diff_scopes_to_focus_files(self) -> None:
        completed = mock.Mock(stdout="scoped diff")

        with mock.patch("reviewer._run_git_command", return_value=completed) as run_git:
            result = reviewer.get_git_diff(
                ".",
                "unstaged",
                scope_files=["library/a.py", "./library/b.py", "library/a.py"],
            )

        self.assertEqual(result, "scoped diff")
        args = run_git.call_args.args[1]
        self.assertEqual(args, ["diff", "--", "library/a.py", "library/b.py"])

    def test_get_changed_files_handles_missing_git(self) -> None:
        with mock.patch("reviewer._run_git_command", side_effect=FileNotFoundError):
            result = reviewer.get_changed_files(".")

        self.assertEqual(result, [])

    def test_get_changed_files_filters_to_focus_scope(self) -> None:
        staged = mock.Mock(stdout="library/a.py\nlibrary/c.py\n")
        unstaged = mock.Mock(stdout="library/b.py\nlibrary/c.py\n")

        with mock.patch("reviewer._run_git_command", side_effect=[staged, unstaged]):
            files = reviewer.get_changed_files(
                ".",
                scope_files=["library/b.py", "library/c.py", "missing.py"],
            )

        self.assertEqual(files, ["library/b.py", "library/c.py"])

    def test_execute_tool_lists_all_changed_files(self) -> None:
        with mock.patch(
            "reviewer.get_changed_files", return_value=["focused.py", "other.py"]
        ) as get_changed_files:
            result = reviewer._execute_tool(
                "list_changed_files",
                {},
                working_dir=".",
            )

        self.assertEqual(result, "focused.py\nother.py")
        get_changed_files.assert_called_once_with(".")

    def test_read_context_files_accepts_none_and_single_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context_file = Path(tmpdir) / "context.md"
            context_file.write_text("context body", encoding="utf-8")

            self.assertEqual(reviewer.read_context_files(None, tmpdir), "")
            self.assertEqual(reviewer.read_context_files(["", "   "], tmpdir), "")
            result = reviewer.read_context_files(str(context_file), tmpdir)

        self.assertIn("context body", result)

    def test_read_context_file_with_links_reports_read_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context_file = Path(tmpdir) / "context.md"
            context_file.write_text("context body", encoding="utf-8")

            with mock.patch.object(
                Path,
                "read_text",
                side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad"),
            ):
                content, diff_files = reviewer.read_context_file_with_links(
                    context_file, tmpdir
                )

        self.assertIn("Error reading context file", content)
        self.assertEqual(diff_files, [])

    def test_max_review_iterations_is_capped(self) -> None:
        with mock.patch.dict(os.environ, {"MAX_REVIEW_ITERATIONS": "500"}, clear=False):
            self.assertEqual(
                reviewer._get_max_iterations(), reviewer.MAX_ALLOWED_ITERATIONS
            )

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
                result = reviewer.run_agentic_review(
                    working_dir=tmpdir, include_trace=True
                )

        self.assertIn("synthetic review", result)
        self.assertIn("## Review Trace", result)
        self.assertIn(f"- Workspace: {tmpdir}", result)
        self.assertIn("- Diff target: staged", result)

    def test_empty_final_response_gets_one_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_message = mock.Mock()
            empty_message.tool_calls = None
            empty_message.content = ""
            final_message = mock.Mock()
            final_message.tool_calls = None
            final_message.content = "final review"

            fake_client = mock.Mock()
            fake_client.chat.completions.create.side_effect = [
                mock.Mock(choices=[mock.Mock(message=empty_message)]),
                mock.Mock(choices=[mock.Mock(message=final_message)]),
            ]

            with mock.patch("reviewer._make_client", return_value=fake_client):
                result = reviewer.run_agentic_review(
                    working_dir=tmpdir, include_trace=True
                )

        self.assertIn("final review", result)
        self.assertIn("model returned an empty final response", result)
        self.assertEqual(fake_client.chat.completions.create.call_count, 2)

    def test_structured_final_response_content_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = mock.Mock()
            fake_message = mock.Mock()
            fake_message.tool_calls = None
            fake_message.content = [{"type": "text", "text": "structured review"}]
            fake_response = mock.Mock()
            fake_response.choices = [mock.Mock(message=fake_message)]
            fake_client.chat.completions.create.return_value = fake_response

            with mock.patch("reviewer._make_client", return_value=fake_client):
                result = reviewer.run_agentic_review(working_dir=tmpdir)

        self.assertEqual(result, "structured review")

    def test_object_final_response_content_is_supported(self) -> None:
        class TextPart:
            text = "object review"

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = mock.Mock()
            fake_message = mock.Mock()
            fake_message.tool_calls = None
            fake_message.content = [TextPart()]
            fake_response = mock.Mock()
            fake_response.choices = [mock.Mock(message=fake_message)]
            fake_client.chat.completions.create.return_value = fake_response

            with mock.patch("reviewer._make_client", return_value=fake_client):
                result = reviewer.run_agentic_review(working_dir=tmpdir)

        self.assertEqual(result, "object review")

    def test_model_can_be_overridden_by_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = mock.Mock()
            fake_message = mock.Mock()
            fake_message.tool_calls = None
            fake_message.content = "synthetic review"
            fake_response = mock.Mock()
            fake_response.choices = [mock.Mock(message=fake_message)]
            fake_client.chat.completions.create.return_value = fake_response

            with mock.patch.dict(
                os.environ, {"AI_MODEL": "custom-review-model"}, clear=False
            ):
                with mock.patch("reviewer._make_client", return_value=fake_client):
                    reviewer.run_agentic_review(working_dir=tmpdir)

        self.assertEqual(
            fake_client.chat.completions.create.call_args.kwargs["model"],
            "custom-review-model",
        )

    def test_api_error_mentions_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = mock.Mock()
            fake_client.chat.completions.create.side_effect = RuntimeError("boom")

            with mock.patch.dict(
                os.environ, {"AI_MODEL": "custom-review-model"}, clear=False
            ):
                with mock.patch("reviewer._make_client", return_value=fake_client):
                    result = reviewer.run_agentic_review(working_dir=tmpdir)

        self.assertIn("Error calling model 'custom-review-model' API", result)

    def test_non_dict_tool_arguments_are_ignored(self) -> None:
        result = reviewer._execute_tool(
            "read_files", ["not", "a", "dict"], working_dir="."
        )

        self.assertEqual(result, "")

    def test_get_uncommitted_changes_defaults_to_staged_and_unstaged(self) -> None:
        staged = mock.Mock(stdout="staged diff")
        unstaged = mock.Mock(stdout="unstaged diff")

        with mock.patch(
            "reviewer._run_git_command", side_effect=[staged, unstaged]
        ) as run_git:
            result = reviewer._execute_tool(
                "get_uncommitted_changes",
                {},
                working_dir=".",
            )

        self.assertIn("# Staged changes\nstaged diff", result)
        self.assertIn("# Unstaged changes\nunstaged diff", result)
        self.assertEqual(run_git.call_count, 2)

    def test_repository_tree_search_and_targeted_read_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "src" / "nested"
            source_dir.mkdir(parents=True)
            source_file = source_dir / "example.py"
            source_file.write_text(
                "\n".join(f"line {line}: needle" for line in range(1, 31)),
                encoding="utf-8",
            )

            tree = reviewer.list_repository_tree(tmpdir, max_depth=3)
            search = reviewer.search_repository("needle", tmpdir, max_results=3)
            excerpt = reviewer.read_repository_file(
                "src/nested/example.py",
                tmpdir,
                start_line=10,
                end_line=12,
            )
            escaped = reviewer.read_repository_file("../outside.py", tmpdir)

        self.assertIn("src/", tree)
        self.assertIn("nested/", tree)
        self.assertIn("example.py", tree)
        self.assertEqual(search.count("needle"), 3)
        self.assertIn("TRUNCATED after 3 matches", search)
        self.assertIn("10 | line 10: needle", excerpt)
        self.assertIn("12 | line 12: needle", excerpt)
        self.assertNotIn("line 13", excerpt)
        self.assertIn("outside the repository", escaped)

    def test_repository_search_has_its_own_deadline(self) -> None:
        class BlockingStream:
            def __iter__(self):
                time.sleep(0.05)
                return iter(())

        class FakeProcess:
            stdout = BlockingStream()
            stderr = iter(())
            returncode = None

            def terminate(self) -> None:
                self.returncode = -15

            def kill(self) -> None:
                self.returncode = -9

            def wait(self, timeout=None) -> int:
                return self.returncode or 0

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch(
                    "reviewer.subprocess.Popen",
                    return_value=FakeProcess(),
                ),
                mock.patch(
                    "reviewer.REPOSITORY_SEARCH_TIMEOUT_SECONDS",
                    0.001,
                ),
            ):
                result = reviewer.search_repository("needle", tmpdir)

        self.assertIn("timed out", result)

    def test_truncated_initial_focus_diff_is_not_readded_by_uncommitted_tool(
        self,
    ) -> None:
        tool_message = _tool_message(
            "get_uncommitted_changes",
            {"target": "unstaged"},
        )
        final_message = SimpleNamespace(content="final review", tool_calls=None)
        responses = iter(
            [_model_response(tool_message), _model_response(final_message)]
        )
        captured_messages: list[list[dict | object]] = []

        def create_response(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return next(responses)

        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = create_response
        large_diff = "same diff\n" * 3000
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict(
                    os.environ,
                    {"MAX_REVIEW_CONTEXT_CHARS": "10000"},
                    clear=False,
                ),
                mock.patch("reviewer._make_client", return_value=fake_client),
                mock.patch(
                    "reviewer.get_git_diff", return_value=large_diff
                ) as get_diff,
            ):
                result = reviewer.run_agentic_review(
                    working_dir=tmpdir,
                    diff_target="unstaged",
                    focus_files=["reviewer.py"],
                    include_trace=True,
                )

        self.assertIn("final review", result)
        self.assertIn("Initial diff truncated", result)
        self.assertIn("Tool result deduplicated: get_uncommitted_changes", result)
        self.assertLessEqual(reviewer._messages_size_chars(captured_messages[0]), 10000)
        self.assertEqual(
            get_diff.call_args_list[0].kwargs["scope_files"], ["reviewer.py"]
        )
        self.assertNotIn("scope_files", get_diff.call_args_list[1].kwargs)
        tool_results = [
            message["content"]
            for message in captured_messages[1]
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        self.assertEqual(len(tool_results), 1)
        self.assertIn("already included", tool_results[0])
        self.assertNotIn("same diff", tool_results[0])

    def test_equivalent_default_read_ranges_share_a_dedup_key(self) -> None:
        omitted = reviewer._normalized_tool_call_key(
            "read_file",
            {"path": "reviewer.py"},
            ".",
        )
        explicit_null = reviewer._normalized_tool_call_key(
            "read_file",
            {"path": "reviewer.py", "end_line": None},
            ".",
        )

        self.assertEqual(omitted, explicit_null)

    def test_read_files_handles_null_and_invalid_paths(self) -> None:
        self.assertEqual(
            reviewer._execute_tool(
                "read_files",
                {"paths": None},
                working_dir=".",
            ),
            "",
        )
        self.assertIn(
            "paths must be a list",
            reviewer._execute_tool(
                "read_files",
                {"paths": 123},
                working_dir=".",
            ),
        )

    def test_repeated_tool_request_is_deduplicated(self) -> None:
        first_tool_message = _tool_message("read_file", {"path": "sample.py"}, "call-1")
        second_tool_message = _tool_message(
            "read_file", {"path": "sample.py"}, "call-2"
        )
        final_message = SimpleNamespace(content="final review", tool_calls=None)
        responses = iter(
            [
                _model_response(first_tool_message),
                _model_response(second_tool_message),
                _model_response(final_message),
            ]
        )
        captured_messages: list[list[dict | object]] = []

        def create_response(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return next(responses)

        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = create_response
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "sample.py").write_text("first\nsecond\n", encoding="utf-8")
            with mock.patch("reviewer._make_client", return_value=fake_client):
                result = reviewer.run_agentic_review(
                    working_dir=tmpdir,
                    include_trace=True,
                )

        self.assertIn("Tool call deduplicated: read_file", result)
        tool_results = [
            message["content"]
            for message in captured_messages[2]
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        self.assertIn("first", tool_results[0])
        self.assertIn("exact tool request was already completed", tool_results[1])
        self.assertNotIn("first", tool_results[1])

    def test_tool_exception_is_contained_and_review_can_finish(self) -> None:
        tool_message = _tool_message("list_repository_tree", {})
        final_message = SimpleNamespace(content="final review", tool_calls=None)
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = [
            _model_response(tool_message),
            _model_response(final_message),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch(
                    "reviewer._make_client",
                    return_value=fake_client,
                ),
                mock.patch(
                    "reviewer._execute_tool",
                    side_effect=OSError("synthetic tool failure"),
                ),
            ):
                result = reviewer.run_agentic_review(
                    working_dir=tmpdir,
                    include_trace=True,
                )

        self.assertIn("final review", result)
        self.assertIn("Tool error contained: list_repository_tree", result)

    def test_tool_result_and_total_context_are_hard_bounded(self) -> None:
        tool_message = _tool_message(
            "read_file",
            {"path": "large.py", "start_line": 1, "end_line": 400},
        )
        final_message = SimpleNamespace(content="final review", tool_calls=None)
        responses = iter(
            [_model_response(tool_message), _model_response(final_message)]
        )
        captured_messages: list[list[dict | object]] = []

        def create_response(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return next(responses)

        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = create_response
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "large.py").write_text(
                "\n".join("x" * 200 for _ in range(500)),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "MAX_REVIEW_CONTEXT_CHARS": "10000",
                    "MAX_REVIEW_TOOL_RESULT_CHARS": "2000",
                },
                clear=False,
            ):
                with mock.patch("reviewer._make_client", return_value=fake_client):
                    result = reviewer.run_agentic_review(
                        working_dir=tmpdir,
                        include_trace=True,
                    )

        self.assertIn("final review", result)
        self.assertIn("Tool result truncated: read_file", result)
        tool_result = next(
            message["content"]
            for message in captured_messages[1]
            if isinstance(message, dict) and message.get("role") == "tool"
        )
        self.assertLessEqual(len(tool_result), 2000)
        self.assertIn("TRUNCATED", tool_result)
        self.assertLessEqual(reviewer._messages_size_chars(captured_messages[1]), 10000)

    def test_openspec_change_directory_expands_to_context_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            change_dir = (
                Path(tmpdir) / "openspec" / "changes" / "resource-intelligence-system"
            )
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

    def test_run_agentic_review_includes_openspec_change_directory_context(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            change_dir = (
                Path(tmpdir) / "openspec" / "changes" / "resource-intelligence-system"
            )
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
