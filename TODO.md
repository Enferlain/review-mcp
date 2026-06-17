- Add gemini-cli support
- Add the ability to specify reviewer skills for specific review aspects (frontend specialist, tests specialist, python specialist, etc.)
- Add a way for the agent to create files, would need to use a sandbox or docker or something, like the docker mcp (check for reference)
- Add local llm support (openai api/ollama/koboldcpp/llamacpp)
- Add separate tool for pure review where the full repo is accessible
- Add separate tool to review pull requests and comments/commits in it
- Add separate tool to check issues if open
- Maybe split off to a github bot?
- Do a check if functionality is correct or if there are any gaps and make existing functionality stronger particularly in what artifacts can be accessed (need to check task/walkthrough/implementation plan formats in brain folder)
- organize/modernize folder structure even if it needs placeholder files

codex:

A few quick ones, yeah.

The main annoyance is that the repo mixes real code, manual debug scripts, and test-like files in the top level, which makes discovery noisy and led directly to the confusing “tests” situation. I’d probably clean that up first by moving things like `debug_api.py`, `check_quota.py`, `heavy_test.py`, and `reproduce_user_error.py` into a `scripts/` or `experiments/` folder.

The other small friction points are in [`reviewer.py`](/mnt/d/Projects/review_mcp/reviewer.py):
- cap `MAX_REVIEW_ITERATIONS` on the high end too
- log JSON tool-arg decode failures instead of silently swallowing them
- either add a real `read_file` tool definition or remove the dead handler so the tool surface is consistent

If you want, I can knock out those cleanup fixes next in one pass.