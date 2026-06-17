import reviewer
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

target_dir = r"d:\Projects\model-benchmark-explorer"
print(f"Starting review test on {target_dir}...")

try:
    result = reviewer.run_agentic_review(
        working_dir=target_dir,
        diff_target="staged",
        task_description="Refactored Dashboard.tsx into modular components and custom hooks to improve maintainability and readability. Resolved ReferenceErrors and fixed prop mismatches.",
    )
    print("\n--- RESULT ---")
    print(result)
except Exception as e:
    print(f"\nERROR: {e}")
