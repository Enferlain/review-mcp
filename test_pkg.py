import reviewer
import sys
import logging
import os


def main() -> None:
    """Run a quick manual smoke test against the current working directory."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    print("Starting review test...")
    try:
        result = reviewer.run_agentic_review(working_dir=os.getcwd(), diff_target="staged")
        print("\n--- RESULT ---")
        print(result)
    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
