"""Recoverable review files, independent of client or terminal output."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time


class ReviewArtifacts:
    """Keep each run in a private temporary directory, without auto-deletion."""

    def __init__(self, request: dict) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="review-mcp-run-"))
        self.result_path = self.directory / "review.md"
        self.progress_path = self.directory / "progress.ndjson"
        self.status_path = self.directory / "status.json"
        self._lock = threading.Lock()
        self._status = {
            "schema_version": 1,
            "pid": os.getpid(),
            "started_at": time.time(),
            "cwd": str(Path.cwd()),
            "state": "running",
            "result_available": False,
            "request": request,
        }
        self.progress_path.touch(mode=0o600)
        self._write_status()

    def _atomic_write(self, path: Path, text: str) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                delete=False,
                prefix=".pending-",
            ) as stream:
                temporary = Path(stream.name)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _write_status(self) -> None:
        self._atomic_write(self.status_path, json.dumps(self._status, indent=2) + "\n")

    def append_progress(self, record: dict) -> None:
        with self._lock, self.progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    def save_result(self, result: str, *, ok: bool) -> None:
        with self._lock:
            self._atomic_write(self.result_path, result + "\n")
            self._status.update(result_available=True, result_ok=ok, chars=len(result))
            if self._status["state"] == "running":
                self._status.update(
                    state="completed" if ok else "failed",
                    exit_code=0 if ok else 1,
                    finished_at=time.time(),
                )
            self._write_status()

    def finish(self, state: str, *, exit_code: int, message: str) -> None:
        with self._lock:
            self._status.update(
                state=state,
                exit_code=exit_code,
                finished_at=time.time(),
                message=message,
            )
            self._write_status()
