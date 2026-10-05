"""Cross-process ownership lock for reminder delivery."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import BinaryIO


class SchedulerAlreadyRunningError(RuntimeError):
    """Raised when another process already owns reminder delivery."""


class ReminderSchedulerLock:
    def __init__(self, database_path: str | Path):
        path = str(database_path)
        self.path = None if path == ":memory:" else Path(path).expanduser().resolve()
        self._file: BinaryIO | None = None

    def acquire(self) -> None:
        if self.path is None or self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".scheduler.lock")
        lock_file = lock_path.open("a+b")
        if lock_path.stat().st_size == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            lock_file.close()
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise SchedulerAlreadyRunningError(
                    "Another reminder scheduler already owns this database"
                ) from error
            raise
        self._file = lock_file

    @property
    def is_owned(self) -> bool:
        return self.path is None or self._file is not None

    def release(self) -> None:
        lock_file = self._file
        self._file = None
        if lock_file is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()