from __future__ import annotations

import contextlib
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_LOCK_DIR = Path(__file__).resolve().parent / ".locks"

_K8S_NAMESPACE_MAX_LEN = 63
_MAX_LEN = 63


def _slug(value: object, sep: str = "-") -> str:
    s = re.sub(r"[^a-z0-9]+", sep, str(value).strip().lower()).strip(sep)
    return s or "x"


def _stable_short_hash(*parts: object, length: int = 8) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:length]


def _derive(prefix: str, worker_id: object, rollout_id: Optional[object], sep: str, max_len: int) -> str:
    slug = _slug(worker_id, sep) if rollout_id is None else f"{_slug(worker_id, sep)}{sep}{_slug(rollout_id, sep)}"
    digest = _stable_short_hash(worker_id, rollout_id)
    name = f"{prefix}{sep}{slug}{sep}{digest}"
    if len(name) > max_len:
        name = f"{prefix}{sep}{digest}"[:max_len]
    return name.strip(sep)


def derive_k8s_namespace(worker_id: object, rollout_id: Optional[object] = None, prefix: str = "aiops-rl") -> str:
    return _derive(prefix, worker_id, rollout_id, "-", _K8S_NAMESPACE_MAX_LEN)


def derive_docker_prefix(worker_id: object, rollout_id: Optional[object] = None, prefix: str = "aiops-rl") -> str:
    return _derive(prefix, worker_id, rollout_id, "-", _MAX_LEN)


def derive_milvus_collection_suffix(worker_id: object, rollout_id: Optional[object] = None, prefix: str = "aiops_rl") -> str:
    name = _derive(prefix, worker_id, rollout_id, "_", _MAX_LEN)
    if not re.match(r"^[A-Za-z_]", name):
        name = "_" + name
    return name


class WorkerLockTimeout(TimeoutError):
    pass


def _lock_path(resource_key: object, lock_dir: Optional[Path]) -> Path:
    lock_dir = Path(lock_dir) if lock_dir is not None else DEFAULT_LOCK_DIR
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(resource_key)) or "resource"
    return lock_dir / f"{safe}.lock"


@contextlib.contextmanager
def acquire_worker_lock(
    worker_id: object,
    resource_key: object,
    lock_dir: Optional[Path] = None,
    timeout: Optional[float] = None,
    poll_interval: float = 0.05,
) -> Iterator[None]:
    import fcntl

    path = _lock_path(resource_key, lock_dir)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise WorkerLockTimeout(
                        f"worker {worker_id!r} timed out after {timeout}s waiting for lock on {resource_key!r}"
                    )
                time.sleep(poll_interval)
        os.ftruncate(fd, 0)
        os.write(fd, str(worker_id).encode("utf-8"))
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
