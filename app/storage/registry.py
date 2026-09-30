"""Run registry: run snapshots, queue, trace log, idempotency cache.

Redis is the production backend. A file-backed registry implements the same
interface for environments without Redis (used by the offline integration
test); it is NOT the production path.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path


class RunRegistry(ABC):
    @abstractmethod
    def create_run(self, run_id: str, question: str, snapshot_json: str) -> None: ...
    @abstractmethod
    def enqueue(self, run_id: str) -> None: ...
    @abstractmethod
    def claim(self, timeout: int = 5) -> str | None: ...
    @abstractmethod
    def save_snapshot(self, run_id: str, snapshot_json: str) -> None: ...
    @abstractmethod
    def get_snapshot(self, run_id: str) -> str | None: ...
    @abstractmethod
    def append_trace(self, run_id: str, step_json: str) -> None: ...
    @abstractmethod
    def get_trace(self, run_id: str) -> list[str]: ...
    @abstractmethod
    def idem_get(self, key: str) -> str | None: ...
    @abstractmethod
    def idem_set(self, key: str, value: str, ttl_sec: int = 86400) -> None: ...
    @abstractmethod
    def running_run_ids(self) -> list[str]: ...
    @abstractmethod
    def ping(self) -> bool: ...


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


class RedisRunRegistry(RunRegistry):
    QUEUE = "research:queue"
    PROCESSING = "research:processing"

    def __init__(self, redis_client):
        self.r = redis_client

    # -- runs ----------------------------------------------------------
    def _run_key(self, run_id: str) -> str:
        return f"research:run:{run_id}"

    def create_run(self, run_id: str, question: str, snapshot_json: str) -> None:
        self.r.set(self._run_key(run_id), snapshot_json)

    def save_snapshot(self, run_id: str, snapshot_json: str) -> None:
        self.r.set(self._run_key(run_id), snapshot_json)

    def get_snapshot(self, run_id: str) -> str | None:
        v = self.r.get(self._run_key(run_id))
        return v.decode() if isinstance(v, bytes) else v

    # -- queue ---------------------------------------------------------
    def enqueue(self, run_id: str) -> None:
        self.r.lpush(self.QUEUE, run_id)

    def claim(self, timeout: int = 5) -> str | None:
        # BRPOPLPUSH: atomically move to processing so a crashed worker's
        # items can be reclaimed on restart.
        v = self.r.brpoplpush(self.QUEUE, self.PROCESSING, timeout=timeout)
        if v is None:
            return None
        return v.decode() if isinstance(v, bytes) else v

    def ack(self, run_id: str) -> None:
        self.r.lrem(self.PROCESSING, 1, run_id)

    def reclaim_orphans(self) -> list[str]:
        """Move items stuck in processing (crashed worker) back to the queue."""
        orphans = self.r.lrange(self.PROCESSING, 0, -1)
        out = []
        for o in orphans:
            rid = o.decode() if isinstance(o, bytes) else o
            self.r.lrem(self.PROCESSING, 1, rid)
            self.r.lpush(self.QUEUE, rid)
            out.append(rid)
        return out

    # -- trace ---------------------------------------------------------
    def append_trace(self, run_id: str, step_json: str) -> None:
        self.r.rpush(f"research:trace:{run_id}", step_json)

    def get_trace(self, run_id: str) -> list[str]:
        items = self.r.lrange(f"research:trace:{run_id}", 0, -1)
        return [i.decode() if isinstance(i, bytes) else i for i in items]

    # -- idempotency ---------------------------------------------------
    def idem_get(self, key: str) -> str | None:
        v = self.r.get(f"research:idem:{key}")
        return v.decode() if isinstance(v, bytes) else v

    def idem_set(self, key: str, value: str, ttl_sec: int = 86400) -> None:
        self.r.set(f"research:idem:{key}", value, ex=ttl_sec)

    # -- introspection -------------------------------------------------
    def running_run_ids(self) -> list[str]:
        out = []
        for k in self.r.scan_iter("research:run:run_*"):
            raw = self.r.get(k)
            if not raw:
                continue
            try:
                snap = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                if snap.get("status") == "RUNNING":
                    out.append(snap["run_id"])
            except Exception:
                continue
        return out

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False


class FileRunRegistry(RunRegistry):
    """Non-production fallback: JSON files under a directory.

    Used by the offline integration test when no Redis server is available.
    Same interface, no durability/queue-across-processes guarantees beyond
    the filesystem.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        (self.root / "trace").mkdir(parents=True, exist_ok=True)
        (self.root / "idem").mkdir(parents=True, exist_ok=True)
        self._queue_file = self.root / "queue.json"
        if not self._queue_file.exists():
            self._queue_file.write_text("[]")

    def _read_queue(self) -> list[str]:
        try:
            return json.loads(self._queue_file.read_text())
        except Exception:
            return []

    def create_run(self, run_id: str, question: str, snapshot_json: str) -> None:
        (self.root / "runs" / f"{run_id}.json").write_text(snapshot_json)

    def enqueue(self, run_id: str) -> None:
        q = self._read_queue()
        q.insert(0, run_id)
        self._queue_file.write_text(json.dumps(q))

    def claim(self, timeout: int = 5) -> str | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            q = self._read_queue()
            if q:
                rid = q.pop()
                self._queue_file.write_text(json.dumps(q))
                return rid
            time.sleep(0.2)
        return None

    def save_snapshot(self, run_id: str, snapshot_json: str) -> None:
        (self.root / "runs" / f"{run_id}.json").write_text(snapshot_json)

    def get_snapshot(self, run_id: str) -> str | None:
        p = self.root / "runs" / f"{run_id}.json"
        return p.read_text() if p.exists() else None

    def append_trace(self, run_id: str, step_json: str) -> None:
        p = self.root / "trace" / f"{run_id}.jsonl"
        with p.open("a") as f:
            f.write(step_json + "\n")

    def get_trace(self, run_id: str) -> list[str]:
        p = self.root / "trace" / f"{run_id}.jsonl"
        if not p.exists():
            return []
        return [ln for ln in p.read_text().splitlines() if ln.strip()]

    def idem_get(self, key: str) -> str | None:
        p = self.root / "idem" / f"{key}.json"
        return p.read_text() if p.exists() else None

    def idem_set(self, key: str, value: str, ttl_sec: int = 86400) -> None:
        (self.root / "idem" / f"{key}.json").write_text(value)

    def running_run_ids(self) -> list[str]:
        out = []
        for p in (self.root / "runs").glob("run_*.json"):
            try:
                snap = json.loads(p.read_text())
                if snap.get("status") == "RUNNING":
                    out.append(snap["run_id"])
            except Exception:
                continue
        return out

    def ping(self) -> bool:
        return self.root.exists()
