"""LangGraph checkpointers.

RedisCheckpointer is the production backend: every node completion persists
the graph checkpoint to Redis, so a killed worker resumes from the last
completed node via thread_id. FileCheckpointer implements the same contract
on disk for environments without Redis (offline tests only).
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any, Iterator, Sequence

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.base import SerializerProtocol


def _ids(config: dict) -> tuple[str, str, str | None]:
    c = config.get("configurable", {})
    return c["thread_id"], c.get("checkpoint_ns", ""), c.get("checkpoint_id")


class _SerdeMixin:
    """dumps_typed/loads_typed helpers (langgraph>=0.6 serializer API)."""

    def _dump(self, obj) -> bytes:  # type: ignore[no-untyped-def]
        t, data = self.serde.dumps_typed(obj)  # type: ignore[attr-defined]
        return t.encode() + b"\x00" + data

    def _load(self, raw: bytes):  # type: ignore[no-untyped-def]
        t, data = raw.split(b"\x00", 1)
        return self.serde.loads_typed((t.decode(), data))  # type: ignore[attr-defined]


class RedisCheckpointer(_SerdeMixin, BaseCheckpointSaver):
    """Durable LangGraph checkpointer backed by Redis (sync client)."""

    def __init__(self, redis_client, serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self.r = redis_client  # decode_responses=False

    # -- key layout ------------------------------------------------------
    def _data(self, tid: str, ns: str, cid: str) -> bytes:
        return f"ckpt:data:{tid}:{ns}:{cid}".encode()

    def _idx(self, tid: str, ns: str) -> bytes:
        return f"ckpt:idx:{tid}:{ns}".encode()

    def _meta(self, tid: str, ns: str, cid: str) -> bytes:
        return f"ckpt:meta:{tid}:{ns}:{cid}".encode()

    def _write_prefix(self, tid: str, ns: str, cid: str) -> bytes:
        return f"ckpt:w:{tid}:{ns}:{cid}:".encode()

    # -- reads -------------------------------------------------------------
    def get_tuple(self, config: dict) -> CheckpointTuple | None:
        tid, ns, cid = _ids(config)
        if cid is None:
            raw_cid = self.r.get(self._idx(tid, ns))
            if raw_cid is None:
                return None
            cid = raw_cid.decode()
        raw = self.r.get(self._data(tid, ns, cid))
        if raw is None:
            return None
        checkpoint: Checkpoint = self._load(raw)
        raw_meta = self.r.get(self._meta(tid, ns, cid))
        metadata: CheckpointMetadata = self._load(raw_meta) if raw_meta else {}
        pending: list[tuple[str, str, Any]] = []
        for k in self.r.scan_iter(self._write_prefix(tid, ns, cid) + b"*"):
            # key: ckpt:w:{tid}:{ns}:{cid}:{task_id}:{seq}
            # A concurrent put() may delete the key between scan and get;
            # that only happens once the checkpoint is committed, so skip.
            raw_w = self.r.get(k)
            if raw_w is None:
                continue
            tail = k.decode().rsplit(":", 2)
            task_id = tail[-2]
            channel, value = self._load(raw_w)
            pending.append((task_id, channel, value))
        parent_cid = checkpoint.get("id")
        parent_config = (
            {"configurable": {"thread_id": tid, "checkpoint_ns": ns,
                              "checkpoint_id": parent_cid}}
            if parent_cid and parent_cid != cid else None
        )
        return CheckpointTuple(
            config={"configurable": {"thread_id": tid, "checkpoint_ns": ns,
                                     "checkpoint_id": cid}},
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending,
        )

    # -- writes ------------------------------------------------------------
    def put(self, config: dict, checkpoint: Checkpoint,
            metadata: CheckpointMetadata, new_versions: dict) -> dict:
        tid, ns, _ = _ids(config)
        cid: str = checkpoint["id"]
        pipe = self.r.pipeline()
        pipe.set(self._data(tid, ns, cid), self._dump(checkpoint))
        pipe.set(self._meta(tid, ns, cid), self._dump(metadata))
        pipe.set(self._idx(tid, ns), cid)
        # Drop pending writes for this thread+ns. LangGraph may use a
        # different checkpoint id for put_writes than for the committed
        # checkpoint (e.g. after a killed superstep), so scope the cleanup
        # to the thread, not just the committed cid. Orphaned writes are
        # never read (get_tuple scopes by cid), this just reclaims space.
        prefix = f"ckpt:w:{tid}:{ns}:".encode()
        for k in self.r.scan_iter(prefix + b"*"):
            pipe.delete(k)
        pipe.execute()
        return {"configurable": {"thread_id": tid, "checkpoint_ns": ns,
                                 "checkpoint_id": cid}}

    def put_writes(self, config: dict, writes: Sequence[tuple[str, Any]],
                   task_id: str, task_path: str = "") -> None:
        tid, ns, cid = _ids(config)
        if cid is None:
            return
        pipe = self.r.pipeline()
        for i, (channel, value) in enumerate(writes):
            k = self._write_prefix(tid, ns, cid) + f"{task_id}:{i}".encode()
            pipe.set(k, self._dump((channel, value)))
        pipe.execute()

    def list(self, config: dict | None, *, filter: dict | None = None,
             before: dict | None = None,
             limit: int | None = None) -> Iterator[CheckpointTuple]:
        if config is None:
            return
        tid, ns, _ = _ids(config)
        count = 0
        for k in self.r.scan_iter(f"ckpt:data:{tid}:{ns}:".encode() + b"*"):
            cid = k.decode().rsplit(":", 1)[-1]
            tup = self.get_tuple({"configurable": {"thread_id": tid,
                                                   "checkpoint_ns": ns,
                                                   "checkpoint_id": cid}})
            if tup is None:
                continue
            if filter and not all(tup.metadata.get(kk) == vv
                                  for kk, vv in filter.items()):
                continue
            yield tup
            count += 1
            if limit is not None and count >= limit:
                break


class FileCheckpointer(_SerdeMixin, BaseCheckpointSaver):
    """Same contract on local disk. Offline-test fallback only."""

    def __init__(self, root: str | Path, serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, tid: str, ns: str) -> Path:
        d = self.root / tid / (ns or "default")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Write via temp file + rename so concurrent readers never see a
        partially-written file. The temp name is unique per call so two
        threads writing the same destination don't clobber each other
        (os.replace serializes; last writer wins with complete content)."""
        tmp = path.with_name(
            f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}-"
            f"{uuid.uuid4().hex[:8]}")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get_tuple(self, config: dict) -> CheckpointTuple | None:
        tid, ns, cid = _ids(config)
        d = self._dir(tid, ns)
        if cid is None:
            idx = d / "LATEST"
            if not idx.exists():
                return None
            cid = idx.read_text().strip()
        data = d / f"{cid}.bin"
        if not data.exists():
            return None
        checkpoint: Checkpoint = self._load(data.read_bytes())
        meta_p = d / f"{cid}.meta"
        metadata: CheckpointMetadata = (
            self._load(meta_p.read_bytes()) if meta_p.exists() else {})
        pending: list[tuple[str, str, Any]] = []
        for w in sorted(d.glob(f"{cid}.w.*")):
            if ".tmp-" in w.name:
                continue  # leftover from a crashed writer; never complete
            # A concurrent put() may unlink the file between glob and read;
            # that only happens once the checkpoint is committed, so skip.
            try:
                raw_w = w.read_bytes()
            except FileNotFoundError:
                continue
            task_id = w.name.rsplit(".", 2)[-2]
            channel, value = self._load(raw_w)
            pending.append((task_id, channel, value))
        return CheckpointTuple(
            config={"configurable": {"thread_id": tid, "checkpoint_ns": ns,
                                     "checkpoint_id": cid}},
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=None,
            pending_writes=pending,
        )

    def put(self, config: dict, checkpoint: Checkpoint,
            metadata: CheckpointMetadata, new_versions: dict) -> dict:
        tid, ns, _ = _ids(config)
        d = self._dir(tid, ns)
        cid: str = checkpoint["id"]
        self._atomic_write(d / f"{cid}.bin", self._dump(checkpoint))
        self._atomic_write(d / f"{cid}.meta", self._dump(metadata))
        self._atomic_write(d / "LATEST", cid.encode())
        for w in d.glob(f"{cid}.w.*"):
            if ".tmp-" in w.name:
                continue  # in-flight atomic write; not ours to delete
            w.unlink()
        return {"configurable": {"thread_id": tid, "checkpoint_ns": ns,
                                 "checkpoint_id": cid}}

    def put_writes(self, config: dict, writes: Sequence[tuple[str, Any]],
                   task_id: str, task_path: str = "") -> None:
        tid, ns, cid = _ids(config)
        if cid is None:
            return
        d = self._dir(tid, ns)
        for i, (channel, value) in enumerate(writes):
            self._atomic_write(d / f"{cid}.w.{task_id}.{i}",
                               self._dump((channel, value)))

    def list(self, config: dict | None, *, filter: dict | None = None,
             before: dict | None = None,
             limit: int | None = None) -> Iterator[CheckpointTuple]:
        if config is None:
            return
        tid, ns, _ = _ids(config)
        d = self._dir(tid, ns)
        count = 0
        for bin_p in sorted(d.glob("*.bin")):
            cid = bin_p.stem
            tup = self.get_tuple({"configurable": {"thread_id": tid,
                                                   "checkpoint_ns": ns,
                                                   "checkpoint_id": cid}})
            if tup is None:
                continue
            if filter and not all(tup.metadata.get(kk) == vv
                                  for kk, vv in filter.items()):
                continue
            yield tup
            count += 1
            if limit is not None and count >= limit:
                break
