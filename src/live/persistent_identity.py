from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from live.identity_engine import IdentityEngine, ActiveIdentitySet, _unit


class PersistentIdentityRegistry:
    """Durable identity namespace and gallery for the live ReID engine.

    SQLite owns permanent GID allocation and identity evidence. The live
    ActiveIdentitySet remains a bounded execution cache; this registry is the
    durable source of truth and survives process restarts, empty scenes, tracker
    resets and active-set sweeps.
    """

    def __init__(self, path: str | os.PathLike[str], model_id: str = "live") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_id = str(model_id)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._init()
        self._validate_model()

    def _init(self) -> None:
        with self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS identities (
                    gid INTEGER PRIMARY KEY,
                    obs INTEGER NOT NULL,
                    last_ts REAL NOT NULL,
                    last_cam TEXT NOT NULL,
                    spans_json TEXT NOT NULL,
                    embedding_dim INTEGER
                )"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS embeddings (
                    gid INTEGER NOT NULL,
                    idx INTEGER NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (gid, idx),
                    FOREIGN KEY (gid) REFERENCES identities(gid) ON DELETE CASCADE
                )"""
            )
            self._db.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('next_gid', '1')"
            )
            self._db.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('model_id', ?)" , (self.model_id,)
            )

    def _validate_model(self) -> None:
        row = self._db.execute("SELECT value FROM meta WHERE key='model_id'").fetchone()
        stored = None if row is None else str(row[0])
        if stored and stored != self.model_id:
            raise RuntimeError(
                f"Persistent identity DB model mismatch: stored={stored!r}, current={self.model_id!r}. "
                "Use a separate identity DB for a different ReID model."
            )

    @property
    def next_gid(self) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key='next_gid'").fetchone()
        return int(row[0]) if row else 1

    def allocate_gid(self) -> int:
        """Atomically reserve the next permanent GID; IDs are never recycled."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT value FROM meta WHERE key='next_gid'").fetchone()
                value = int(row[0]) if row else 1
                self._db.execute(
                    "INSERT INTO meta(key, value) VALUES('next_gid', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(value + 1),),
                )
                self._db.commit()
                return value
            except BaseException:
                self._db.rollback()
                raise

    def set_dimension(self, dim: int) -> None:
        dim = int(dim)
        rows = self._db.execute(
            "SELECT DISTINCT embedding_dim FROM identities WHERE embedding_dim IS NOT NULL"
        ).fetchall()
        existing = {int(row[0]) for row in rows}
        if existing and existing != {dim}:
            raise RuntimeError(
                f"Persistent identity DB embedding dimension mismatch: stored={sorted(existing)}, current={dim}"
            )

    @staticmethod
    def _encode(vec: np.ndarray) -> tuple[int, bytes]:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("Cannot persist empty/non-finite identity embedding")
        return int(arr.size), arr.tobytes(order="C")

    @staticmethod
    def _decode(dim: int, blob: bytes) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32).copy()
        if arr.size != int(dim):
            raise RuntimeError(f"Corrupt persisted identity embedding: expected {dim}, got {arr.size}")
        return _unit(arr)

    def save_identity(self, store: ActiveIdentitySet, gid: int) -> None:
        gid = int(gid)
        if not store.has(gid):
            return
        bank = list(store._banks.get(gid, ()))
        last = store.last_seen(gid)
        if last is None:
            return
        last_cam, last_ts = last
        spans = store._spans.get(gid, {})
        obs = int(store.obs_count(gid))
        dim = int(bank[0].size) if bank else None
        if dim is not None:
            self.set_dimension(dim)
        spans_json = json.dumps(spans, separators=(",", ":"), sort_keys=True)
        with self._lock:
            with self._db:
                self._db.execute(
                    "INSERT INTO identities(gid, obs, last_ts, last_cam, spans_json, embedding_dim) "
                    "VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(gid) DO UPDATE SET obs=excluded.obs, last_ts=excluded.last_ts, "
                    "last_cam=excluded.last_cam, spans_json=excluded.spans_json, embedding_dim=excluded.embedding_dim",
                    (gid, obs, float(last_ts), str(last_cam), spans_json, dim),
                )
                self._db.execute("DELETE FROM embeddings WHERE gid=?", (gid,))
                for idx, vector in enumerate(bank):
                    edim, blob = self._encode(vector)
                    self._db.execute(
                        "INSERT INTO embeddings(gid, idx, dim, vector) VALUES(?,?,?,?)",
                        (gid, idx, edim, sqlite3.Binary(blob)),
                    )

    def save_all(self, store: ActiveIdentitySet) -> None:
        for gid in store.gids():
            self.save_identity(store, int(gid))

    def gids(self) -> list[int]:
        rows = self._db.execute("SELECT gid FROM identities ORDER BY gid").fetchall()
        return [int(row[0]) for row in rows]

    def hydrate(self, store: ActiveIdentitySet, gids: Optional[Iterable[int]] = None) -> int:
        ids = list(self.gids() if gids is None else [int(x) for x in gids])
        loaded = 0
        for gid in ids:
            if store.has(gid):
                continue
            row = self._db.execute(
                "SELECT obs, last_ts, last_cam, spans_json, embedding_dim FROM identities WHERE gid=?",
                (gid,),
            ).fetchone()
            if row is None:
                continue
            obs, last_ts, last_cam, spans_json, dim = row
            emb_rows = self._db.execute(
                "SELECT idx, dim, vector FROM embeddings WHERE gid=? ORDER BY idx",
                (gid,),
            ).fetchall()
            bank = [_unit(self._decode(int(edim), blob)) for _, edim, blob in emb_rows]
            bank = [x for x in bank if x is not None]
            store._banks[gid] = __import__("collections").deque(bank, maxlen=store.bank_size)
            store._spans[gid] = json.loads(spans_json) if spans_json else {}
            store._obs[gid] = int(obs)
            store._last_ts[gid] = float(last_ts)
            store._last_cam[gid] = str(last_cam)
            if dim is not None:
                self.set_dimension(int(dim))
            loaded += 1
        return loaded

    def close(self) -> None:
        with self._lock:
            self._db.close()


class PersistentIdentityEngine(IdentityEngine):
    """Known-good V6 IdentityEngine with durable GID/gallery lifecycle.

    The matching algorithm is inherited unchanged. This class only adds:
      - durable monotonic GID allocation;
      - persistent gallery hydration before a new-track match;
      - persistence of newly resolved/reinforced identities;
      - preservation of the permanent gallery when the active cache is swept.
    """

    def __init__(self, *args, state_path: str = "identity_state/reid_live.sqlite3",
                 model_id: str = "live", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.registry = PersistentIdentityRegistry(state_path, model_id=model_id)
        self._next_gid = self.registry.next_gid
        self.registry.hydrate(self.store)
        self.persistent_loaded = len(self.registry.gids())

    def _mint(self):
        gid = self.registry.allocate_gid()
        self._next_gid = self.registry.next_gid
        self.minted += 1
        return gid

    def _match(self, cam, agg, ts, exclude_key):
        # Make all durable identities available to the proven V6 matcher. The
        # active cache remains disposable; the registry is the permanent gallery.
        self.registry.hydrate(self.store)
        return super()._match(cam, agg, ts, exclude_key)

    def _resolve(self, cam, st, ts, key):
        super()._resolve(cam, st, ts, key)
        if st.get("status") == "assigned" and isinstance(st.get("gid"), int):
            self.registry.save_identity(self.store, st["gid"])

    def _reinforce(self, cam, st, unit, ts):
        super()._reinforce(cam, st, unit, ts)
        if st.get("status") == "assigned" and isinstance(st.get("gid"), int):
            self.registry.save_identity(self.store, st["gid"])

    def sweep(self, now):
        # Deliberately allow the proven active cache to evict cold identities for
        # bounded memory. Permanent identities remain in SQLite and are rehydrated
        # on the next identity search. Nothing is deleted from the registry here.
        return super().sweep(now)

    def close(self) -> None:
        self.registry.save_all(self.store)
        self.registry.close()
