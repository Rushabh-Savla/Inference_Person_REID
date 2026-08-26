from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np


class PersistentMultimodelRegistry:
    """Durable GID namespace and multimodel exemplar gallery.

    SQLite is the identity authority. It stores the permanent GID namespace and
    compact exemplar banks for each embedding space. No active-track state is
    persisted here, so tracker resets and scene-empty periods cannot recycle IDs.
    """

    def __init__(self, path: str | os.PathLike[str], model_id: str = "multimodel-v1", bank_size: int = 32) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_id = str(model_id)
        self.bank_size = max(4, int(bank_size))
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._init()
        self._validate_model()

    def _init(self) -> None:
        with self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS identities (
                    gid INTEGER PRIMARY KEY,
                    obs INTEGER NOT NULL DEFAULT 0,
                    last_ts REAL NOT NULL DEFAULT 0,
                    last_cam TEXT NOT NULL DEFAULT '',
                    spans_json TEXT NOT NULL DEFAULT '{}'
                )"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS embeddings (
                    gid INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (gid, model, idx),
                    FOREIGN KEY (gid) REFERENCES identities(gid) ON DELETE CASCADE
                )"""
            )
            self._db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('next_gid', '1')")
            self._db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('model_id', ?)", (self.model_id,))

    def _validate_model(self) -> None:
        row = self._db.execute("SELECT value FROM meta WHERE key='model_id'").fetchone()
        if row is not None and str(row[0]) != self.model_id:
            raise RuntimeError(
                f"Persistent multimodel DB model mismatch: stored={row[0]!r}, current={self.model_id!r}. "
                "Use a new DB for a different identity model stack."
            )

    @staticmethod
    def _unit(value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if arr.size == 0 or not np.isfinite(norm) or norm <= 0:
            raise ValueError("invalid identity embedding")
        return arr / norm

    @staticmethod
    def _encode(value: np.ndarray) -> tuple[int, bytes]:
        arr = PersistentMultimodelRegistry._unit(value)
        return int(arr.size), arr.tobytes(order="C")

    @staticmethod
    def _decode(dim: int, blob: bytes) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32).copy()
        if arr.size != int(dim):
            raise RuntimeError(f"corrupt persisted embedding: expected {dim}, got {arr.size}")
        return PersistentMultimodelRegistry._unit(arr)

    @property
    def next_gid(self) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key='next_gid'").fetchone()
        return int(row[0]) if row else 1

    def allocate_gid(self) -> int:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                gid = self.next_gid
                self._db.execute("UPDATE meta SET value=? WHERE key='next_gid'", (str(gid + 1),))
                self._db.commit()
                return gid
            except BaseException:
                self._db.rollback()
                raise

    def gids(self) -> list[int]:
        rows = self._db.execute("SELECT gid FROM identities ORDER BY gid").fetchall()
        return [int(x[0]) for x in rows]

    def save_component(
        self,
        gid: int,
        *,
        model_banks: Mapping[str, Iterable[np.ndarray]],
        cameras: Iterable[str],
        last_ts: float,
        obs: int,
    ) -> None:
        gid = int(gid)
        cams = sorted(set(str(x) for x in cameras))
        with self._lock:
            with self._db:
                self._db.execute(
                    "INSERT INTO identities(gid, obs, last_ts, last_cam, spans_json) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(gid) DO UPDATE SET obs=excluded.obs, last_ts=excluded.last_ts, "
                    "last_cam=excluded.last_cam, spans_json=excluded.spans_json",
                    (gid, int(obs), float(last_ts), cams[-1] if cams else "", json.dumps(cams)),
                )
                for model, values in model_banks.items():
                    vectors = []
                    seen = set()
                    for value in values:
                        arr = self._unit(value)
                        key = arr.tobytes()
                        if key in seen:
                            continue
                        seen.add(key)
                        vectors.append(arr)
                    vectors = vectors[-self.bank_size :]
                    self._db.execute("DELETE FROM embeddings WHERE gid=? AND model=?", (gid, str(model)))
                    for idx, vector in enumerate(vectors):
                        dim, blob = self._encode(vector)
                        self._db.execute(
                            "INSERT INTO embeddings(gid, model, idx, dim, vector) VALUES(?,?,?,?,?)",
                            (gid, str(model), idx, dim, sqlite3.Binary(blob)),
                        )

    def load_gallery(self) -> Dict[int, Dict[str, list[np.ndarray]]]:
        rows = self._db.execute("SELECT gid, model, idx, dim, vector FROM embeddings ORDER BY gid, model, idx").fetchall()
        result: Dict[int, Dict[str, list[np.ndarray]]] = {}
        for gid, model, _idx, dim, blob in rows:
            result.setdefault(int(gid), {}).setdefault(str(model), []).append(self._decode(int(dim), blob))
        return result

    def close(self) -> None:
        with self._lock:
            self._db.close()
