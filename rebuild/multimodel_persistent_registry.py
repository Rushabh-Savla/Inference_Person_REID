from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


class MultiModelPersistentRegistry:
    """Durable global identity namespace and multimodel exemplar gallery.

    SQLite is authoritative for GID allocation and identity metadata. Each model
    space is stored independently; no embedding from one model is ever queried
    against another model's vectors.
    """

    def __init__(self, path: str | os.PathLike[str], model_id: str = "final-multimodel-v1") -> None:
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
            self._db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS identities (
                    gid INTEGER PRIMARY KEY,
                    last_ts REAL NOT NULL DEFAULT 0,
                    last_cam TEXT NOT NULL DEFAULT '',
                    cameras_json TEXT NOT NULL,
                    components INTEGER NOT NULL DEFAULT 0
                )"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS exemplars (
                    gid INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (gid, model, idx),
                    FOREIGN KEY (gid) REFERENCES identities(gid) ON DELETE CASCADE
                )"""
            )
            self._db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('next_gid','1')")
            self._db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('model_id',?)", (self.model_id,))

    def _validate_model(self) -> None:
        row = self._db.execute("SELECT value FROM meta WHERE key='model_id'").fetchone()
        if row is not None and str(row[0]) != self.model_id:
            raise RuntimeError(
                f"Identity DB model mismatch: stored={row[0]!r}, current={self.model_id!r}; use a new DB"
            )

    @property
    def next_gid(self) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key='next_gid'").fetchone()
        return int(row[0]) if row else 1

    def allocate(self) -> int:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                gid = self.next_gid
                self._db.execute(
                    "INSERT INTO meta(key,value) VALUES('next_gid',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(gid + 1),),
                )
                self._db.commit()
                return gid
            except BaseException:
                self._db.rollback()
                raise

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, np.float32).reshape(-1)
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("invalid identity exemplar")
        norm = np.linalg.norm(value)
        if norm <= 0:
            raise ValueError("zero identity exemplar")
        return value / (norm + 1e-12)

    def save(
        self,
        gid: int,
        models: Dict[str, Iterable[np.ndarray]],
        cameras: Iterable[str],
        timestamp: float = 0.0,
    ) -> None:
        gid = int(gid)
        clean = {}
        for name, values in models.items():
            clean[name] = [self._unit(x) for x in values if x is not None]
        cameras_json = json.dumps(sorted(set(str(x) for x in cameras)), separators=(",", ":"))
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO identities(gid,last_ts,last_cam,cameras_json,components) VALUES(?,?,?,?,?) "
                "ON CONFLICT(gid) DO UPDATE SET last_ts=excluded.last_ts,cameras_json=excluded.cameras_json,components=excluded.components",
                (gid, float(timestamp), sorted(set(str(x) for x in cameras))[-1] if cameras else "", sum(len(v) for v in clean.values())),
            )
            self._db.execute("DELETE FROM exemplars WHERE gid=?", (gid,))
            for model, values in clean.items():
                for idx, vector in enumerate(values[-32:]):
                    blob = vector.astype(np.float32).tobytes(order="C")
                    self._db.execute(
                        "INSERT INTO exemplars(gid,model,idx,dim,vector) VALUES(?,?,?,?,?)",
                        (gid, model, idx, int(vector.size), sqlite3.Binary(blob)),
                    )

    def gids(self) -> list[int]:
        return [int(x[0]) for x in self._db.execute("SELECT gid FROM identities ORDER BY gid").fetchall()]

    def gallery(self) -> Dict[int, Dict[str, list[np.ndarray]]]:
        result: Dict[int, Dict[str, list[np.ndarray]]] = {}
        rows = self._db.execute("SELECT gid,model,dim,vector FROM exemplars ORDER BY gid,model,idx").fetchall()
        for gid, model, dim, blob in rows:
            arr = np.frombuffer(blob, dtype=np.float32).copy()
            if arr.size != int(dim):
                raise RuntimeError(f"corrupt exemplar for {gid}/{model}: {arr.size} != {dim}")
            result.setdefault(int(gid), {}).setdefault(str(model), []).append(self._unit(arr))
        return result

    def close(self) -> None:
        with self._lock:
            self._db.close()
