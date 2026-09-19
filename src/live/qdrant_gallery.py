
from __future__ import annotations

import hashlib
import uuid
from typing import Dict, Iterable, Mapping

import numpy as np
from qdrant_client import QdrantClient, models


class QdrantGallery:
    """Persistent multimodal vector retrieval for global person identities."""

    DIMS = {
        "resnet": 256,
        "swin": 1024,
        "solider": 1024,
        "attributes": 112,
        "face": 512,
        "pose": 51,
    }

    def __init__(
        self,
        path: str = "qdrant_storage",
        prefix: str = "person_reid",
        limit: int = 32,
        url: str | None = None,
        api_key: str | None = None,
    ):
        import os

        self.path = str(path)
        self.prefix = str(prefix)
        self.limit = max(4, int(limit))
        self.url = str(url or os.environ.get("QDRANT_URL", "")).strip()
        key = api_key or os.environ.get("QDRANT_API_KEY")
        if self.url:
            self.client = QdrantClient(
                url=self.url,
                api_key=key or None,
                timeout=30.0,
            )
        else:
            self.client = QdrantClient(path=self.path)
        self.collections: Dict[str, str] = {}

        for name, dim in self.DIMS.items():
            collection = f"{self.prefix}_{name}"
            self.collections[name] = collection
            try:
                self.client.get_collection(collection)
            except Exception:
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    ),
                )

    @staticmethod
    def _unit(value) -> np.ndarray | None:
        arr = np.asarray(value, np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if (
            arr.size == 0
            or not np.isfinite(norm)
            or norm <= 0.0
            or not np.isfinite(arr).all()
        ):
            return None
        return arr / norm

    @staticmethod
    def _point_id(
        gid: int,
        model_name: str,
        view: str,
        vector: np.ndarray,
    ) -> str:
        digest = hashlib.sha1(vector.tobytes()).hexdigest()
        return uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"person-reid:{gid}:{model_name}:{view}:{digest}",
        ).hex

    def _upsert(
        self,
        gid: int,
        model_name: str,
        view: str,
        values: Iterable[np.ndarray],
    ) -> None:
        points = []
        dim = self.DIMS[model_name]

        for value in values:
            vector = self._unit(value)
            if vector is None or vector.size != dim:
                continue
            points.append(
                models.PointStruct(
                    id=self._point_id(
                        gid,
                        model_name,
                        view,
                        vector,
                    ),
                    vector=vector.tolist(),
                    payload={
                        "gid": int(gid),
                        "model": model_name,
                        "view": view,
                    },
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collections[model_name],
                points=points,
                wait=True,
            )

    def upsert_component(self, gid: int, component) -> None:
        gid = int(gid)

        for group in component:
            banks = getattr(group, "state_bank", {}) or {}

            for model_name in (
                "resnet",
                "swin",
                "solider",
                "pose",
            ):
                views = banks.get(model_name, {}) or {}
                for view, values in views.items():
                    if model_name == "pose":
                        if view not in ("pose", "full"):
                            continue
                    elif view not in (
                        "full",
                        "upper",
                        "torso",
                        "lower",
                    ):
                        continue
                    self._upsert(
                        gid,
                        model_name,
                        view,
                        list(values)[-64:],
                    )

            attrs = list(
                getattr(group, "attribute_bank", []) or []
            )
            if not attrs:
                attrs = list(
                    banks.get("resnet", {})
                    .get("attributes", [])
                    or []
                )
            self._upsert(
                gid,
                "attributes",
                "attributes",
                attrs[-64:],
            )

            faces = list(
                getattr(group, "face_bank", []) or []
            )
            vectors = []
            for item in faces:
                if isinstance(item, Mapping):
                    if item.get("valid"):
                        vectors.append(item.get("vector"))
                elif item is not None:
                    vectors.append(item)
            self._upsert(
                gid,
                "face",
                "face",
                vectors[-32:],
            )

    def _query(self, model_name: str, vector):
        query = self._unit(vector)
        if query is None or query.size != self.DIMS[model_name]:
            return []

        result = self.client.query_points(
            collection_name=self.collections[model_name],
            query=query.tolist(),
            limit=self.limit,
            with_payload=True,
            with_vectors=True,
        )
        return list(
            getattr(result, "points", []) or []
        )

    def search_component(self, component):
        """Retrieve Qdrant candidates and their best matching exemplars."""
        candidates: Dict[int, Dict[str, list[float]]] = {}
        gallery: Dict[
            int,
            Dict[str, Dict[str, list[np.ndarray]]],
        ] = {}

        def add_hit(model_name, hit):
            payload = getattr(hit, "payload", {}) or {}
            gid = payload.get("gid")
            if gid is None:
                return

            gid = int(gid)
            view = str(payload.get("view", model_name))
            score = float(getattr(hit, "score", 0.0))
            candidates.setdefault(gid, {}).setdefault(
                model_name, []
            ).append(score)

            vector = getattr(hit, "vector", None)
            if vector is None:
                return

            arr = self._unit(vector)
            if arr is None:
                return

            gallery.setdefault(gid, {}).setdefault(
                model_name, {}
            ).setdefault(view, []).append(arr)

        for group in component:
            banks = getattr(group, "state_bank", {}) or {}

            for model_name in (
                "resnet",
                "swin",
                "solider",
            ):
                model_bank = banks.get(model_name, {}) or {}
                for view in (
                    "full",
                    "upper",
                    "torso",
                    "lower",
                ):
                    for value in model_bank.get(
                        view, []
                    )[-3:]:
                        for hit in self._query(
                            model_name,
                            value,
                        ):
                            add_hit(model_name, hit)

            pose_bank = banks.get("pose", {}) or {}
            for value in (
                pose_bank.get("pose", [])
                or pose_bank.get("full", [])
            )[-3:]:
                for hit in self._query("pose", value):
                    add_hit("pose", hit)

            attrs = list(
                getattr(group, "attribute_bank", []) or []
            )
            if not attrs:
                attrs = list(
                    banks.get("resnet", {})
                    .get("attributes", [])
                    or []
                )
            for value in attrs[-3:]:
                for hit in self._query(
                    "attributes",
                    value,
                ):
                    add_hit("attributes", hit)

            for item in list(
                getattr(group, "face_bank", []) or []
            )[-2:]:
                value = (
                    item.get("vector")
                    if isinstance(item, Mapping)
                    else item
                )
                if value is None:
                    continue
                for hit in self._query("face", value):
                    add_hit("face", hit)

        return candidates, gallery
