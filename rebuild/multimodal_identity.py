
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from rebuild.face_v4 import FaceExtractorV4
from rebuild.identity_v2 import crop, quality
from rebuild.person_attributes import pack
from src.live.persistent_multimodel import PersistentMultimodelRegistry
from src.live.qdrant_gallery import QdrantGallery
from reid.nvidia_reid import NVIDIAReIDExtractor
from reid.nvidia_swin import NVIDIASwinReIDExtractor
from reid.solider_reid import SOLIDERReIDExtractor
from ultralytics import YOLO


class MultiModal:
    """Feature-first global identity resolver.

    Global IDs come only from multimodal evidence. Tracker IDs are temporal
    bookkeeping and are never used as identity keys after overlap.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reidcfg = cfg["reid"]
        self.mod = cfg["cross_camera_models"]
        self.idcfg = cfg["identity"]
        self.facecfg = cfg["face"]
        self.posecfg = cfg["pose"]

        self.resnet = NVIDIAReIDExtractor(
            weights=self.reidcfg["weights"],
            device=str(self.reidcfg.get("device", "cuda")),
            max_batch=int(self.reidcfg.get("max_batch", 32)),
        )
        self.swin = NVIDIASwinReIDExtractor(
            self.mod["swin_weights"],
            device="cuda",
            max_batch=int(self.mod.get("swin_batch", 16)),
        )
        self.solider = SOLIDERReIDExtractor(
            self.mod["solider_weights"],
            device="cuda",
            max_batch=int(self.mod.get("solider_batch", 16)),
        )
        self.face = None
        if bool(self.facecfg.get("enabled", True)):
            self.face = FaceExtractorV4(
                model=str(self.facecfg.get("model", "buffalo_l")),
                det_size=tuple(self.facecfg.get("det_size", [640, 640])),
                min_detection=float(self.facecfg.get("min_detection", 0.55)),
                min_size=int(self.facecfg.get("min_size", 32)),
                min_quality=float(self.facecfg.get("min_quality", 0.50)),
                min_visibility=float(self.facecfg.get("min_visibility", 0.68)),
                device=str(self.facecfg.get("device", "cuda")),
            )
        self.pose = None
        if bool(self.posecfg.get("enabled", True)):
            self.pose = YOLO(str(self.posecfg.get("model", "yolo11n-pose.pt")))
        state = cfg["identity_state"]
        self.registry = PersistentMultimodelRegistry(
            state["path"],
            model_id=str(state["model_id"]),
            bank_size=int(state.get("bank_size", 96)),
        )
        self.qdrant = QdrantGallery(
            str(state["qdrant_path"]),
            str(state.get("qdrant_prefix", "person_reid")),
            int(state.get("qdrant_limit", 32)),
        )
        self.profiles = self._load()
        self.trackmap = {}
        self.recovery = {}
        self.overlap = {}
        self.stats = {
            "frames": 0,
            "feature_frames": 0,
            "recovery_frames": 0,
            "recovery_matches": 0,
            "face_frames": 0,
            "face_matches": 0,
            "new_gids": 0,
            "duplicate_frames": 0,
            "pending_frames": 0,
            "cross_camera_matches": 0,
        }

    @staticmethod
    def _unit(value):
        arr = np.asarray(value, np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if arr.size == 0 or not np.isfinite(norm) or norm <= 0:
            return None
        return arr / norm

    @classmethod
    def _sim(cls, left, right):
        a = cls._unit(left)
        b = cls._unit(right)
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        return float(np.dot(a, b))

    @classmethod
    def _best(cls, left, right):
        vals = []
        for a in left or []:
            for b in right or []:
                value = cls._sim(a, b)
                if value:
                    vals.append(value)
        if not vals:
            return 0.0
        vals.sort(reverse=True)
        return float(np.mean(vals[: min(3, len(vals))]))

    @classmethod
    def _attrscores(cls, current, stored):
        q = np.asarray(current, np.float32).reshape(-1)
        bank = [
            np.asarray(x, np.float32).reshape(-1)
            for x in stored
        ]
        bank = [
            x for x in bank
            if x.size == 112 and np.isfinite(x).all()
        ]
        if q.size != 112 or not bank:
            return None
        top = cls._best([q[:20]], [x[:20] for x in bank])
        bottom = cls._best([q[20:40]], [x[20:40] for x in bank])
        uppat = cls._best([q[40:54]], [x[40:54] for x in bank])
        lowpat = cls._best([q[54:68]], [x[54:68] for x in bank])
        return {
            "top": float(top),
            "bottom": float(bottom),
            "upper_pattern": float(uppat),
            "lower_pattern": float(lowpat),
            "pattern": float(0.50 * uppat + 0.50 * lowpat),
        }

    @staticmethod
    def _pose(value):
        arr = np.asarray(value, np.float32).reshape(-1)
        return arr if arr.size == 51 and np.isfinite(arr).all() else None

    @staticmethod
    def _face(item):
        if not item:
            return None
        if not item.get("valid"):
            return None
        if float(item.get("visibility", 0.0)) < 0.68:
            return None
        if float(item.get("quality", 0.0)) < 0.50:
            return None
        vec = np.asarray(item.get("vector"), np.float32).reshape(-1)
        if vec.size != 512 or not np.isfinite(vec).all():
            return None
        return vec

    def _load(self):
        gallery = self.registry.load_gallery()
        profiles = {}
        for gid, banks in gallery.items():
            profiles[int(gid)] = {
                "resnet": list(banks.get("resnet", [])),
                "swin": list(banks.get("swin", [])),
                "solider": list(banks.get("solider", [])),
                "attributes": list(banks.get("attributes", [])),
                "face": list(banks.get("face", [])),
                "pose": list(banks.get("pose", [])),
                "camera": set(),
            }
        return profiles

    @classmethod
    def _poses(cls, model, frame):
        if model is None:
            return []
        result = model(frame, conf=0.20, iou=0.65, verbose=False)
        if not result or result[0].boxes is None:
            return []
        kp = getattr(result[0], "keypoints", None)
        if kp is None or getattr(kp, "data", None) is None:
            return []
        data = kp.data.cpu().numpy()
        out = []
        for index, item in enumerate(result[0].boxes):
            if index >= len(data):
                continue
            raw = np.asarray(data[index], np.float32)
            if raw.ndim != 2 or raw.shape[1] < 3 or len(raw) != 17:
                continue
            box = np.asarray(item.xyxy[0].tolist(), np.float32)
            out.append((box, raw))
        return out

    @classmethod
    def _posevec(cls, poses, box):
        if not poses:
            return None, 0.0
        target = np.asarray(box, np.float32)
        best = None
        score = 0.0
        for posebox, raw in poses:
            ix1 = max(target[0], posebox[0])
            iy1 = max(target[1], posebox[1])
            ix2 = min(target[2], posebox[2])
            iy2 = min(target[3], posebox[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            ta = max(1.0, (target[2] - target[0]) * (target[3] - target[1]))
            pa = max(1.0, (posebox[2] - posebox[0]) * (posebox[3] - posebox[1]))
            iou = inter / max(1.0, ta + pa - inter)
            if iou <= score:
                continue
            width = max(1.0, target[2] - target[0])
            height = max(1.0, target[3] - target[1])
            vec = raw.copy()
            vec[:, 0] = (vec[:, 0] - target[0]) / width
            vec[:, 1] = (vec[:, 1] - target[1]) / height
            vec[:, 2] = np.clip(vec[:, 2], 0.0, 1.0)
            vec = vec.reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm <= 0 or not np.isfinite(norm):
                continue
            best = vec / norm
            score = float(iou)
        if best is None:
            return None, 0.0
        return best, float(score)

    def _probe(self, gid):
        profile = self.profiles.get(int(gid), {})
        return SimpleNamespace(
            state_bank={
                "resnet": {"full": profile.get("resnet", [])[-64:]},
                "swin": {"full": profile.get("swin", [])[-64:]},
                "solider": {"full": profile.get("solider", [])[-64:]},
                "pose": {"pose": profile.get("pose", [])[-64:]},
            },
            attribute_bank=profile.get("attributes", [])[-64:],
            face_bank=[
                {
                    "vector": x,
                    "valid": True,
                    "quality": 1.0,
                    "visibility": 1.0,
                }
                for x in profile.get("face", [])[-32:]
            ],
        )

    def _candidates(self, obs):
        group = SimpleNamespace(
            state_bank={
                "resnet": {"full": [obs["resnet"]]},
                "swin": {"full": [obs["swin"]]},
                "solider": {"full": [obs["solider"]]},
                "pose": {"pose": [obs["pose"]]}
                if obs.get("pose") is not None else {},
            },
            attribute_bank=[obs["attributes"]],
            face_bank=(
                [
                    {
                        "vector": obs["face"]["vector"],
                        "valid": True,
                        "quality": obs["face"]["quality"],
                        "visibility": obs["face"]["visibility"],
                    }
                ]
                if obs.get("face") else []
            ),
        )
        hits, got = self.qdrant.search_component([group])
        ids = set(int(x) for x in hits)
        ids.update(int(x) for x in self.profiles)
        return ids, got

    def _score(self, obs, gid, got):
        profile = self.profiles.get(int(gid), {})
        remote = got.get(int(gid), {})
        values = {}
        for model in ("resnet", "swin", "solider"):
            remotevals = []
            for view in remote.get(model, {}).values():
                remotevals.extend(view)
            localvals = profile.get(model, [])
            values[model] = self._best(
                [obs[model]],
                remotevals or localvals,
            )

        deep = (
            0.30 * values["resnet"]
            + 0.37 * values["swin"]
            + 0.33 * values["solider"]
        )

        stored_attrs = (
            remote.get("attributes", {}).get("attributes", [])
            or profile.get("attributes", [])
        )
        attr = self._attrscores(
            obs["attributes"],
            stored_attrs,
        )
        if attr is None:
            return None

        top = attr["top"]
        bottom = attr["bottom"]
        uppat = attr["upper_pattern"]
        lowpat = attr["lower_pattern"]
        pattern = attr["pattern"]

        posevals = (
            remote.get("pose", {}).get("pose", [])
            or profile.get("pose", [])
        )
        pose = (
            self._best([obs["pose"]], posevals)
            if obs.get("pose") is not None and posevals
            else 0.0
        )

        face = 0.0
        faceused = False
        if obs.get("face"):
            faces = (
                remote.get("face", {}).get("face", [])
                or profile.get("face", [])
            )
            face = self._best(
                [obs["face"]["vector"]],
                faces,
            )
            faceused = bool(faces)

        if faceused:
            total = (
                0.55 * face
                + 0.20 * deep
                + 0.10 * top
                + 0.10 * bottom
                + 0.05 * pose
            )
        else:
            total = (
                0.45 * deep
                + 0.20 * top
                + 0.20 * bottom
                + 0.10 * pose
                + 0.05 * pattern
            )

        contradiction = bool(
            top >= 0.70 and bottom < 0.34
        )
        if contradiction and not (
            faceused and face >= 0.88
        ):
            total -= 0.12

        return {
            "score": float(np.clip(total, 0.0, 0.99)),
            "deep": float(deep),
            "resnet": float(values["resnet"]),
            "swin": float(values["swin"]),
            "solider": float(values["solider"]),
            "top": float(top),
            "bottom": float(bottom),
            "upper_pattern": float(uppat),
            "lower_pattern": float(lowpat),
            "pose": float(pose),
            "face": float(face),
            "faceused": faceused,
        }

    def _accept(self, row, second, recovery=False):
        margin = float(row["score"] - second)
        if row["faceused"] and row["face"] >= 0.86:
            return bool(
                row["score"] >= 0.72
                and margin >= 0.025
            )
        if recovery:
            return bool(
                row["score"] >= float(
                    self.idcfg["recovery_min"]
                )
                and margin >= float(
                    self.idcfg["recovery_margin"]
                )
                and row["deep"] >= 0.52
                and row["top"] >= 0.48
                and row["bottom"] >= 0.46
                and sum(
                    row[x] >= 0.50
                    for x in ("resnet", "swin", "solider")
                ) >= 2
            )
        return bool(
            row["score"] >= float(
                self.idcfg["existing_min"]
            )
            and margin >= float(self.idcfg["margin"])
            and row["deep"] >= 0.54
            and row["top"] >= 0.48
            and row["bottom"] >= 0.46
            and sum(
                row[x] >= 0.50
                for x in ("resnet", "swin", "solider")
            ) >= 2
        )

    def _save(self, gid, obs):
        profile = self.profiles[int(gid)]
        for model in ("resnet", "swin", "solider"):
            profile[model].append(
                np.asarray(obs[model], np.float32)
            )
            profile[model] = profile[model][-96:]
        profile["attributes"].append(
            np.asarray(obs["attributes"], np.float32)
        )
        profile["attributes"] = profile["attributes"][-64:]
        if obs.get("face"):
            profile["face"].append(
                np.asarray(
                    obs["face"]["vector"],
                    np.float32,
                )
            )
            profile["face"] = profile["face"][-32:]
        if obs.get("pose") is not None:
            profile["pose"].append(
                np.asarray(obs["pose"], np.float32)
            )
            profile["pose"] = profile["pose"][-64:]
        profile["camera"].add(str(obs["camera"]))

        group = SimpleNamespace(
            key=f"G{int(gid):06d}",
            members=[f"G{int(gid):06d}"],
            state_bank={
                "resnet": {"full": profile["resnet"][-8:]},
                "swin": {"full": profile["swin"][-8:]},
                "solider": {"full": profile["solider"][-8:]},
                "pose": {"pose": profile["pose"][-8:]},
            },
            attribute_bank=profile["attributes"][-8:],
            face_bank=[
                {
                    "vector": x,
                    "valid": True,
                    "quality": 1.0,
                    "visibility": 1.0,
                }
                for x in profile["face"][-8:]
            ],
        )
        self.qdrant.upsert_component(
            int(gid),
            [group],
        )
        banks = {
            "resnet": profile["resnet"][-32:],
            "swin": profile["swin"][-32:],
            "solider": profile["solider"][-32:],
            "attributes": profile["attributes"][-32:],
            "face": profile["face"][-16:],
            "pose": profile["pose"][-32:],
        }
        self.registry.save_component(
            int(gid),
            model_banks=banks,
            cameras=profile["camera"],
            last_ts=float(obs["time"]),
            obs=len(profile["resnet"]),
        )

    def _new(self, obs):
        gid = int(self.registry.allocate_gid())
        self.profiles[gid] = {
            "resnet": [],
            "swin": [],
            "solider": [],
            "attributes": [],
            "face": [],
            "pose": [],
            "camera": set(),
        }
        self._save(gid, obs)
        self.stats["new_gids"] += 1
        return gid

    def observe(self, frame, rows, commit=True):
        """Extract mandatory features and solve a one-to-one GID assignment."""
        poses = self._poses(self.pose, frame)
        base = []
        images = []
        for item in rows:
            person = crop(frame, item["bbox"])
            q = float(quality(person)) if person is not None else 0.0
            if (
                person is None
                or q < float(
                    self.reidcfg.get("min_quality", 0.20)
                )
            ):
                continue
            base.append((item, person, q))
            images.append(person)
        if not base:
            return {}
        self.stats["feature_frames"] += 1

        resnet = self.resnet.extract_batch(images)
        swin = self.swin.extract_batch(images)
        solider = self.solider.extract_batch(images)
        if not (
            len(resnet)
            == len(base)
            == len(swin)
            == len(solider)
        ):
            raise RuntimeError(
                "Multimodal extractors returned mismatched batch sizes"
            )

        observations = []
        for index, (item, person, q) in enumerate(base):
            attrs = pack(
                person,
                frame,
                item["bbox"],
            )
            face = None
            if self.face is not None:
                faceobj = self.face.extract(
                    frame,
                    item["bbox"],
                )
                if faceobj is not None and faceobj.valid:
                    face = {
                        "vector": faceobj.vector,
                        "quality": float(faceobj.quality),
                        "visibility": float(
                            faceobj.visibility
                        ),
                    }
                    self.stats["face_frames"] += 1
            pose, posescore = self._posevec(
                poses,
                item["bbox"],
            )
            observations.append(
                {
                    "row": item,
                    "camera": item["camera"],
                    "time": float(item["timestamp"]),
                    "resnet": np.asarray(
                        resnet[index],
                        np.float32,
                    ),
                    "swin": np.asarray(
                        swin[index],
                        np.float32,
                    ),
                    "solider": np.asarray(
                        solider[index],
                        np.float32,
                    ),
                    "attributes": np.asarray(
                        attrs,
                        np.float32,
                    ),
                    "face": face,
                    "pose": pose,
                    "pose_score": float(posescore),
                }
            )

        gids = sorted(self.profiles)
        matrix = np.full(
            (
                len(observations),
                len(gids) + len(observations),
            ),
            float(self.idcfg["new_floor"]),
            np.float32,
        )
        scoreset = []
        gets = []
        for index, obs in enumerate(observations):
            ids, got = self._candidates(obs)
            gets.append(got)
            scores = {}
            for gid in ids:
                value = self._score(obs, int(gid), got)
                if value is not None:
                    scores[int(gid)] = value
            scoreset.append(scores)
            for col, gid in enumerate(gids):
                if gid in scores:
                    matrix[index, col] = scores[gid]["score"]

        rr, cc = linear_sum_assignment(-matrix)
        chosen = {}
        for row, col in zip(rr, cc):
            if row >= len(observations):
                continue
            obs = observations[row]
            scores = scoreset[row]
            ranked = sorted(
                scores.items(),
                key=lambda x: x[1]["score"],
                reverse=True,
            )
            best = ranked[0][1] if ranked else None
            second = (
                ranked[1][1]["score"]
                if len(ranked) > 1
                else 0.0
            )
            key = (
                str(obs["row"]["camera"]),
                int(obs["row"]["track_id"]),
            )
            recovery = bool(
                self.recovery.get(key, False)
            )
            selected = None
            if col < len(gids):
                gid = gids[col]
                if gid in scores and self._accept(
                    scores[gid],
                    second,
                    recovery,
                ):
                    selected = int(gid)

            if selected is None:
                # Never create a new identity during an overlap recovery
                # window. A failed recovery remains pending until a later
                # feature observation can match an established profile.
                if recovery:
                    self.stats["pending_frames"] += 1
                    continue
                if best is not None and best["score"] >= 0.55:
                    self.stats["pending_frames"] += 1
                    continue
                if not commit:
                    self.stats["pending_frames"] += 1
                    continue
                selected = self._new(obs)

            chosen[row] = selected

        # This is intentionally redundant with Hungarian assignment. It is a
        # hard invariant: at most one person row can leave this function with
        # a given GID.
        seen = set()
        for row in list(chosen):
            gid = chosen[row]
            if gid in seen:
                chosen.pop(row)
            else:
                seen.add(gid)

        for row, gid in chosen.items():
            obs = observations[row]
            key = (
                str(obs["row"]["camera"]),
                int(obs["row"]["track_id"]),
            )
            if commit:
                self.trackmap[key] = gid
                self._save(gid, obs)
                if bool(self.recovery.get(key, False)):
                    self.stats["recovery_matches"] += 1
                if obs.get("face") is not None:
                    self.stats["face_matches"] += 1
                if len(self.profiles[gid]["camera"]) > 1:
                    self.stats["cross_camera_matches"] += 1

        return {
            int(observations[row]["row"]["track_id"]): f"G{int(gid):06d}"
            for row, gid in chosen.items()
        }

    def close(self):
        self.registry.close()
''