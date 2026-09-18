from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

from rebuild.face_v4 import FaceExtractorV4
from rebuild.identity_v2 import crop, quality
from rebuild.person_attributes import pack
from reid.nvidia_reid import NVIDIAReIDExtractor
from reid.nvidia_swin import NVIDIASwinReIDExtractor
from reid.solider_reid import SOLIDERReIDExtractor
from src.live.persistent_multimodel import PersistentMultimodelRegistry
from src.live.qdrant_gallery import QdrantGallery
from ultralytics import YOLO


class MultiModal:
    """Strict multimodal identity resolver for the NVIDIA NvDCF pipeline."""

    MODELS = ("resnet", "swin", "solider")

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

        self.face_threshold = float(self.facecfg.get("min_visibility", 0.68))
        self.face_quality = float(self.facecfg.get("min_quality", 0.50))
        if not bool(self.facecfg.get("enabled", True)):
            raise RuntimeError("Face matching is mandatory for the strict NvDCF ReID pipeline")
        self.face = FaceExtractorV4(
            model=str(self.facecfg.get("model", "buffalo_l")),
            det_size=tuple(self.facecfg.get("det_size", [640, 640])),
            min_detection=float(self.facecfg.get("min_detection", 0.55)),
            min_size=int(self.facecfg.get("min_size", 32)),
            min_quality=self.face_quality,
            min_visibility=self.face_threshold,
            device=str(self.facecfg.get("device", "cuda")),
        )

        if not bool(self.posecfg.get("enabled", True)):
            raise RuntimeError("Pose matching is mandatory for the strict NvDCF ReID pipeline")
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
        self.pending = []
        self.pending_min = max(2, int(self.idcfg.get("new_confirm_frames", 4)))
        self.pending_match = float(self.idcfg.get("new_pending_match_min", 0.78))
        self.pending_min_model = float(self.idcfg.get("new_pending_model_min", 0.50))
        self.pending_min_clothing = float(self.idcfg.get("new_pending_clothing_min", 0.55))
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
            "pending_new_observations": 0,
            "pending_new_confirmed": 0,
        }

    @staticmethod
    def _unit(value):
        arr = np.asarray(value, np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if arr.size == 0 or not np.isfinite(norm) or norm <= 0.0:
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
        values = []
        for a in left or []:
            for b in right or []:
                values.append(cls._sim(a, b))
        if not values:
            return 0.0
        values.sort(reverse=True)
        return float(np.mean(values[: min(3, len(values))]))

    @classmethod
    def _attrscores(cls, current, stored):
        query = np.asarray(current, np.float32).reshape(-1)
        bank = [np.asarray(value, np.float32).reshape(-1) for value in stored or []]
        bank = [value for value in bank if value.size == 112 and np.isfinite(value).all()]
        if query.size != 112 or not bank:
            return None
        top = cls._best([query[:20]], [value[:20] for value in bank])
        bottom = cls._best([query[20:40]], [value[20:40] for value in bank])
        upper_pattern = cls._best([query[40:54]], [value[40:54] for value in bank])
        lower_pattern = cls._best([query[54:68]], [value[54:68] for value in bank])
        return {
            "top": float(top),
            "bottom": float(bottom),
            "upper_pattern": float(upper_pattern),
            "lower_pattern": float(lower_pattern),
            "pattern": float(0.50 * upper_pattern + 0.50 * lower_pattern),
        }

    @classmethod
    def _face_score(cls, current, stored):
        if not current:
            return {"used": False, "score": 0.0}
        vector = cls._unit(current.get("vector"))
        if vector is None:
            return {"used": False, "score": 0.0}
        values = []
        for value in stored or []:
            if isinstance(value, dict):
                if not value.get("valid", True):
                    continue
                value = value.get("vector")
            other = cls._unit(value)
            if other is not None and other.shape == vector.shape:
                values.append(float(np.dot(vector, other)))
        if not values:
            return {"used": False, "score": 0.0}
        values.sort(reverse=True)
        return {"used": True, "score": float(np.mean(values[: min(3, len(values))]))}

    @classmethod
    def _pose_vector(cls, value):
        if value is None:
            return None
        arr = np.asarray(value, np.float32).reshape(-1)
        if arr.size != 51 or not np.isfinite(arr).all():
            return None
        return cls._unit(arr)

    @classmethod
    def _poses(cls, model, frame):
        result = model(frame, classes=[0], conf=0.20, iou=0.65, verbose=False)
        if not result or result[0].boxes is None:
            return []
        keypoints = getattr(result[0], "keypoints", None)
        if keypoints is None or getattr(keypoints, "data", None) is None:
            return []
        data = keypoints.data.cpu().numpy()
        output = []
        for index, box in enumerate(result[0].boxes):
            if index >= len(data):
                continue
            raw = np.asarray(data[index], np.float32)
            if raw.shape != (17, 3) or not np.isfinite(raw).all():
                continue
            bbox = np.asarray(box.xyxy[0].tolist(), np.float32)
            output.append((bbox, raw))
        return output

    @classmethod
    def _posevec(cls, poses, box):
        target = np.asarray(box, np.float32)
        best = None
        best_iou = 0.0
        for pose_box, raw in poses:
            ix1 = max(target[0], pose_box[0])
            iy1 = max(target[1], pose_box[1])
            ix2 = min(target[2], pose_box[2])
            iy2 = min(target[3], pose_box[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            ta = max(1.0, (target[2] - target[0]) * (target[3] - target[1]))
            pa = max(1.0, (pose_box[2] - pose_box[0]) * (pose_box[3] - pose_box[1]))
            iou = inter / max(1.0, ta + pa - inter)
            if iou <= best_iou:
                continue
            width = max(1.0, target[2] - target[0])
            height = max(1.0, target[3] - target[1])
            vector = raw.copy()
            vector[:, 0] = (raw[:, 0] - target[0]) / width
            vector[:, 1] = (raw[:, 1] - target[1]) / height
            vector[:, 2] = np.clip(raw[:, 2], 0.0, 1.0)
            vector = cls._pose_vector(vector)
            if vector is None:
                continue
            best = vector
            best_iou = float(iou)
        return best, float(best_iou)

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

    def _candidates(self, obs):
        group = SimpleNamespace(
            state_bank={
                "resnet": {"full": [obs["resnet"]]},
                "swin": {"full": [obs["swin"]]},
                "solider": {"full": [obs["solider"]]},
                "pose": {"pose": [obs["pose"]]} if obs.get("pose") is not None else {},
            },
            attribute_bank=[obs["attributes"]],
            face_bank=[obs["face"]] if obs.get("face") else [],
        )
        hits, retrieved = self.qdrant.search_component([group])
        ids = {int(gid) for gid in self.profiles}
        ids.update(int(gid) for gid in hits)
        return ids, retrieved

    def _score(self, obs, gid, retrieved):
        profile = self.profiles.get(int(gid), {})
        remote = retrieved.get(int(gid), {})

        values = {}
        for model in self.MODELS:
            remote_values = []
            for view in remote.get(model, {}).values():
                remote_values.extend(view)
            values[model] = self._best([obs[model]], remote_values or profile.get(model, []))

        deep = 0.30 * values["resnet"] + 0.37 * values["swin"] + 0.33 * values["solider"]

        stored_attrs = remote.get("attributes", {}).get("attributes", []) or profile.get("attributes", [])
        attrs = self._attrscores(obs["attributes"], stored_attrs)
        if attrs is None:
            return None

        pose_values = remote.get("pose", {}).get("pose", []) or profile.get("pose", [])
        pose = self._best([obs["pose"]], pose_values) if obs.get("pose") is not None and pose_values else 0.0

        face_visible = obs.get("face") is not None
        face = self._face_score(
            obs.get("face"),
            remote.get("face", {}).get("face", []) or profile.get("face", []),
        )

        top = attrs["top"]
        bottom = attrs["bottom"]
        pattern = attrs["pattern"]

        # A reliable detected face is the strongest cue. A low face similarity
        # remains a strong negative signal; it never silently falls back to
        # clothing/ReID for a candidate when a face gallery exists.
        if face_visible and face["used"]:
            score = (
                0.62 * face["score"]
                + 0.17 * deep
                + 0.09 * top
                + 0.09 * bottom
                + 0.03 * pose
            )
            faceused = True
        else:
            score = (
                0.46 * deep
                + 0.22 * top
                + 0.22 * bottom
                + 0.07 * pose
                + 0.03 * pattern
            )
            faceused = False

        lower_conflict = bool(top >= 0.70 and bottom < 0.34)
        if lower_conflict and not (faceused and face["score"] >= 0.88):
            score -= 0.12

        return {
            "score": float(np.clip(score, 0.0, 0.995)),
            "deep": float(deep),
            "resnet": float(values["resnet"]),
            "swin": float(values["swin"]),
            "solider": float(values["solider"]),
            "top": float(top),
            "bottom": float(bottom),
            "upper_pattern": float(attrs["upper_pattern"]),
            "lower_pattern": float(attrs["lower_pattern"]),
            "pose": float(pose),
            "face": float(face["score"]),
            "faceused": faceused,
        }

    def _accept(self, row, second, recovery=False):
        margin = float(row["score"] - second)
        models = sum(row[name] >= 0.48 for name in self.MODELS)

        if row["faceused"]:
            return bool(
                row["face"] >= self.face_threshold
                and row["score"] >= float(self.idcfg.get("face_existing_min", 0.70))
                and margin >= float(self.idcfg.get("face_margin", 0.015))
                and row["top"] >= 0.40
                and row["bottom"] >= 0.40
            )

        if recovery:
            return bool(
                row["score"] >= float(self.idcfg.get("recovery_min", 0.60))
                and margin >= float(self.idcfg.get("recovery_margin", 0.012))
                and row["deep"] >= 0.48
                and row["top"] >= 0.45
                and row["bottom"] >= 0.45
                and models >= 2
            )

        return bool(
            row["score"] >= float(self.idcfg.get("existing_min", 0.62))
            and margin >= float(self.idcfg.get("margin", 0.015))
            and row["deep"] >= 0.48
            and row["top"] >= 0.45
            and row["bottom"] >= 0.45
            and models >= 2
        )

    def _save(self, gid, obs):
        profile = self.profiles[int(gid)]
        for model in self.MODELS:
            profile[model].append(np.asarray(obs[model], np.float32))
            profile[model] = profile[model][-96:]
        profile["attributes"].append(np.asarray(obs["attributes"], np.float32))
        profile["attributes"] = profile["attributes"][-64:]
        if obs.get("face"):
            profile["face"].append(np.asarray(obs["face"]["vector"], np.float32))
            profile["face"] = profile["face"][-32:]
        if obs.get("pose") is not None:
            profile["pose"].append(np.asarray(obs["pose"], np.float32))
            profile["pose"] = profile["pose"][-64:]
        profile["camera"].add(str(obs["camera"]))

        group = SimpleNamespace(
            key=f"G{int(gid):06d}",
            members=[f"G{int(gid):06d}"],
            state_bank={
                "resnet": {"full": profile["resnet"][-12:]},
                "swin": {"full": profile["swin"][-12:]},
                "solider": {"full": profile["solider"][-12:]},
                "pose": {"pose": profile["pose"][-12:]},
            },
            attribute_bank=profile["attributes"][-12:],
            face_bank=[
                {"vector": value, "valid": True, "quality": 1.0, "visibility": 1.0}
                for value in profile["face"][-12:]
            ],
        )
        self.qdrant.upsert_component(int(gid), [group])
        self.registry.save_component(
            int(gid),
            model_banks={
                "resnet": profile["resnet"][-48:],
                "swin": profile["swin"][-48:],
                "solider": profile["solider"][-48:],
                "attributes": profile["attributes"][-48:],
                "face": profile["face"][-24:],
                "pose": profile["pose"][-48:],
            },
            cameras=profile["camera"],
            last_ts=float(obs["time"]),
            obs=len(profile["resnet"]),
        )

    def _pending_score(self, obs, pending):
        values = {
            model: self._best([obs[model]], pending.get(model, []))
            for model in self.MODELS
        }
        deep = (
            0.30 * values["resnet"]
            + 0.37 * values["swin"]
            + 0.33 * values["solider"]
        )
        attrs = self._attrscores(
            obs["attributes"],
            pending.get("attributes", []),
        )
        if attrs is None:
            return None
        top = float(attrs["top"])
        bottom = float(attrs["bottom"])
        pose = (
            self._best([obs["pose"]], pending.get("pose", []))
            if obs.get("pose") is not None and pending.get("pose")
            else 0.0
        )
        face = self._face_score(
            obs.get("face"),
            pending.get("face", []),
        )
        if face.get("used"):
            score = (
                0.64 * float(face["score"])
                + 0.18 * deep
                + 0.09 * top
                + 0.08 * bottom
                + 0.01 * pose
            )
        else:
            score = (
                0.46 * deep
                + 0.26 * top
                + 0.26 * bottom
                + 0.02 * pose
            )
        models = sum(
            values[name] >= self.pending_min_model
            for name in self.MODELS
        )
        return {
            "score": float(np.clip(score, 0.0, 0.995)),
            "resnet": float(values["resnet"]),
            "swin": float(values["swin"]),
            "solider": float(values["solider"]),
            "top": top,
            "bottom": bottom,
            "pose": float(pose),
            "face": float(face.get("score", 0.0)),
            "faceused": bool(face.get("used")),
            "models": int(models),
        }

    def _stage_new(self, obs):
        best = None
        best_index = None
        for index, item in enumerate(self.pending):
            score = self._pending_score(obs, item)
            if score is None:
                continue
            if (
                score["top"] < self.pending_min_clothing
                or score["bottom"] < self.pending_min_clothing
            ):
                continue
            if score["models"] < 2:
                continue
            if best is None or score["score"] > best["score"]:
                best = score
                best_index = index

        if best is not None and best["score"] >= self.pending_match:
            item = self.pending[best_index]
            item["resnet"].append(np.asarray(obs["resnet"], np.float32))
            item["swin"].append(np.asarray(obs["swin"], np.float32))
            item["solider"].append(np.asarray(obs["solider"], np.float32))
            item["attributes"].append(np.asarray(obs["attributes"], np.float32))
            if obs.get("pose") is not None:
                item["pose"].append(np.asarray(obs["pose"], np.float32))
            if obs.get("face"):
                item["face"].append(dict(obs["face"]))
            item["observations"].append(obs)
            item["count"] += 1
            item["last_camera"] = str(obs["camera"])
            self.stats["pending_new_observations"] += 1
            if item["count"] >= self.pending_min:
                seed = item["observations"][0]
                gid = self._new(seed)
                for extra in item["observations"][1:]:
                    self._save(gid, extra)
                self.pending.pop(best_index)
                self.stats["pending_new_confirmed"] += 1
                return gid
            return None

        self.pending.append({
            "resnet": [np.asarray(obs["resnet"], np.float32)],
            "swin": [np.asarray(obs["swin"], np.float32)],
            "solider": [np.asarray(obs["solider"], np.float32)],
            "attributes": [np.asarray(obs["attributes"], np.float32)],
            "pose": (
                [np.asarray(obs["pose"], np.float32)]
                if obs.get("pose") is not None
                else []
            ),
            "face": [dict(obs["face"])] if obs.get("face") else [],
            "observations": [obs],
            "count": 1,
            "last_camera": str(obs["camera"]),
        })
        self.stats["pending_new_observations"] += 1
        return None

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

    def assign_rows(self, observations, scoresets, commit=True, recovery=False):
        """Strict one-to-one assignment for a single frame."""
        count = len(observations)
        gids = sorted(self.profiles)
        if count == 0:
            return {}

        floor = float(self.idcfg.get("new_floor", 0.50))
        matrix = np.full((count, len(gids) + count), floor, np.float32)
        for row, scores in enumerate(scoresets):
            for column, gid in enumerate(gids):
                if gid in scores:
                    matrix[row, column] = float(scores[gid]["score"])

        _, columns = linear_sum_assignment(-matrix)
        chosen = {}
        for row in range(count):
            column = int(columns[row]) if row < len(columns) else len(gids) + row
            scores = scoresets[row]
            ranked = sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)
            best_score = float(ranked[0][1]["score"]) if ranked else 0.0
            second = float(ranked[1][1]["score"]) if len(ranked) > 1 else 0.0

            selected = None
            if column < len(gids):
                gid = gids[column]
                candidate = scores.get(gid)
                if candidate is not None and self._accept(candidate, second, recovery=recovery):
                    selected = int(gid)

            if selected is None:
                if recovery or not commit:
                    self.stats["pending_frames"] += 1
                    continue
                known_floor = float(self.idcfg.get("known_floor", 0.58))
                if best_score >= known_floor:
                    self.stats["pending_frames"] += 1
                    continue
                staged = self._stage_new(observations[row])
                if staged is None:
                    self.stats["pending_frames"] += 1
                    continue
                selected = int(staged)

            chosen[row] = selected

        used = set()
        for row in list(chosen):
            gid = chosen[row]
            if gid in used:
                chosen.pop(row)
                self.stats["duplicate_frames"] += 1
                continue
            used.add(gid)

        for row, gid in chosen.items():
            obs = observations[row]
            if not commit:
                continue
            # Tracker IDs are observation bookkeeping only. They never
            # participate in Global ID recovery or selection.
            self._save(gid, obs)
            if recovery:
                self.stats["recovery_matches"] += 1
            if obs.get("face"):
                self.stats["face_matches"] += 1
            if len(self.profiles[gid]["camera"]) > 1:
                self.stats["cross_camera_matches"] += 1

        return {
            int(observations[row]["row"]["track_id"]): f"G{int(gid):06d}"
            for row, gid in chosen.items()
        }

    def observe(self, frame, rows, commit=True, recovery=False):
        """Extract face + top/bottom clothing + ReID + pose and solve one-to-one."""
        poses = self._poses(self.pose, frame)
        observations = []
        images = []
        for item in rows:
            person = crop(frame, item["bbox"])
            if person is None or person.size == 0:
                continue
            images.append(person)
            observations.append({
                "row": item,
                "camera": str(item["camera"]),
                "time": float(item["timestamp"]),
                "crop_quality": float(quality(person)),
                "person": person,
            })

        if not observations: