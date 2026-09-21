from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
from rebuild.assignment_guard import solve
from rebuild.identity_v2 import crop, quality



class MultiModalStrict:
    """Feature-first identity resolver for the real NvDCF pipeline."""

    models = ("resnet", "swin", "solider")

    def __init__(self, cfg):
        from rebuild.face_v4 import FaceExtractorV4
        from rebuild.person_attributes import pack
        from reid.nvidia_reid import NVIDIAReIDExtractor
        from reid.nvidia_swin import NVIDIASwinReIDExtractor
        from reid.solider_reid import SOLIDERReIDExtractor
        from src.live.persistent_multimodel import PersistentMultimodelRegistry
        from src.live.qdrant_gallery import QdrantGallery
        from ultralytics import YOLO
        self.pack = pack
        self.cfg = cfg
        self.rcfg = cfg["reid"]
        self.mcfg = cfg["cross_camera_models"]
        self.icfg = cfg["identity"]
        self.fcfg = cfg["face"]
        self.pcfg = cfg["pose"]
        if not bool(self.fcfg.get("enabled", True)):
            raise RuntimeError("Face matching is mandatory in the strict identity runtime")
        self.face = FaceExtractorV4(
            model=str(self.fcfg.get("model", "buffalo_l")),
            det_size=tuple(self.fcfg.get("det_size", [640, 640])),
            min_detection=float(self.fcfg.get("min_detection", 0.55)),
            min_size=int(self.fcfg.get("min_size", 28)),
            min_quality=float(self.fcfg.get("min_quality", 0.50)),
            min_visibility=float(self.fcfg.get("min_visibility", 0.68)),
            device="cuda",
        )
        self.resnet = NVIDIAReIDExtractor(
            weights=self.rcfg["weights"],
            device=str(self.rcfg.get("device", "cuda")),
            max_batch=int(self.rcfg.get("max_batch", 32)),
        )
        self.swin = NVIDIASwinReIDExtractor(
            self.mcfg["swin_weights"],
            device="cuda",
            max_batch=int(self.mcfg.get("swin_batch", 16)),
        )
        self.solider = SOLIDERReIDExtractor(
            self.mcfg["solider_weights"],
            device="cuda",
            max_batch=int(self.mcfg.get("solider_batch", 16)),
        )
        if not bool(self.pcfg.get("enabled", True)):
            raise RuntimeError("Pose matching is mandatory")
        self.pose = YOLO(str(self.pcfg.get("model", "yolo11n-pose.pt")))
        state = cfg["identity_state"]
        self.reg = PersistentMultimodelRegistry(
            state["path"],
            model_id=str(state["model_id"]),
            bank_size=int(state.get("bank_size", 96)),
        )
        qdrant_url = os.environ.get("QDRANT_URL") or state.get("qdrant_url")
        qdrant_key = os.environ.get("QDRANT_API_KEY") or state.get("qdrant_api_key")
        self.q = QdrantGallery(
            str(state["qdrant_path"]),
            str(state.get("qdrant_prefix", "person_reid")),
            int(state.get("qdrant_limit", 32)),
            url=qdrant_url,
            api_key=qdrant_key,
        )
        self.pro = self.load()
        self.pending = []
        self.pending_min = max(3, int(self.icfg.get("new_confirm_frames", 4)))
        self.pending_match = float(self.icfg.get("new_pending_match_min", 0.82))
        self.pending_margin = float(self.icfg.get("new_pending_margin", 0.08))
        self.pending_model_min = float(self.icfg.get("new_pending_model_min", 0.55))
        self.pending_clothing_min = float(self.icfg.get("new_pending_clothing_min", 0.58))
        self.pending_required_models = 3
        self.pending_ttl = int(self.icfg.get("pending_ttl_seconds", 300))
        self.pending_max = int(self.icfg.get("pending_max", 64))
        self.pending_time_gap = float(self.icfg.get("pending_time_gap", 0.75))
        self.last_evidence = {}
        self.stats = {
            "feature": 0,
            "recovery": 0,
            "recovery_match": 0,
            "new": 0,
            "pending": 0,
            "cross": 0,
            "duplicate": 0,
            "pending_new_observations": 0,
            "pending_new_confirmed": 0,
            "face_observations": 0,
            "face_reliable": 0,
            "qdrant_retrievals": 0,
            "memory_reject": 0,
            "recovery_feature_verified": 0,
        }

    @staticmethod
    def unit(value):
        arr = np.asarray(value, np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if arr.size == 0 or not np.isfinite(norm) or norm <= 0.0:
            return None
        return arr / norm

    @classmethod
    def sim(cls, left, right):
        a = cls.unit(left)
        b = cls.unit(right)
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        return float(np.dot(a, b))

    @classmethod
    def best(cls, left, right):
        vals = []
        for a in left or []:
            for b in right or []:
                value = cls.sim(a, b)
                if np.isfinite(value):
                    vals.append(value)
        if not vals:
            return 0.0
        vals.sort(reverse=True)
        return float(np.mean(vals[: min(3, len(vals))]))

    @classmethod
    def attr(cls, cur, old):
        query = np.asarray(cur, np.float32).reshape(-1)
        bank = [np.asarray(x, np.float32).reshape(-1) for x in old or []]
        bank = [x for x in bank if x.size == 112 and np.isfinite(x).all()]
        if query.size != 112 or not bank:
            return None

        # Couple upper and lower clothing to the same stored exemplar. The old
        # implementation maximized top and bottom independently, which could
        # combine the shirt of one gallery view with the trousers of another.
        pairs = []
        for item in bank:
            upper = cls.sim(query[0:20], item[0:20])
            lower = cls.sim(query[20:40], item[20:40])
            upper_pattern = cls.sim(query[40:54], item[40:54])
            lower_pattern = cls.sim(query[54:68], item[54:68])
            joint = min(float(upper), float(lower))
            pairs.append(
                (
                    joint,
                    0.50 * float(upper) + 0.50 * float(lower),
                    float(upper),
                    float(lower),
                    float(upper_pattern),
                    float(lower_pattern),
                )
            )

        pairs.sort(key=lambda value: (value[0], value[1]), reverse=True)
        joint, clothing, upper, lower, upper_pattern, lower_pattern = pairs[0]
        return {
            "top": float(upper),
            "bottom": float(lower),
            "joint": float(joint),
            "clothing": float(clothing),
            "upper_pattern": float(upper_pattern),
            "lower_pattern": float(lower_pattern),
            "pattern": 0.50 * float(upper_pattern) + 0.50 * float(lower_pattern),
            "upper_visible": bool(query[108] > 0.0),
            "lower_visible": bool(query[109] > 0.0),
        }

    @classmethod
    def faceval(cls, cur, old):
        if not cur or not bool(cur.get("valid", False)):
            return {"used": False, "score": 0.0}
        vec = cls.unit(cur.get("vector"))
        if vec is None:
            return {"used": False, "score": 0.0}
        vals = []
        for item in old or []:
            if isinstance(item, dict):
                if not item.get("valid", True):
                    continue
                item = item.get("vector")
            other = cls.unit(item)
            if other is not None and other.shape == vec.shape:
                vals.append(float(np.dot(vec, other)))
        if not vals:
            return {"used": False, "score": 0.0}
        vals.sort(reverse=True)
        return {"used": True, "score": float(np.mean(vals[: min(3, len(vals))]))}

    @classmethod
    def poses(cls, model, frame):
        result = model(frame, classes=[0], conf=0.20, iou=0.65, verbose=False)
        if not result or result[0].boxes is None:
            return []
        kp = getattr(result[0], "keypoints", None)
        if kp is None or getattr(kp, "data", None) is None:
            return []
        data = kp.data.cpu().numpy()
        out = []
        for i, box in enumerate(result[0].boxes):
            if i >= len(data):
                continue
            raw = np.asarray(data[i], np.float32)
            if raw.shape != (17, 3) or not np.isfinite(raw).all():
                continue
            b = np.asarray(box.xyxy[0].tolist(), np.float32)
            out.append((b, raw))
        return out

    @classmethod
    def poseval(cls, poses, box):
        target = np.asarray(box, np.float32)
        best = None
        score = 0.0
        for pbox, raw in poses:
            x1 = max(target[0], pbox[0])
            y1 = max(target[1], pbox[1])
            x2 = min(target[2], pbox[2])
            y2 = min(target[3], pbox[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            ta = max(1.0, (target[2] - target[0]) * (target[3] - target[1]))
            pa = max(1.0, (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]))
            iou = inter / max(1.0, ta + pa - inter)
            if iou <= score:
                continue
            w = max(1.0, target[2] - target[0])
            h = max(1.0, target[3] - target[1])
            vec = raw.copy()
            vec[:, 0] = (raw[:, 0] - target[0]) / w
            vec[:, 1] = (raw[:, 1] - target[1]) / h
            vec[:, 2] = np.clip(raw[:, 2], 0.0, 1.0)
            vec = cls.unit(vec)
            if vec is None or vec.size != 51:
                continue
            best = vec
            score = float(iou)
        return best, float(score)

    def load(self):
        data = self.reg.load_gallery()
        meta = {}
        if hasattr(self.reg, "identity_meta"):
            try:
                meta = self.reg.identity_meta()
            except Exception:
                meta = {}
        out = {}
        for gid, bank in data.items():
            info = meta.get(int(gid), {})
            out[int(gid)] = {
                "resnet": list(bank.get("resnet", [])),
                "swin": list(bank.get("swin", [])),
                "solider": list(bank.get("solider", [])),
                "attributes": list(bank.get("attributes", [])),
                "face": list(bank.get("face", [])),
                "pose": list(bank.get("pose", [])),
                "camera": set(info.get("cameras", [])),
            }
        return out

    def group(self, obs):
        return SimpleNamespace(
            state_bank={
                "resnet": {"full": [obs["resnet"]]},
                "swin": {"full": [obs["swin"]]},
                "solider": {"full": [obs["solider"]]},
                "pose": {"pose": [obs["pose"]]} if obs.get("pose") is not None else {},
            },
            attribute_bank=[obs["attributes"]],
            face_bank=(
                [{"vector": obs["face"]["vector"], "valid": True}]
                if obs.get("face") is not None
                else []
            ),
        )

    def cand(self, obs):
        hits, got = self.q.search_component([self.group(obs)])
        if hits:
            self.stats["qdrant_retrievals"] += 1

        # Qdrant already returns the cross-camera gallery candidates. Adding
        # every persistent GID here defeated retrieval and turned every frame
        # into an O(persons-in-gallery) comparison. Keep only retrieved IDs
        # plus deterministic same-track/overlap anchors.
        reserved = {
            int(x)
            for x in (obs.get("reserved_gids") or [])
            if str(x).lstrip("-").isdigit()
        }
        ids = {
            int(x)
            for x in hits
            if int(x) not in reserved
        }
        track_hint = obs.get("track_hint")
        if track_hint is not None:
            try:
                gid = int(track_hint)
                if gid in self.pro and gid not in reserved:
                    ids.add(gid)
            except (TypeError, ValueError):
                pass
        for value in obs.get("recovery_hints", []) or []:
            try:
                gid = int(value)
            except (TypeError, ValueError):
                continue
            if gid in self.pro and gid not in reserved:
                ids.add(gid)
        return sorted(ids), got

    def score(self, obs, gid, got):
        prof = self.pro.get(int(gid), {})
        rem = got.get(int(gid), {})
        vals = {}
        for model in self.models:
            src = []
            for view in rem.get(model, {}).values():
                src.extend(view)
            vals[model] = self.best([obs[model]], src or prof.get(model, []))
        crop_quality = float(np.clip(obs.get("quality", 1.0), 0.0, 1.0))
        qweight = 0.75 + 0.25 * crop_quality
        deep = (
            0.25 * vals["resnet"]
            + 0.40 * vals["swin"]
            + 0.35 * vals["solider"]
        ) * qweight
        attrs = self.attr(
            obs["attributes"],
            rem.get("attributes", {}).get("attributes", []) or prof.get("attributes", []),
        )
        if attrs is None:
            return None

        top_floor = float(self.icfg.get("top_min", 0.52))
        bottom_floor = float(self.icfg.get("bottom_min", 0.52))
        joint_floor = float(self.icfg.get("clothing_joint_min", min(top_floor, bottom_floor)))
        if (
            float(attrs["top"]) < top_floor
            or float(attrs["bottom"]) < bottom_floor
            or float(attrs["joint"]) < joint_floor
        ):
            self.stats["clothing_reject"] = self.stats.get("clothing_reject", 0) + 1
            return None
        pose = 0.0
        pbank = rem.get("pose", {}).get("pose", []) or prof.get("pose", [])
        if obs.get("pose") is not None and pbank:
            pose = self.best([obs["pose"]], pbank)
        top = float(attrs["top"]) * qweight
        bot = float(attrs["bottom"]) * qweight
        pattern = float(attrs["pattern"]) * qweight
        facebank = rem.get("face", {}).get("face", []) or prof.get("face", [])
        face = self.faceval(obs.get("face"), facebank)
        faceq = (
            float(np.clip(obs["face"]["quality"], 0.0, 1.0))
            if obs.get("face") is not None else 0.0
        )
        face_visibility = (
            float(obs["face"].get("visibility", 0.0))
            if obs.get("face") is not None else 0.0
        )
        reliable_face = bool(
            obs.get("face") is not None
            and obs["face"].get("valid", False)
            and faceq >= float(self.icfg.get("face_quality_min", 0.50))
            and face_visibility >= float(self.fcfg.get("min_visibility", 0.68))
        )
        if reliable_face and facebank:
            if not face.get("used") or float(face.get("score", 0.0)) < float(
                self.icfg.get("face_min", 0.60)
            ):
                return None
        if reliable_face and face.get("used"):
            face_conf = float(face["score"]) * (0.75 + 0.25 * faceq)
            value = (
                0.78 * face_conf
                + 0.12 * float(deep)
                + 0.04 * float(top)
                + 0.04 * float(bot)
                + 0.015 * float(pose)
                + 0.005 * float(pattern)
            )
        else:
            value = (
                0.54 * float(deep)
                + 0.20 * float(top)
                + 0.20 * float(bot)
                + 0.04 * float(pose)
                + 0.02 * float(pattern)
            )
        hints = {
            int(x) for x in (obs.get("recovery_hints", []) or [])
            if str(x).lstrip("-").isdigit()
        }
        return {
            "score": float(np.clip(value, 0.0, 0.995)),
            "deep": float(deep),
            "resnet": float(vals["resnet"]),
            "swin": float(vals["swin"]),
            "solider": float(vals["solider"]),
            "top": float(top),
            "bottom": float(bot),
            "clothing_joint": float(attrs["joint"]),
            "upper_pattern": float(attrs["upper_pattern"]),
            "lower_pattern": float(attrs["lower_pattern"]),
            "pose": float(pose),
            "face": float(face["score"]),
            "face_used": bool(face.get("used")),
            "face_reliable": bool(reliable_face and face.get("used")),
            "face_quality": float(faceq),
            "face_visibility": float(face_visibility),
            "quality": crop_quality,
            "recovery_hint": bool(int(gid) in hints),
            "recovery_hints": sorted(hints),
        }

    def accept(self, row, second, recovery=False):
        if row is None:
            return False
        margin = float(row["score"] - second)
        support = sum(
            float(row[name]) >= float(self.icfg.get("model_min", 0.46))
            for name in self.models
        )
        if float(row["top"]) < float(self.icfg.get("top_min", 0.52)):
            return False
        if float(row["bottom"]) < float(self.icfg.get("bottom_min", 0.52)):
            return False
        if float(row.get("clothing_joint", min(row["top"], row["bottom"]))) < float(
            self.icfg.get("clothing_joint_min", 0.52)
        ):
            return False
        if support < 2:
            return False
        if float(row.get("quality", 0.0)) < float(
            self.icfg.get("memory_quality_min", 0.45)
        ):
            return False
        if recovery and row.get("recovery_hints"):
            if not bool(row.get("recovery_hint")):
                if (
                    float(row["score"]) < 0.82
                    or support < 3
                    or float(row.get("deep", 0.0)) < 0.58
                ):
                    return False
        floor = float(
            self.icfg.get("recovery_min", 0.62)
            if recovery
            else self.icfg.get("existing_min", 0.62)
        )
        gap = float(
            self.icfg.get("recovery_margin", 0.025)
            if recovery
            else self.icfg.get("margin", 0.025)
        )
        if bool(row.get("face_reliable")):
            if (
                float(row.get("face", 0.0)) < float(self.icfg.get("face_min", 0.60))
                or float(row.get("face_quality", 0.0))
                < float(self.icfg.get("face_quality_min", 0.50))
            ):
                return False
        return bool(
            row["score"] >= floor
            and row["deep"] >= float(self.icfg.get("deep_min", 0.48))
            and margin >= gap
        )

    def _memory_ok(self, gid, obs):
        prof = self.pro.get(int(gid))
        if not prof or not prof.get("resnet"):
            return True

        checks = {}
        for name in self.models:
            checks[name] = self.best(
                [obs[name]],
                prof.get(name, [])[-24:],
            )

        attrs = self.attr(
            obs["attributes"],
            prof.get("attributes", [])[-24:],
        )
        if attrs is None:
            return False

        deep = (
            0.25 * checks["resnet"]
            + 0.40 * checks["swin"]
            + 0.35 * checks["solider"]
        )
        top = float(attrs["top"])
        bottom = float(attrs["bottom"])

        # Long-term memory updates are stricter than frame assignment. This
        # protects a stable identity from a single contaminated/occluded crop.
        required = int(self.icfg.get("memory_required_models", 2))
        support = sum(
            float(checks[name]) >= float(self.icfg.get("memory_model_min", 0.52))
            for name in self.models
        )
        if support < required:
            return False
        if deep < float(self.icfg.get("memory_deep_min", 0.56)):
            return False
        if top < float(self.icfg.get("memory_top_min", 0.50)):
            return False
        if bottom < float(self.icfg.get("memory_bottom_min", 0.50)):
            return False
        if min(top, bottom) < float(self.icfg.get("clothing_joint_min", 0.52)):
            return False

        current_face = obs.get("face")
        stored_face = prof.get("face", [])
        if current_face is not None and current_face.get("valid") and stored_face:
            face_score = self.faceval(current_face, stored_face)["score"]
            if face_score < float(self.icfg.get("memory_face_min", 0.65)):
                return False
        return True

    def save(self, gid, obs, force=False):
        gid = int(gid)
        if not force and not self._memory_ok(gid, obs):
            self.stats["memory_reject"] = self.stats.get("memory_reject", 0) + 1
            return

        prof = self.pro[gid]
        for model in self.models:
            prof[model].append(np.asarray(obs[model], np.float32))
            prof[model] = prof[model][-96:]
        prof["attributes"].append(np.asarray(obs["attributes"], np.float32))
        prof["attributes"] = prof["attributes"][-64:]
        if obs.get("pose") is not None:
            prof["pose"].append(np.asarray(obs["pose"], np.float32))
            prof["pose"] = prof["pose"][-64:]
        if obs.get("face") is not None and obs["face"].get("valid"):
            prof["face"].append(np.asarray(obs["face"]["vector"], np.float32))
            prof["face"] = prof["face"][-48:]
        prof["camera"].add(str(obs["camera"]))
        node = SimpleNamespace(
            key=f"G{gid:06d}",
            members=[f"G{gid:06d}"],
            state_bank={
                "resnet": {"full": prof["resnet"][-16:]},
                "swin": {"full": prof["swin"][-16:]},
                "solider": {"full": prof["solider"][-16:]},
                "pose": {"pose": prof["pose"][-16:]},
            },
            attribute_bank=prof["attributes"][-16:],
            face_bank=[
                {"vector": value, "valid": True}
                for value in prof["face"][-16:]
            ],
        )
        self.q.upsert_component(gid, [node])
        self.reg.save_component(
            gid,
            model_banks={
                "resnet": prof["resnet"][-64:],
                "swin": prof["swin"][-64:],
                "solider": prof["solider"][-64:],
                "attributes": prof["attributes"][-64:],
                "face": prof["face"][-48:],
                "pose": prof["pose"][-64:],
            },
            cameras=prof["camera"],
            last_ts=float(obs["time"]),
            obs=len(prof["resnet"]),
        )

    def pending_score(self, obs, item):
        values = {
            model: self.best([obs[model]], item.get(model, []))
            for model in self.models
        }
        deep = (
            0.25 * values["resnet"]
            + 0.40 * values["swin"]
            + 0.35 * values["solider"]
        )
        crop_quality = float(np.clip(obs.get("quality", 1.0), 0.0, 1.0))
        qweight = 0.75 + 0.25 * crop_quality
        deep *= qweight
        attrs = self.attr(obs["attributes"], item.get("attributes", []))
        if attrs is None:
            return None
        top = float(attrs["top"]) * qweight
        bottom = float(attrs["bottom"]) * qweight
        pose = 0.0
        if obs.get("pose") is not None and item.get("pose"):
            pose = self.best([obs["pose"]], item["pose"])
        face = self.faceval(
            obs.get("face"),
            item.get("face", []),
        )
        if face.get("used") and face["score"] >= float(self.icfg.get("face_min", 0.60)):
            score = (
                0.45 * float(face["score"])
                + 0.31 * deep
                + 0.12 * top
                + 0.12 * bottom
            )
        else:
            score = (
                0.50 * deep
                + 0.25 * top
                + 0.25 * bottom
            )
        support = sum(
            values[name] >= self.pending_model_min
            for name in self.models
        )
        return {
            "score": float(np.clip(score, 0.0, 0.995)),
            "resnet": float(values["resnet"]),
            "swin": float(values["swin"]),
            "solider": float(values["solider"]),
            "top": top,
            "bottom": bottom,
            "clothing_joint": float(attrs["joint"]),
            "pose": float(pose),
            "face": float(face["score"]),
            "face_used": bool(face.get("used")),
            "support": int(support),
            "quality": crop_quality,
        }

    def _pending_add(self, obs):
        item = {
            "resnet": [np.asarray(obs["resnet"], np.float32)],
            "swin": [np.asarray(obs["swin"], np.float32)],
            "solider": [np.asarray(obs["solider"], np.float32)],
            "attributes": [np.asarray(obs["attributes"], np.float32)],
            "pose": (
                [np.asarray(obs["pose"], np.float32)]
                if obs.get("pose") is not None else []
            ),
            "face": (
                [np.asarray(obs["face"]["vector"], np.float32)]
                if obs.get("face") is not None and obs["face"].get("valid")
                else []
            ),
            "observations": [obs],
            "frames": {int(obs["row"].get("frame", -1))},
            "cameras": {str(obs["camera"])},
            "count": 1,
            "last_camera": str(obs["camera"]),
            "last_frame": int(obs["row"].get("frame", -1)),
            "last_time": float(obs["time"]),
        }
        self.pending.append(item)
        self.pending = self.pending[-self.pending_max:]
        self.stats["pending_new_observations"] += 1

    def _pending_update(self, item, obs):
        item["resnet"].append(np.asarray(obs["resnet"], np.float32))
        item["swin"].append(np.asarray(obs["swin"], np.float32))
        item["solider"].append(np.asarray(obs["solider"], np.float32))
        item["attributes"].append(np.asarray(obs["attributes"], np.float32))
        if obs.get("pose") is not None:
            item["pose"].append(np.asarray(obs["pose"], np.float32))
        if obs.get("face") is not None and obs["face"].get("valid"):
            item["face"].append(np.asarray(obs["face"]["vector"], np.float32))
        item["observations"].append(obs)
        item["frames"].add(int(obs["row"].get("frame", -1)))
        item["cameras"].add(str(obs["camera"]))
        item["count"] += 1
        item["last_camera"] = str(obs["camera"])
        item["last_frame"] = int(obs["row"].get("frame", -1))
        item["last_time"] = float(obs["time"])
        for name, size in (
            ("resnet", 96),
            ("swin", 96),
            ("solider", 96),
            ("attributes", 64),
            ("pose", 64),
            ("face", 48),
        ):
            item[name] = item[name][-size:]
        item["observations"] = item["observations"][-96:]
        self.stats["pending_new_observations"] += 1

    def _pending_prune(self, obs):
        now = float(obs["time"])
        self.pending = [
            item for item in self.pending
            if now - float(item.get("last_time", now)) <= self.pending_ttl
        ]
        self.pending = self.pending[-self.pending_max:]

    def stage_new(self, obs):
        self._pending_prune(obs)
        camera = str(obs["camera"])
        frame = int(obs["row"].get("frame", -1))
        now = float(obs["time"])
        ranked = []

        for index, item in enumerate(self.pending):
            if (
                item.get("last_camera") == camera
                and int(item.get("last_frame", -2)) == frame
            ):
                continue
            # Prevent simultaneous observations from different cameras from
            # being absorbed into the same not-yet-confirmed identity.
            if abs(now - float(item.get("last_time", now))) < self.pending_time_gap:
                continue

            score = self.pending_score(obs, item)
            if score is None:
                continue
            if (
                score["top"] < self.pending_clothing_min
                or score["bottom"] < self.pending_clothing_min
                or score["clothing_joint"] < self.pending_clothing_min
                or score["support"] < self.pending_required_models
            ):
                continue
            ranked.append((float(score["score"]), index, score))

        ranked.sort(key=lambda value: value[0], reverse=True)
        if ranked:
            best, index, _ = ranked[0]
            second = float(ranked[1][0]) if len(ranked) > 1 else 0.0
            if best >= self.pending_match and best - second >= self.pending_margin:
                item = self.pending[index]
                self._pending_update(item, obs)

                if (
                    item["count"] >= self.pending_min
                    and len(item["frames"]) >= 3
                ):
                    seed = item["observations"][0]
                    gid = self.new(seed)
                    for extra in item["observations"][1:]:
                        self.save(gid, extra)
                    self.pending.pop(index)
                    self.stats["pending_new_confirmed"] += 1
                    return gid
                return None

        self._pending_add(obs)
        return None

    def new(self, obs):
        gid = int(self.reg.allocate_gid())
        self.pro[gid] = {
            "resnet": [],
            "swin": [],
            "solider": [],
            "attributes": [],
            "face": [],
            "pose": [],
            "camera": set(),
        }
        self.save(gid, obs, force=True)
        self.stats["new"] += 1
        return gid

    def assign(self, obs, sets, commit=True, recovery=False):
        count = len(obs)
        if count == 0:
            return []

        out = ["UNKNOWN"] * count
        used_ids = set()

        def good(value):
            if value is None:
                return False
            support = sum(
                float(value.get(name, 0.0))
                >= float(self.icfg.get("model_min", 0.46))
                for name in self.models
            )
            return (
                float(value.get("top", 0.0))
                >= float(self.icfg.get("top_min", 0.52))
                and float(value.get("bottom", 0.0))
                >= float(self.icfg.get("bottom_min", 0.52))
                and float(value.get("clothing_joint", 0.0))
                >= float(self.icfg.get("clothing_joint_min", 0.52))
                and support >= 2
            )

        # First lock every continuing NvDCF track to its own previously
        # verified identity. Re-ID still verifies the live crop and BOTH
        # clothing halves, but another gallery person can never steal a stable
        # tracker track because it happened to score slightly higher.
        remaining_obs = []
        remaining_sets = []
        remaining_indices = []
        for index, item in enumerate(obs):
            hint = item.get("track_hint")
            if hint is None:
                remaining_obs.append(item)
                remaining_sets.append(sets[index])
                remaining_indices.append(index)
                continue
            try:
                hinted_gid = int(hint)
            except (TypeError, ValueError):
                remaining_obs.append(item)
                remaining_sets.append(sets[index])
                remaining_indices.append(index)
                continue

            value = sets[index].get(hinted_gid)
            if (
                hinted_gid not in used_ids
                and good(value)
                and float(value.get("score", 0.0))
                >= (0.44 if not recovery else 0.48)
            ):
                out[index] = f"G{hinted_gid:06d}"
                used_ids.add(hinted_gid)
                if commit:
                    self.save(hinted_gid, item)
                    if recovery:
                        self.stats["recovery_match"] += 1
                    if len(self.pro.get(hinted_gid, {}).get("camera", set())) > 1:
                        self.stats["cross"] += 1
                continue

            # Outside an overlap-recovery event, NvDCF continuity may hold a
            # previously verified identity through a temporary feature dip. This
            # is temporal support, not identity creation.
            # During recovery, however, the tracker hint is NEVER authoritative:
            # the candidate must be accepted from the current multimodal evidence.
            if (
                not recovery
                and hinted_gid not in used_ids
                and hinted_gid in self.pro
            ):
                out[index] = f"G{hinted_gid:06d}"
                used_ids.add(hinted_gid)
                if commit:
                    self.save(hinted_gid, item)
                continue

            remaining_obs.append(item)
            remaining_sets.append(sets[index])
            remaining_indices.append(index)

        if remaining_obs:
            chosen = solve(
                remaining_sets,
                lambda item, second: self.accept(
                    item,
                    second,
                    recovery=recovery,
                ),
                floor=float(self.icfg.get("dummy_floor", 0.35)),
            )
            for local_index, gid in sorted(chosen.items()):
                gid = int(gid)
                original = remaining_indices[int(local_index)]
                if gid in used_ids:
                    self.stats["duplicate"] += 1
                    continue
                used_ids.add(gid)
                out[original] = f"G{gid:06d}"
                if commit:
                    self.save(gid, remaining_obs[int(local_index)])
                    if recovery:
                        self.stats["recovery_match"] += 1
                    if len(self.pro.get(gid, {}).get("camera", set())) > 1:
                        self.stats["cross"] += 1

            # Resolve new-track and overlap rows. Existing gallery matches must
            # still pass the strict top+bottom clothing gates. During overlap we
            # never invent a new identity from a difficult crop.
            for local_index, item in enumerate(remaining_obs):
                original = remaining_indices[local_index]
                if out[original].startswith("G") or item is None:
                    continue

                recovery_ids = {
                    int(x)
                    for x in (item.get("recovery_hints") or [])
                    if str(x).lstrip("-").isdigit()
                }
                recovery_order = []
                for value in (item.get("recovery_hints") or []):
                    if not str(value).lstrip("-").isdigit():
                        continue
                    value = int(value)
                    if value not in recovery_order:
                        recovery_order.append(value)
                ranked = sorted(
                    (
                        (int(gid), value)
                        for gid, value in remaining_sets[local_index].items()
                    ),
                    key=lambda pair: float(pair[1].get("score", 0.0)),
                    reverse=True,
                )

                # Prefer a feature-verified overlap anchor over a generic
                # gallery candidate. This is the key one-to-one overlap path.
                if recovery:
                    hinted_rows = [
                        (gid, value)
                        for gid, value in ranked
                        if gid in recovery_ids
                        and gid not in used_ids
                        and good(value)
                        and float(value.get("score", 0.0)) >= 0.48
                    ]
                    if hinted_rows:
                        gid, value = hinted_rows[0]
                        out[original] = f"G{gid:06d}"
                        used_ids.add(gid)
                        if commit:
                            self.save(gid, item)
                            self.stats["recovery_match"] += 1
                            if len(self.pro.get(gid, {}).get("camera", set())) > 1:
                                self.stats["cross"] += 1
                        continue

                    shadow = bool(
                        item.get("row", {}).get("shadow", False)
                        or str(item.get("row", {}).get("shadow_reason", "")) == "collapse"
                    )
                    if shadow:
                        ordered = [
                            value
                            for value in recovery_order
                            if value not in used_ids
                        ]
                        if ordered:
                            gid = ordered[0]
                            out[original] = f"G{gid:06d}"
                            used_ids.add(gid)
                            self.stats["occlusion_continuity"] = (
                                self.stats.get("occlusion_continuity", 0) + 1
                            )
                            continue

                    raise RuntimeError(
                        f"overlap/recovery identity could not be verified: "
                        f"camera={item['camera']} frame={item['row'].get('frame')} "
                        f"track={item['row'].get('track_id')} "
                        f"anchors={sorted(recovery_ids)}"
                    )

                # New track: accept the strongest existing identity only if
                # the complete strict multimodal gate passes.
                for gid, value in ranked:
                    if gid in used_ids or not good(value):
                        continue
                    if float(value.get("score", 0.0)) >= float(
                        self.icfg.get("existing_min", 0.62)
                    ):
                        out[original] = f"G{gid:06d}"
                        used_ids.add(gid)
                        if commit:
                            self.save(gid, item)
                            if len(self.pro.get(gid, {}).get("camera", set())) > 1:
                                self.stats["cross"] += 1
                        break

                if out[original].startswith("G"):
                    continue

                if commit:
                    gid = int(self.new(item))
                else:
                    gid = int(self.reg.allocate_gid())
                    self.pro[gid] = {
                        "resnet": [np.asarray(item["resnet"], np.float32)],
                        "swin": [np.asarray(item["swin"], np.float32)],
                        "solider": [np.asarray(item["solider"], np.float32)],
                        "attributes": [np.asarray(item["attributes"], np.float32)],
                        "face": [],
                        "pose": [],
                        "camera": {str(item["camera"])},
                    }
                while gid in used_ids:
                    gid = int(self.new(item)) if commit else int(self.reg.allocate_gid())
                used_ids.add(gid)
                out[original] = f"G{gid:06d}"

        for index, item in enumerate(obs):
            if item is None:
                raise RuntimeError(
                    f"Invalid person crop for observation index {index}"
                )
            if not out[index].startswith("G"):
                raise RuntimeError(
                    f"identity resolver returned no GID for observation index {index}"
                )

        # Emit the actual evidence used for each decision. This is diagnostic
        # metadata only; the assignment itself has already been completed above.
        self.last_evidence = {}
        for index, item in enumerate(obs):
            value = str(out[index])
            if not value.startswith("G"):
                continue
            gid = int(value[1:])
            score = sets[index].get(gid) if index < len(sets) else None
            evidence = {
                "gid": value,
                "decision": (
                    "feature"
                    if score is not None and (
                        recovery or bool(item.get("track_hint") is None)
                    )
                    else "temporal"
                ),
            }
            if score is not None:
                for name in (
                    "score",
                    "deep",
                    "resnet",
                    "swin",
                    "solider",
                    "top",
                    "bottom",
                    "clothing_joint",
                    "pose",
                    "face",
                    "face_used",
                    "face_reliable",
                    "face_quality",
                    "face_visibility",
                    "quality",
                    "recovery_hint",
                ):
                    if name in score:
                        evidence[name] = score[name]
            self.last_evidence[int(item["row"].get("track_id", -1))] = evidence

        return out

    def observe(self, frame, rows, commit=True, recovery=False, recovery_hints=None):
        poses = self.poses(self.pose, frame)
        obs = []
        self.stats["feature"] += 1
        if recovery:
            self.stats["recovery"] += 1
        for row in rows:
            person = crop(frame, row["bbox"])
            if person is None or person.size == 0:
                obs.append(None)
                continue
            hints = list((recovery_hints or {}).get(int(row["track_id"]), []) or [])
            item = {
                "row": row,
                "track_hint": row.get("track_hint"),
                "camera": str(row["camera"]),
                "time": float(row["timestamp"]),
                "person": person,
                "quality": float(quality(person)),
                "recovery_hints": [int(x) for x in hints if str(x).lstrip("-").isdigit()],
            }
            obs.append(item)
        valid = [x for x in obs if x is not None]
        if not valid:
            return ["PENDING"] * len(rows)
        imgs = [x["person"] for x in valid]
        resnet = self.resnet.extract_batch(imgs)
        swin = self.swin.extract_batch(imgs)
        solider = self.solider.extract_batch(imgs)
        if not (len(resnet) == len(swin) == len(solider) == len(valid)):
            raise RuntimeError("ReID extractors returned mismatched batch lengths")
        vi = 0
        for item in obs:
            if item is None:
                continue
            item["resnet"] = np.asarray(resnet[vi], np.float32)
            item["swin"] = np.asarray(swin[vi], np.float32)
            item["solider"] = np.asarray(solider[vi], np.float32)
            item["attributes"] = np.asarray(
                self.pack(item["person"], frame, item["row"]["bbox"]),
                np.float32,
            )
            face = self.face.extract(item["person"])
            item["face"] = None if face is None else {
                "vector": np.asarray(face.vector, np.float32),
                "quality": float(face.quality),
                "visibility": float(face.visibility),
                "valid": bool(face.valid),
            }
            if item["face"] is not None:
                self.stats["face_observations"] += 1
                if item["face"]["valid"]:
                    self.stats["face_reliable"] += 1
            item["pose"], item["posscore"] = self.poseval(
                poses,
                item["row"]["bbox"],
            )
            vi += 1
        sets = []
        for item in obs:
            if item is None:
                sets.append({})
                continue
            ids, got = self.cand(item)
            rows = {}
            for gid in ids:
                value = self.score(item, gid, got)
                if value is not None:
                    rows[int(gid)] = value
            sets.append(rows)
        good = []
        scores = []
        for item, value in zip(obs, sets):
            if item is None:
                continue
            good.append(item)
            scores.append(value)
        # Preserve the one-to-one row alignment even when an observation has
        # no existing GID candidate. Empty candidate sets must still reach
        # stage_new() so genuine new people can be confirmed across frames.
        amap = self.assign(good, scores, commit=commit, recovery=recovery)
        out = []
        vi = 0
        for item in obs:
            if item is None:
                out.append("PENDING")
            else:
                out.append(amap[vi])
                vi += 1
        if recovery:
            for value in out:
                if str(value).startswith("G"):
                    self.stats["recovery_feature_verified"] += 1

        return out

    def close(self):
        self.reg.close()


__all__ = ["MultiModalStrict"]