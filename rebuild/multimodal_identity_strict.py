from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

from rebuild.identity_v2 import crop, quality
from rebuild.face_v4 import FaceExtractorV4
from rebuild.person_attributes import pack
from reid.nvidia_reid import NVIDIAReIDExtractor
from reid.nvidia_swin import NVIDIASwinReIDExtractor
from reid.solider_reid import SOLIDERReIDExtractor
from src.live.persistent_multimodel import PersistentMultimodelRegistry
from src.live.qdrant_gallery import QdrantGallery
from ultralytics import YOLO


class MultiModalStrict:
    """Feature-first identity resolver for the real NvDCF pipeline."""

    models = ("resnet", "swin", "solider")

    def __init__(self, cfg):
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
        self.q = QdrantGallery(
            str(state["qdrant_path"]),
            str(state.get("qdrant_prefix", "person_reid")),
            int(state.get("qdrant_limit", 32)),
        )
        self.pro = self.load()
        self.pending = []
        self.pending_min = max(3, int(self.icfg.get("new_confirm_frames", 4)))
        self.pending_match = float(self.icfg.get("new_pending_match_min", 0.82))
        self.pending_margin = float(self.icfg.get("new_pending_margin", 0.08))
        self.pending_model_min = float(self.icfg.get("new_pending_model_min", 0.55))
        self.pending_clothing_min = float(self.icfg.get("new_pending_clothing_min", 0.58))
        self.pending_required_models = 3
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
        top = cls.best([query[:20]], [x[:20] for x in bank])
        bot = cls.best([query[20:40]], [x[20:40] for x in bank])
        upp = cls.best([query[40:54]], [x[40:54] for x in bank])
        low = cls.best([query[54:68]], [x[54:68] for x in bank])
        return {
            "top": float(top),
            "bottom": float(bot),
            "upper_pattern": float(upp),
            "lower_pattern": float(low),
            "pattern": float((upp + low) * 0.50),
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
        out = {}
        for gid, bank in data.items():
            out[int(gid)] = {
                "resnet": list(bank.get("resnet", [])),
                "swin": list(bank.get("swin", [])),
                "solider": list(bank.get("solider", [])),
                "attributes": list(bank.get("attributes", [])),
                "face": list(bank.get("face", [])),
                "pose": list(bank.get("pose", [])),
                "camera": set(),
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
        ids = set(int(x) for x in hits)
        ids.update(int(x) for x in self.pro)
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
            if face.get("used") and obs.get("face") else 0.0
        )
        weights = {"face": float(self.fcfg.get("face_weight", 0.46)), "deep": 0.28, "top": 0.10, "bottom": 0.10, "pose": 0.04, "pattern": 0.02}
        qualities = {
            "face": faceq,
            "deep": qweight,
            "top": qweight,
            "bottom": qweight,
            "pose": float(np.clip(obs.get("posscore", 0.0), 0.0, 1.0)),
            "pattern": qweight,
        }
        active = []
        for name, weight in weights.items():
            if name == "face" and (
                not face.get("used")
                or face["score"] < float(self.icfg.get("face_min", 0.60))
            ):
                continue
            if name == "pose" and qualities[name] <= 0.0:
                continue
            active.append((name, weight * qualities[name]))
        total = sum(value for _, value in active)
        if total <= 0.0:
            return None
        value = 0.0
        for name, weight in active:
            raw = {
                "face": float(face["score"]),
                "deep": float(deep),
                "top": float(top),
                "bottom": float(bot),
                "pose": float(pose),
                "pattern": float(pattern),
            }[name]
            value += (weight / total) * raw
        return {
            "score": float(np.clip(value, 0.0, 0.995)),
            "deep": float(deep),
            "resnet": float(vals["resnet"]),
            "swin": float(vals["swin"]),
            "solider": float(vals["solider"]),
            "top": float(top),
            "bottom": float(bot),
            "upper_pattern": float(attrs["upper_pattern"]),
            "lower_pattern": float(attrs["lower_pattern"]),
            "pose": float(pose),
            "face": float(face["score"]),
            "face_used": bool(face.get("used")),
            "face_quality": float(faceq),
            "quality": crop_quality,
        }

    def accept(self, row, second, recovery=False):
        if row is None:
            return False
        margin = float(row["score"] - second)
        mins = self.models
        support = sum(float(row[name]) >= float(self.icfg.get("model_min", 0.46)) for name in mins)
        top = float(row["top"])
        bot = float(row["bottom"])
        if top < float(self.icfg.get("top_min", 0.42)) or bot < float(self.icfg.get("bottom_min", 0.42)):
            return False
        if support < 2:
            return False
        if float(row.get("quality", 0.0)) < float(self.icfg.get("memory_quality_min", 0.45)):
            return False
        floor = float(self.icfg.get("recovery_min", 0.58) if recovery else self.icfg.get("existing_min", 0.61))
        gap = float(self.icfg.get("recovery_margin", 0.018) if recovery else self.icfg.get("margin", 0.025))
        face_ok = (
            not bool(row.get("face_used"))
            or (
                float(row.get("face", 0.0)) >= float(self.icfg.get("face_min", 0.60))
                and float(row.get("face_quality", 0.0))
                >= float(self.icfg.get("face_quality_min", 0.50))
            )
        )
        return bool(
            row["score"] >= floor
            and row["deep"] >= float(self.icfg.get("deep_min", 0.48))
            and margin >= gap
            and face_ok
        )

    def save(self, gid, obs):
        gid = int(gid)
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
            "pose": float(pose),
            "face": float(face["score"]),
            "face_used": bool(face.get("used")),
            "support": int(support),
            "quality": crop_quality,
        }

    def stage_new(self, obs):
        frame = int(obs["row"].get("frame", -1))
        camera = str(obs["camera"])
        ranked = []
        for index, item in enumerate(self.pending):
            if (
                item.get("last_camera") == camera
                and int(item.get("last_frame", -2)) == frame
            ):
                continue
            score = self.pending_score(obs, item)
            if score is None:
                continue
            if (
                score["top"] < self.pending_clothing_min
                or score["bottom"] < self.pending_clothing_min
                or score["support"] < self.pending_required_models
            ):
                continue
            ranked.append((float(score["score"]), index, score))
        ranked.sort(reverse=True)
        if ranked:
            best, index, _ = ranked[0]
            second = float(ranked[1][0]) if len(ranked) > 1 else 0.0
            margin = best - second
            if best >= self.pending_match and margin >= self.pending_margin:
                item = self.pending[index]
                item["resnet"].append(np.asarray(obs["resnet"], np.float32))
                item["swin"].append(np.asarray(obs["swin"], np.float32))
                item["solider"].append(np.asarray(obs["solider"], np.float32))
                item["attributes"].append(np.asarray(obs["attributes"], np.float32))
                if obs.get("pose") is not None:
                    item["pose"].append(np.asarray(obs["pose"], np.float32))
                if obs.get("face") is not None and obs["face"].get("valid"):
                    item["face"].append(np.asarray(obs["face"]["vector"], np.float32))
                item["observations"].append(obs)
                item["count"] += 1
                item["last_camera"] = camera
                item["last_frame"] = frame
                self.stats["pending_new_observations"] += 1
                if item["count"] >= self.pending_min:
                    seed = item["observations"][0]
                    gid = self.new(seed)
                    for extra in item["observations"][1:]:
                        self.save(gid, extra)
                    self.pending.pop(index)
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
                if obs.get("pose") is not None else []
            ),
            "face": (
                [np.asarray(obs["face"]["vector"], np.float32)]
                if obs.get("face") is not None and obs["face"].get("valid")
                else []
            ),
            "observations": [obs],
            "count": 1,
            "last_camera": camera,
            "last_frame": frame,
        })
        self.stats["pending_new_observations"] += 1
        return None

    def new(self, obs):
        gid = int(self.reg.allocate_gid())
        self.pro[gid] = {
            "resnet": [], "swin": [], "solider": [],
            "attributes": [], "face": [], "pose": [], "camera": set(),
        }
        self.save(gid, obs)
        self.stats["new"] += 1
        return gid

    def assign(self, obs, sets, commit=True, recovery=False):
        count = len(obs)
        gids = sorted(self.pro)
        if count == 0:
            return []
        floor = float(self.icfg.get("dummy_floor", 0.35))
        cols = len(gids) + count
        mat = np.full((count, cols), floor, np.float32)
        for i, rows in enumerate(sets):
            for j, gid in enumerate(gids):
                item = rows.get(gid)
                if item is not None:
                    mat[i, j] = float(item["score"])
        rr, cc = linear_sum_assignment(-mat)
        assignments = {
            int(i): int(gids[j])
            for i, j in zip(rr.tolist(), cc.tolist())
            if j < len(gids)
        }
        take = {}
        for i, j in zip(rr.tolist(), cc.tolist()):
            rows = sets[i]
            ranked = sorted(rows.items(), key=lambda x: x[1]["score"], reverse=True)
            second = float(ranked[1][1]["score"]) if len(ranked) > 1 else 0.0
            item = rows.get(gids[j]) if j < len(gids) else None
            gid = None
            if item is not None:
                selected_gid = int(gids[j])
                pair_second = max(
                    (
                        float(value["score"])
                        for other_gid, value in rows.items()
                        if (
                            int(other_gid) != selected_gid
                            and int(other_gid) not in assignments.values()
                        )
                    ),
                    default=0.0,
                )
                if self.accept(item, pair_second, recovery=recovery):
                    gid = selected_gid
            if gid is None and not recovery and commit:
                best = float(ranked[0][1]["score"]) if ranked else 0.0
                deep = float(ranked[0][1]["deep"]) if ranked else 0.0
                if (
                    best < float(self.icfg.get("new_max", 0.48))
                    and deep < float(self.icfg.get("new_deep_max", 0.54))
                ):
                    gid = self.stage_new(obs[i])
            if gid is not None:
                take[i] = gid
        used = set()
        out = ["PENDING"] * count
        for i in range(count):
            gid = take.get(i)
            if gid is None:
                self.stats["pending"] += 1
                continue
            if gid in used:
                self.stats["duplicate"] += 1
                continue
            used.add(gid)
            out[i] = f"G{gid:06d}"
        for i, gid in take.items():
            if out[i] == "PENDING" or not commit:
                continue
            item = obs[i]
            self.save(gid, item)
            if recovery:
                self.stats["recovery_match"] += 1
            if len(self.pro[gid]["camera"]) > 1:
                self.stats["cross"] += 1
        return out

    def observe(self, frame, rows, commit=True, recovery=False):
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
            item = {
                "row": row,
                "camera": str(row["camera"]),
                "time": float(row["timestamp"]),
                "person": person,
                "quality": float(quality(person)),
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
                pack(item["person"], frame, item["row"]["bbox"]),
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
        return out

    def close(self):
        self.reg.close()


__all__ = ["MultiModalStrict"]