from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np

from rebuild.multimodel_state_invariant_attributes import AttributeAwareResolver
from src.live.qdrant_gallery import QdrantGallery


class QdrantAuthoritativeResolver(AttributeAwareResolver):
    """Conservative MTMC resolver using Qdrant for identity retrieval."""

    MODEL_WEIGHTS = {
        "resnet": 0.30,
        "swin": 0.37,
        "solider": 0.33,
    }

    def __init__(self, cfg, registry=None):
        super().__init__(cfg, registry=registry)
        self.exist = float(cfg.get("authoritative_existing_min", 0.72))
        self.margin = float(cfg.get("authoritative_margin", 0.06))
        self.novel = float(cfg.get("authoritative_novel_max", 0.55))
        self.clean = max(3, int(cfg.get("authoritative_clean_required", 4)))
        self.lower_floor = float(cfg.get("lower_conflict_floor", 0.38))
        self.clothing_weight = float(cfg.get("clothing_match_weight", 0.18))
        self.face_weight = float(cfg.get("face_match_weight", 0.06))
        self.face_match_min = float(cfg.get("face_match_min", 0.72))
        self.face_strong = float(cfg.get("face_strong", 0.82))
        self._gallery_diagnostics = []
        self._decision_diagnostics = []
        path = str(cfg.get("qdrant_path", "qdrant_storage"))
        prefix = str(cfg.get("qdrant_prefix", "person_reid"))
        limit = int(cfg.get("qdrant_limit", 24))
        self.qdrant = QdrantGallery(path, prefix, limit)

    @staticmethod
    def _flat(component):
        result = defaultdict(dict)
        for group in component:
            for model in ("resnet", "swin", "solider"):
                banks = getattr(group, "state_bank", {}).get(model, {}) or {}
                for view, values in banks.items():
                    if view in ("full", "upper", "torso", "lower"):
                        result[model].setdefault(view, []).extend(values)
        return {model: dict(views) for model, views in result.items()}

    @staticmethod
    def _attrs_flat(component):
        values = []
        for group in component:
            values.extend(getattr(group, "attribute_bank", []) or [])
            if not getattr(group, "attribute_bank", None):
                values.extend(
                    getattr(group, "state_bank", {})
                    .get("resnet", {})
                    .get("attributes", [])
                    or []
                )
        return values[-64:]

    @staticmethod
    def _faces_flat(component):
        values = []
        for group in component:
            values.extend(getattr(group, "face_bank", []) or [])
        return values[-32:]

    @staticmethod
    def _profile(attrs=None, faces=None):
        return type(
            "Profile",
            (),
            {
                "attribute_bank": list(attrs or []),
                "face_bank": list(faces or []),
                "state_bank": {},
            },
        )()

    def _candidate_score(self, component, stored):
        current = self._flat(component)
        model_scores = {}

        for model in self.MODEL_WEIGHTS:
            left = current.get(model, {})
            right = stored.get(model, {})
            if not left or not right:
                continue
            score, _, _ = self._best_view_score(left, right)
            if score > 0.0:
                model_scores[model] = float(score)

        if len(model_scores) < 2:
            return None

        total = sum(self.MODEL_WEIGHTS[name] for name in model_scores)
        deep = sum(
            self.MODEL_WEIGHTS[name] * model_scores[name]
            for name in model_scores
        ) / max(total, 1e-9)

        current_attrs = self._attrs_flat(component)
        stored_attrs = list(stored.get("attributes", {}).get("attributes", []))
        attrs = {"ready": False}
        if current_attrs and stored_attrs:
            attrs = self._attrs(
                self._profile(attrs=current_attrs),
                self._profile(attrs=stored_attrs),
            )

        if attrs.get("ready") and attrs.get("conflict"):
            return {
                "score": None,
                "model_scores": model_scores,
                "attrs": attrs,
                "face": {"valid": False, "score": 0.0, "quality": 0.0},
                "rejected": True,
                "reason": "hard_lower_body_conflict",
            }

        score = float(deep)
        clothing = None

        if attrs.get("ready"):
            upper = float(attrs.get("upper", 0.0))
            lower = float(attrs.get("lower", 0.0))
            clothing = 0.35 * upper + 0.65 * lower
            score = (1.0 - self.clothing_weight) * score + self.clothing_weight * clothing

        current_faces = self._faces_flat(component)
        stored_faces = []
        for item in stored.get("face", {}).get("face", []):
            if isinstance(item, dict):
                stored_faces.append(item)
            else:
                stored_faces.append({"vector": item, "valid": True, "quality": 1.0})

        face = self._face(
            self._profile(faces=current_faces),
            self._profile(faces=stored_faces),
        )

        face_used = bool(
            face.get("valid")
            and float(face.get("quality", 0.0)) >= 0.50
            and float(face.get("score", 0.0)) >= self.face_match_min
        )

        if face_used:
            score = (1.0 - self.face_weight) * score + self.face_weight * float(face["score"])

        return {
            "score": float(np.clip(score, 0.0, 0.99)),
            "model_scores": model_scores,
            "attrs": attrs,
            "face": face,
            "face_used": face_used,
            "clothing": clothing,
            "rejected": False,
            "reason": "qdrant_multimodal_candidate",
        }

    def _gallery_score(self, component, _gallery):
        candidates, retrieved = self.qdrant.search_component(component)
        result = {}
        self._gallery_diagnostics = []

        for gid in sorted(candidates):
            stored = retrieved.get(int(gid), {})
            data = self._candidate_score(component, stored)
            summary = {
                "candidate_gid": int(gid),
                "resnet": None,
                "swin": None,
                "solider": None,
                "views": sorted(self._flat(component).keys()),
                "clothing": None,
                "lower_clothing": None,
                "upper_clothing": None,
                "geometry": None,
                "temporal": None,
                "face": 0.0,
                "face_used": False,
                "face_available": bool(self._faces_flat(component)),
                "overlap_recovery": any(bool(getattr(group, "overlap_recovery", False)) for group in component),
            }

            if data is None:
                summary["decision"] = "IGNORED"
                summary["reason"] = "insufficient_model_support"
                self._gallery_diagnostics.append(summary)
                continue

            for model, value in data["model_scores"].items():
                summary[model] = float(value)

            summary["face"] = float(data["face"].get("score", 0.0))
            summary["face_used"] = bool(data.get("face_used", False))

            attrs = data.get("attrs", {})
            summary["clothing"] = data.get("clothing")
            summary["lower_clothing"] = attrs.get("lower")
            summary["upper_clothing"] = attrs.get("upper")

            if data.get("rejected"):
                summary["decision"] = "REJECT"
                summary["reason"] = data["reason"]
                self._gallery_diagnostics.append(summary)
                continue

            result[int(gid)] = float(data["score"])
            summary["decision"] = "CANDIDATE"
            summary["reason"] = data["reason"]
            summary["final_score"] = float(data["score"])
            self._gallery_diagnostics.append(summary)

        return result

    @staticmethod
    def _evidence(component):
        total = 0
        models = set()
        views = set()
        for group in component:
            bank = getattr(group, "state_bank", {}) or {}
            for model in ("resnet", "swin", "solider"):
                for view, values in (bank.get(model, {}) or {}).items():
                    if values:
                        models.add(model)
                        views.add(view)
                total = max(total, len(bank.get("resnet", {}).get("full", []) or []))
        return total, len(models), len(views)

    def _save(self, gid, component):
        if self.registry is not None:
            banks = {}
            for model in ("resnet", "swin", "solider"):
                values = []
                for group in component:
                    values.extend(
                        getattr(group, "state_bank", {})
                        .get(model, {})
                        .get("full", [])
                        or []
                    )
                banks[model] = values

            self.registry.save_component(
                int(gid),
                model_banks=banks,
                cameras={str(group.camera) for group in component},
                last_ts=max(float(group.end) for group in component),
                obs=len({key for group in component for key in group.members}),
            )

        self.qdrant.upsert_component(int(gid), component)

    def _assign(self, components):
        result = {}
        self._decision_diagnostics = []
        used = set()

        for component in components:
            members = [key for group in component for key in group.members]
            recovery = any(
                bool(getattr(group, "overlap_recovery", False))
                for group in component
            )

            ranked = sorted(
                self._gallery_score(component, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )

            best = ranked[0][0] if ranked else None
            best_score = float(ranked[0][1]) if ranked else 0.0
            second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
            margin = best_score - second_score

            if best in used:
                best = None
                best_score = 0.0
                second_score = 0.0
                margin = 0.0

            if recovery:
                if best is not None and best_score >= self.exist and margin >= self.margin:
                    gid = int(best)
                    decision = "RECOVERY_CONFIRMED"
                    reason = "strong_existing_identity_after_overlap"
                else:
                    gid = None
                    decision = "PENDING"
                    reason = "overlap_recovery_not_authoritatively_confirmed"
            elif best is not None and best_score >= self.exist and margin >= self.margin:
                gid = int(best)
                decision = "ASSIGN_EXISTING"
                reason = "strong_qdrant_gallery_match"
            else:
                evidence, model_count, view_count = self._evidence(component)
                if (
                    best_score < self.novel
                    and evidence >= self.clean
                    and model_count >= 2
                    and view_count >= 2
                ):
                    if self.registry is None:
                        gid = 1
                    else:
                        gid = int(self.registry.allocate_gid())
                    decision = "CREATE_NEW"
                    reason = "novel_multiframe_multimodel_evidence"
                    self._save(gid, component)
                else:
                    gid = None
                    decision = "UNKNOWN"
                    reason = (
                        "ambiguous_existing_identity"
                        if best is not None
                        else "insufficient_novelty_or_evidence"
                    )

            self._decision_diagnostics.append({
                "decision": decision,
                "candidate_gid": best,
                "best_score": best_score,
                "second_score": second_score,
                "margin": margin,
                "members": members,
                "reason": reason,
            })

            label = f"G{gid:06d}" if gid is not None else ("PENDING" if recovery else "UNKNOWN")
            for key in members:
                result[key] = label

            if gid is not None and decision != "CREATE_NEW":
                self._save(gid, component)
                used.add(gid)

        return result


__all__ = ["QdrantAuthoritativeResolver"]
