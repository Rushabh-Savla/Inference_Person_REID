from __future__ import annotations

from collections import defaultdict

import numpy as np

from rebuild.multimodel_state_invariant_attributes import AttributeAwareResolver
from src.live.qdrant_gallery import QdrantGallery


class QdrantAuthoritativeResolver(AttributeAwareResolver):
    """Accuracy-first authoritative identity resolver backed by Qdrant."""

    MODEL_WEIGHTS = {"resnet": 0.30, "swin": 0.37, "solider": 0.33}

    def __init__(self, cfg, registry=None):
        super().__init__(cfg, registry=registry)
        self.exist = float(cfg.get("authoritative_existing_min", 0.72))
        self.margin = float(cfg.get("authoritative_margin", 0.06))
        self.novel = float(cfg.get("authoritative_novel_max", 0.55))
        self.clean = max(4, int(cfg.get("authoritative_clean_required", 4)))
        self.lower_floor = float(cfg.get("lower_conflict_floor", 0.45))
        self.clothing_weight = float(cfg.get("clothing_match_weight", 0.30))
        self.face_weight = float(cfg.get("face_match_weight", 0.06))
        self.face_match_min = float(cfg.get("face_match_min", 0.72))
        self._gallery_diagnostics = []
        self._decision_diagnostics = []
        self.qdrant = QdrantGallery(
            str(cfg.get("qdrant_path", "qdrant_storage")),
            str(cfg.get("qdrant_prefix", "person_reid")),
            int(cfg.get("qdrant_limit", 24)),
        )

    @staticmethod
    def _flat(component):
        result = defaultdict(dict)
        for group in component:
            banks = getattr(group, "state_bank", {}) or {}
            for model in ("resnet", "swin", "solider"):
                for view, values in (banks.get(model, {}) or {}).items():
                    if view in ("full", "upper", "torso", "lower") and values:
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

    def _faces_flat(self, component):
        values = []
        for group in component:
            values.extend(getattr(group, "face_bank", []) or [])
        return values[-32:]

    def _candidate_score(self, component, stored):
        current = self._flat(component)
        model_scores = {}
        for model in self.MODEL_WEIGHTS:
            left = current.get(model, {})
            right = stored.get(model, {})
            if left and right:
                score, _, _ = self._best_view_score(left, right)
                if score > 0.0:
                    model_scores[model] = float(score)

        if len(model_scores) < 2:
            return None

        total = sum(self.MODEL_WEIGHTS[m] for m in model_scores)
        deep = sum(self.MODEL_WEIGHTS[m] * model_scores[m] for m in model_scores) / max(total, 1e-9)

        current_attrs = self._attrs_flat(component)
        stored_attrs = list(stored.get("attributes", {}).get("attributes", []) or [])
        attrs = {"ready": False}
        if current_attrs and stored_attrs:
            attrs = self._attrs(
                self._profile(attrs=current_attrs),
                self._profile(attrs=stored_attrs),
            )

        # Lower clothing is a hard contradiction when both people are lower-body
        # visible. This prevents a strong upper-body match from overriding a
        # clearly different trouser/skirt/shorts appearance.
        if attrs.get("ready") and attrs.get("lower_visible"):
            lower = float(attrs.get("lower", 0.0))
            upper = float(attrs.get("upper", 0.0))
            if lower < self.lower_floor and upper >= 0.70:
                return {
                    "score": None,
                    "model_scores": model_scores,
                    "attrs": attrs,
                    "face": {"valid": False, "score": 0.0, "quality": 0.0},
                    "rejected": True,
                    "reason": "hard_lower_clothing_conflict",
                }

        score = float(deep)
        clothing = None
        if attrs.get("ready"):
            upper = float(attrs.get("upper", 0.0))
            lower = float(attrs.get("lower", 0.0))
            # Lower clothing has higher influence than upper clothing because it
            # is the discriminative cue required by the current failure case.
            clothing = 0.20 * upper + 0.80 * lower
            score = (1.0 - self.clothing_weight) * score + self.clothing_weight * clothing

        faces = []
        for item in self._faces_flat(component):
            faces.append(item)
        stored_faces = []
        for item in stored.get("face", {}).get("face", []) or []:
            if isinstance(item, dict):
                stored_faces.append(item)
            else:
                stored_faces.append({"vector": item, "valid": True, "quality": 1.0})

        face = self._face(
            self._profile(faces=faces),
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
            "reason": "qdrant_multimodal_lower_aware",
        }

    def _gallery_score(self, component, _gallery):
        candidates, retrieved = self.qdrant.search_component(component)
        result = {}
        self._gallery_diagnostics = []
        for gid in sorted(candidates):
            data = self._candidate_score(component, retrieved.get(int(gid), {}))
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
                "overlap_recovery": any(bool(getattr(g, "overlap_recovery", False)) for g in component),
            }
            if data is None:
                summary["decision"] = "IGNORED"
                summary["reason"] = "insufficient_model_support"
                self._gallery_diagnostics.append(summary)
                continue
            summary.update({m: float(v) for m, v in data["model_scores"].items()})
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
                        if view in ("full", "upper", "torso", "lower"):
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
                cameras={str(g.camera) for g in component},
                last_ts=max(float(g.end) for g in component),
                obs=len({key for g in component for key in g.members}),
            )
        self.qdrant.upsert_component(int(gid), component)

    def _assign(self, components):
        result = {}
        self._decision_diagnostics = []
        used = set()
        for component in components:
            members = [key for group in component for key in group.members]
            recovery = any(bool(getattr(group, "overlap_recovery", False)) for group in component)
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
                best_score = second_score = margin = 0.0

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
                if best_score < self.novel and evidence >= self.clean and model_count == 3 and view_count >= 2:
                    gid = int(self.registry.allocate_gid()) if self.registry is not None else 1
                    decision = "CREATE_NEW"
                    reason = "novel_multiframe_three_model_multiview_evidence"
                    self._save(gid, component)
                else:
                    gid = None
                    decision = "UNKNOWN"
                    reason = "ambiguous_existing_identity" if best is not None else "insufficient_novelty_or_evidence"

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
