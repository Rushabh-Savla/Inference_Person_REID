from __future__ import annotations

import numpy as np

from rebuild.multimodel_state_invariant_attributes import AttributeAwareResolver
from rebuild.multimodel_state_invariant_final import LocalGroup, PairEvidence


class OverlapReidResolver(AttributeAwareResolver):
    """Appearance-first resolver with continuous same-camera trajectory evidence."""

    def __init__(self, cfg, registry=None):
        super().__init__(cfg, registry=registry)
        self.trajectory_min = float(cfg.get("state_same_trajectory_min", 0.35))

    @classmethod
    def _trajectory(cls, group: LocalGroup) -> list[dict]:
        return list(getattr(group, "trajectory", []) or [])

    @classmethod
    def _trajectory_score(cls, left: LocalGroup, right: LocalGroup) -> float:
        if left.camera != right.camera:
            return 0.5
        a = cls._trajectory(left)
        b = cls._trajectory(right)
        if len(a) < 2 or len(b) < 2:
            return 0.5
        tail = a[-min(6, len(a)):]
        head = b[:min(6, len(b))]
        last = np.asarray(tail[-1]["center"], np.float32)
        first = np.asarray(head[0]["center"], np.float32)
        ah = max(float(tail[-1].get("height", 1.0)), 1.0)
        gap = max(0.0, float(head[0]["timestamp"]) - float(tail[-1]["timestamp"]))
        if len(tail) >= 2:
            prev = np.asarray(tail[-2]["center"], np.float32)
            dt = max(float(tail[-1]["timestamp"]) - float(tail[-2]["timestamp"]), 1e-3)
            velocity = (last - prev) / dt
            predicted = last + velocity * min(gap, 1.0)
        else:
            predicted = last
        position = max(0.0, 1.0 - float(np.linalg.norm(predicted - first) / ah) / 6.0)
        oldv = last - np.asarray(tail[0]["center"], np.float32)
        newv = np.asarray(head[-1]["center"], np.float32) - first
        an = float(np.linalg.norm(oldv)); bn = float(np.linalg.norm(newv))
        direction = 0.5
        if an > 1e-3 and bn > 1e-3:
            direction = 0.5 + 0.5 * float(np.clip(np.dot(oldv, newv) / (an * bn), -1.0, 1.0))
        return float(0.70 * position + 0.30 * direction)

    @classmethod
    def _group_from_track(cls, key: str, track, local_gid: str) -> LocalGroup:
        group = super()._group_from_track(key, track, local_gid)
        setattr(group, "trajectory", list(getattr(track, "trajectory", []) or []))
        return group

    def pair(self, left, right, left_pool, right_pool):
        evidence = super().pair(left, right, left_pool, right_pool)
        if left.camera != right.camera:
            return evidence
        trajectory = self._trajectory_score(left, right)
        continuity = max(evidence.continuity, trajectory)
        geometry = max(evidence.geometry, trajectory)
        fused = max(evidence.fused, 0.79 * evidence.agreement + 0.08 * evidence.colour + 0.13 * continuity)
        return PairEvidence(
            min(0.99, float(fused)), evidence.resnet, evidence.swin, evidence.solider,
            evidence.colour, float(geometry), evidence.temporal, float(continuity),
            evidence.agreement, evidence.model_support, evidence.mutual_models,
            evidence.view_support, evidence.state_transition,
        )

    def _same_accept(self, evidence: PairEvidence) -> bool:
        if evidence.model_support < 2 or evidence.agreement < 0.50:
            return False
        if evidence.continuity < self.same_spatial_min:
            return False
        threshold = self.partial_min if evidence.state_transition else self.same_min
        return evidence.fused >= threshold

    def same_camera_edges(self, groups):
        ordered = sorted(groups, key=lambda group: (group.start, group.end, group.key))
        candidates = []
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                if right.start <= left.end:
                    continue
                if right.start - left.end > self.same_max_gap:
                    break
                evidence = self.pair(left, right, ordered, ordered)
                if not self._same_accept(evidence):
                    continue
                trajectory = self._trajectory_score(left, right)
                if trajectory < self.trajectory_min and evidence.fused < 0.68:
                    continue
                candidates.append((float(evidence.fused), float(trajectory), left, right, evidence))

        candidates.sort(key=lambda item: (item[0], item[1], item[4].agreement), reverse=True)
        incoming = set(); outgoing = set(); chosen = []
        for score, trajectory, left, right, evidence in candidates:
            if left.key in outgoing or right.key in incoming:
                continue
            outgoing.add(left.key); incoming.add(right.key)
            chosen.append({
                "left": left.key, "right": right.key, "fused": score,
                "resnet": evidence.resnet, "swin": evidence.swin, "solider": evidence.solider,
                "colour": evidence.colour, "geometry": evidence.geometry,
                "temporal": evidence.temporal, "continuity": evidence.continuity,
                "agreement": evidence.agreement, "model_support": evidence.model_support,
                "mutual_models": evidence.mutual_models, "view_support": evidence.view_support,
                "state_transition": evidence.state_transition, "trajectory": trajectory,
                "post_overlap_reassignment": True,
            })
        return chosen


__all__ = ["OverlapReidResolver"]
