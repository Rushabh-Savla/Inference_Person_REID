from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from rebuild.identity_v3 import Feature, Identity, Tracklet, unit


@dataclass(frozen=True)
class DecisionV6:
    key: str
    gid: str
    state: str
    reason: str
    score: float
    margin: float
    body: float
    face: float
    temporal: float
    spatial: float
    support: int
    camera: str
    provisional: str
    merged: bool = False


@dataclass
class EdgeV6:
    left: str
    right: str
    body: float
    face: float
    temporal: float
    spatial: float
    score: float
    support: int
    reason: str


class DSU:
    def __init__(self, keys: List[str]):
        self.parent = {key: key for key in keys}
        self.size = {key: 1 for key in keys}

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            nxt = self.parent[key]
            self.parent[key] = root
            key = nxt
        return root

    def union(self, left: str, right: str) -> str:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return left
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return left


class GlobalIdentityV6:
    """Persistent person identity layer built around provisional track hypotheses."""

    def __init__(self, cfg: dict):
        self.body_strong = float(cfg.get("body_strong", 0.70))
        self.body_medium = float(cfg.get("body_medium", 0.62))
        self.face_strong = float(cfg.get("face_strong", 0.84))
        self.face_confirm = float(cfg.get("face_confirm", 0.78))
        self.same_camera_gap = float(cfg.get("same_camera_gap_sec", 3.0))
        self.same_camera_distance = float(cfg.get("same_camera_distance", 4.5))
        self.same_camera_score = float(cfg.get("same_camera_score", 0.50))
        self.edge_margin = float(cfg.get("edge_margin", 0.04))
        self.gallery = int(cfg.get("gallery", 24))
        self.promote = float(cfg.get("promote_quality", 0.70))
        self.face_gallery = int(cfg.get("face_gallery", 8))
        self.face_novelty = float(cfg.get("face_novelty", 0.985))
        self.merge_face = float(cfg.get("merge_face", 0.90))
        self.identity_gap = float(cfg.get("identity_gap_sec", 60.0))
        self.min_cluster_support = int(cfg.get("min_cluster_support", 1))

        self.identities: Dict[str, Identity] = {}
        self.face_trusted: Dict[str, List[Feature]] = {}
        self.face_candidate: Dict[str, List[Feature]] = {}
        self.mapping: Dict[str, str] = {}
        self.provisional: Dict[str, str] = {}
        self.provisional_count = 0
        self.decisions: List[DecisionV6] = []
        self.edges: List[EdgeV6] = []
        self.next_id = 1
        self.identity_merges = 0
        self.reassociated = 0
        self.same_camera_reassociated = 0
        self.cross_camera_reidentified = 0
        self.face_assisted = 0
        self.body_assisted = 0
        self.temporal_assisted = 0

    def gid(self) -> str:
        value = f"G{self.next_id:06d}"
        self.next_id += 1
        return value

    @staticmethod
    def overlap(left: Tracklet, right: Tracklet) -> bool:
        return not (left.end < right.start or right.end < left.start)

    @staticmethod
    def _bbox_center(row: dict) -> Tuple[float, float]:
        box = row.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = map(float, box)
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @staticmethod
    def _bbox_height(row: dict) -> float:
        box = row.get("bbox") or [0, 0, 0, 0]
        return max(1.0, float(box[3]) - float(box[1]))

    @classmethod
    def endpoints(cls, track: Tracklet) -> Tuple[dict | None, dict | None]:
        if not track.observations:
            return None, None
        rows = sorted(track.observations, key=lambda x: float(x.get("timestamp", 0.0)))
        return rows[0], rows[-1]

    @staticmethod
    def _safe_pair(left: List[Feature], right: List[Feature]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        groups = {}
        for kind in {x.kind for x in left} | {x.kind for x in right}:
            a = [x for x in left if x.kind == kind]
            b = [x for x in right if x.kind == kind]
            if not a or not b:
                continue
            sims = np.stack([unit(x.vector) for x in a]) @ np.stack([unit(x.vector) for x in b]).T
            flat = np.sort(sims.reshape(-1))
            best = float(flat[-1])
            top = float(np.mean(flat[-min(3, len(flat)):]))
            groups[kind] = 0.72 * best + 0.28 * top, int((flat >= 0.63).sum())
        if not groups:
            return 0.0, 0
        full = groups.get("full") or groups.get("light")
        upper = groups.get("upper")
        lower = groups.get("lower")
        if full:
            score, support = full
            parts = [x[0] for x in (upper, lower) if x]
            if parts:
                score = 0.82 * score + 0.18 * max(parts)
                support += sum(groups[k][1] for k in ("upper", "lower") if k in groups)
            return float(score), int(support)
        parts = sorted([x for x in (upper, lower) if x], key=lambda x: x[0], reverse=True)
        if not parts:
            return 0.0, 0
        if len(parts) == 1:
            return float(parts[0][0]), int(parts[0][1])
        return float(0.85 * parts[0][0] + 0.15 * parts[1][0]), int(parts[0][1] + parts[1][1])

    @staticmethod
    def _face_pair(left: List[Feature], right: List[Feature]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        sims = np.stack([unit(x.vector) for x in left]) @ np.stack([unit(x.vector) for x in right]).T
        flat = np.sort(sims.reshape(-1))
        return float(0.72 * float(flat[-1]) + 0.28 * float(np.mean(flat[-min(3, len(flat)):]))), int((flat >= 0.55).sum())

    @classmethod
    def _continuity(cls, left: Tracklet, right: Tracklet) -> Tuple[float, float, float, str]:
        if left.camera != right.camera or cls.overlap(left, right):
            return 0.0, 0.0, 0.0, ""
        _, a1 = cls.endpoints(left)
        b0, _ = cls.endpoints(right)
        if a1 is None or b0 is None:
            return 0.0, 0.0, 0.0, ""
        if left.end <= right.start:
            gap = float(right.start - left.end)
            prev, nxt = a1, b0
        else:
            gap = float(left.start - right.end)
            prev, nxt = a1 if right.end <= left.start else b0, b0 if right.end <= left.start else a1
        if gap > 3.0:
            return 0.0, 0.0, gap, ""
        px, py = cls._bbox_center(prev)
        nx, ny = cls._bbox_center(nxt)
        ph = cls._bbox_height(prev)
        distance = float(np.hypot(nx - px, ny - py) / ph)
        temporal = max(0.0, 1.0 - gap / 3.0)
        spatial = max(0.0, 1.0 - distance / 4.5)
        continuity = 0.55 * temporal + 0.45 * spatial
        reason = "recent_lost_track" if continuity >= 0.35 else ""
        return float(temporal), float(spatial), gap, reason

    def pair_edge(self, left: Tracklet, right: Tracklet, faces: Dict[str, List[Feature]]) -> EdgeV6 | None:
        if left.key == right.key or (left.camera == right.camera and self.overlap(left, right)):
            return None
        body, support = self._safe_pair(left.features, right.features)
        face, face_support = self._face_pair(faces.get(left.key, []), faces.get(right.key, []))
        temporal, spatial, _, continuity_reason = self._continuity(left, right)

        score = body
        reason_parts = []
        if continuity_reason:
            reason_parts.append(continuity_reason)
        if face >= self.face_confirm:
            score = max(score, 0.70 * body + 0.30 * face)
            reason_parts.append("face")
        elif face >= self.face_strong:
            score = max(score, 0.65 * body + 0.35 * face)
            reason_parts.append("face")
        if temporal > 0.0 and not continuity_reason:
            score = max(score, 0.72 * body + 0.18 * spatial + 0.10 * temporal)
            reason_parts.append("temporal")

        accepted = (
            (face >= self.face_strong and face_support >= 1)
            or (body >= self.body_strong and support >= self.min_cluster_support)
            or (temporal > 0.0 and spatial > 0.45 and body >= self.same_camera_score)
            or (body >= self.body_medium and face >= self.face_confirm)
        )
        if not accepted:
            return None

        if continuity_reason:
            reason = "recent_lost_track"
        elif "face" in reason_parts:
            reason = "face_assisted"
        elif "temporal" in reason_parts:
            reason = "temporal_assisted"
        else:
            reason = "appearance"

        return EdgeV6(left.key, right.key, float(body), float(face), float(temporal), float(spatial), float(score), int(max(support, face_support)), reason)

    def build_edges(self, tracks: Dict[str, Tracklet], faces: Dict[str, List[Feature]]) -> None:
        values = sorted(tracks.values(), key=lambda x: (x.start, x.camera, x.key))
        self.edges = []
        for i, left in enumerate(values):
            for right in values[i + 1:]:
                edge = self.pair_edge(left, right, faces)
                if edge is not None:
                    self.edges.append(edge)
        self.edges.sort(key=lambda x: x.score, reverse=True)

    def _component_overlap(self, members: List[str], candidate: Tracklet, tracks: Dict[str, Tracklet]) -> bool:
        return any(
            tracks[key].camera == candidate.camera and self.overlap(tracks[key], candidate)
            for key in members
        )

    def _cluster(self, tracks: Dict[str, Tracklet]) -> Dict[str, List[str]]:
        keys = sorted(tracks)
        dsu = DSU(keys)
        members = {key: [key] for key in keys}
        for edge in self.edges:
            left_root, right_root = dsu.find(edge.left), dsu.find(edge.right)
            if left_root == right_root:
                continue
            left_members, right_members = members[left_root], members[right_root]
            if any(self._component_overlap(left_members, tracks[r], tracks) for r in right_members):
                continue
            strong_edge = edge.body >= self.body_strong or edge.face >= self.face_strong
            continuity_edge = edge.temporal >= 0.35 and edge.spatial >= 0.45 and edge.body >= self.same_camera_score
            if not (strong_edge or continuity_edge):
                continue
            root = dsu.union(left_root, right_root)
            if root == left_root:
                members[root] = left_members + right_members
                members.pop(right_root, None)
            else:
                members[root] = right_members + left_members
                members.pop(left_root, None)
        return {dsu.find(key): list(vals) for key, vals in members.items()}

    @staticmethod
    def _canonical(members: List[str], tracks: Dict[str, Tracklet]) -> str:
        return min((tracks[key] for key in members), key=lambda x: (x.start, x.camera, x.key)).key

    def _add_identity(self, gid: str, track: Tracklet, trusted: bool) -> None:
        self.identities.setdefault(gid, Identity(gid)).add(track, trusted, self.gallery, self.promote)

    def run(self, tracks: Dict[str, Tracklet], faces: Dict[str, List[Feature]]):
        self.build_edges(tracks, faces)
        components = self._cluster(tracks)
        self.provisional_count = len(components)
        for index, members in enumerate(sorted(components.values(), key=lambda xs: min((tracks[x].start, tracks[x].camera, x) for x in xs)), 1):
            provisional = f"P{index:04d}"
            gid = self.gid()
            self.provisional[provisional] = gid
            canonical_key = self._canonical(members, tracks)
            self._add_identity(gid, tracks[canonical_key], trusted=True)
            for key in members:
                if key != canonical_key:
                    self._add_identity(gid, tracks[key], trusted=True)
                self.mapping[key] = gid
                self.provisional[key] = provisional

            for key in members:
                reason = "new_identity"
                body = face = temporal = spatial = 0.0
                support = 0
                matched = [
                    e for e in self.edges
                    if (e.left == key and e.right in members) or (e.right == key and e.left in members)
                ]
                if matched:
                    best = max(matched, key=lambda e: e.score)
                    reason = best.reason
                    body, face, temporal, spatial, support = best.body, best.face, best.temporal, best.spatial, best.support
                    if "face" in reason:
                        self.face_assisted += 1
                    if "temporal" in reason or reason == "recent_lost_track":
                        self.temporal_assisted += 1
                    if "appearance" in reason or best.body >= self.body_medium:
                        self.body_assisted += 1
                    if reason == "recent_lost_track":
                        self.reassociated += 1
                        self.same_camera_reassociated += 1
                    elif tracks[key].camera not in {tracks[x].camera for x in members if x != key}:
                        self.cross_camera_reidentified += 1
                    state = "confirmed_existing"
                else:
                    state = "new_person"
                self.decisions.append(DecisionV6(key, gid, state, reason, max(body, face), 0.0, body, face, temporal, spatial, support, tracks[key].camera, provisional, False))

        self.identity_merges = sum(max(0, len(members) - 1) for members in components.values())
        return dict(self.mapping), list(self.decisions)

    def summary(self, tracks: Dict[str, Tracklet]) -> dict:
        groups: Dict[str, set] = {}
        group_tracks: Dict[str, list] = {}
        for key, gid in self.mapping.items():
            groups.setdefault(gid, set()).add(tracks[key].camera)
            group_tracks.setdefault(gid, []).append(key)
        multi = {gid: sorted(cams) for gid, cams in groups.items() if len(cams) > 1}
        fragmentation = {gid: sorted(keys) for gid, keys in group_tracks.items() if len(keys) > 1}
        reasons: Dict[str, int] = {}
        for item in self.decisions:
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
        return {
            "tracklets": len(tracks),
            "global_ids": len(set(self.mapping.values())),
            "multi_camera": multi,
            "multi_camera_count": len(multi),
            "reasons": reasons,
            "new_identities": sum(x.state == "new_person" for x in self.decisions),
            "reidentified": sum(x.state == "confirmed_existing" for x in self.decisions),
            "same_camera_reassociations": self.same_camera_reassociated,
            "recent_lost_track_reassociations": self.reassociated,
            "cross_camera_reidentifications": self.cross_camera_reidentified,
            "identity_merges": self.identity_merges,
            "provisional_identities": self.provisional_count,
            "fragmented_identities": fragmentation,
            "fragmented_identity_count": len(fragmentation),
            "face_assisted": self.face_assisted,
            "body_assisted": self.body_assisted,
            "temporal_assisted": self.temporal_assisted,
            "edge_count": len(self.edges),
            "trusted_body": sum(len(x.trusted) for x in self.identities.values()),
            "candidate_body": sum(len(x.candidate) for x in self.identities.values()),
            "trusted_face": sum(len(x) for x in self.face_trusted.values()),
            "candidate_face": sum(len(x) for x in self.face_candidate.values()),
        }
