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


@dataclass(frozen=True)
class EdgeV6:
    left: str
    right: str
    body: float
    face: float
    temporal: float
    spatial: float
    continuity: float
    score: float
    support: int
    same_camera: bool
    reason: str


class DSU:
    def __init__(self, keys: List[str]):
        self.parent = {x: x for x in keys}
        self.size = {x: 1 for x in keys}

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
    """Persistent identity layer with explicit track-fragment reassociation.

    Tracklets are observations, not people. V6 first builds a graph of reliable
    same-camera fragment links and strict camera-agnostic appearance links. It then
    forms provisional identity components and creates canonical Global IDs only after
    those relationships have been resolved.
    """

    def __init__(self, cfg: dict):
        self.body_strong = float(cfg.get("body_strong", 0.68))
        self.body_medium = float(cfg.get("body_medium", 0.60))
        self.same_camera_body = float(cfg.get("same_camera_body", 0.46))
        self.face_strong = float(cfg.get("face_strong", 0.84))
        self.face_medium = float(cfg.get("face_medium", 0.76))
        self.face_support = int(cfg.get("face_support", 1))
        self.same_camera_gap = float(cfg.get("same_camera_gap_sec", 12.0))
        self.same_camera_overlap = float(cfg.get("same_camera_overlap_sec", 0.75))
        self.same_camera_distance = float(cfg.get("same_camera_distance", 5.0))
        self.same_camera_min_continuity = float(cfg.get("same_camera_min_continuity", 0.35))
        self.gallery = int(cfg.get("gallery", 32))
        self.promote = float(cfg.get("promote_quality", 0.68))
        self.face_gallery = int(cfg.get("face_gallery", 10))
        self.face_novelty = float(cfg.get("face_novelty", 0.985))
        self.cross_margin = float(cfg.get("cross_margin", 0.035))
        self.cluster_min_edge = float(cfg.get("cluster_min_edge", 0.60))

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
    def _center(row: dict) -> Tuple[float, float]:
        box = row.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = map(float, box)
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @staticmethod
    def _height(row: dict) -> float:
        box = row.get("bbox") or [0, 0, 0, 0]
        return max(1.0, float(box[3]) - float(box[1]))

    @staticmethod
    def _width(row: dict) -> float:
        box = row.get("bbox") or [0, 0, 0, 0]
        return max(1.0, float(box[2]) - float(box[0]))

    @classmethod
    def _endpoint_rows(cls, track: Tracklet) -> Tuple[dict | None, dict | None]:
        if not track.observations:
            return None, None
        rows = sorted(track.observations, key=lambda x: float(x.get("timestamp", 0.0)))
        return rows[0], rows[-1]

    @staticmethod
    def _same_kind(left: List[Feature], right: List[Feature], kind: str) -> Tuple[float, int]:
        a = [unit(x.vector) for x in left if x.kind == kind]
        b = [unit(x.vector) for x in right if x.kind == kind]
        if not a or not b:
            return 0.0, 0
        sims = np.stack(a) @ np.stack(b).T
        flat = np.sort(sims.reshape(-1))
        k = min(3, len(flat))
        best = float(flat[-1])
        top = float(np.mean(flat[-k:]))
        support = int((flat >= 0.60).sum())
        return float(0.72 * best + 0.28 * top), support

    @classmethod
    def appearance(cls, left: List[Feature], right: List[Feature]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        kinds = {x.kind for x in left} | {x.kind for x in right}
        scores: Dict[str, Tuple[float, int]] = {}
        for kind in kinds:
            score, support = cls._same_kind(left, right, kind)
            if score > 0:
                scores[kind] = (score, support)
        if not scores:
            return 0.0, 0

        weights = {"full": 0.50, "light": 0.15, "upper": 0.22, "lower": 0.13}
        ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
        numerator = 0.0
        total = 0.0
        for index, (kind, (score, _)) in enumerate(ranked[:3]):
            weight = weights.get(kind, 0.10 if index == 0 else 0.05)
            numerator += weight * score
            total += weight
        return float(numerator / max(total, 1e-8)), int(sum(v[1] for _, v in ranked[:3]))

    @staticmethod
    def face_pair(left: List[Feature], right: List[Feature]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        a = np.stack([unit(x.vector) for x in left])
        b = np.stack([unit(x.vector) for x in right])
        sims = a @ b.T
        flat = np.sort(sims.reshape(-1))
        k = min(3, len(flat))
        return float(0.72 * flat[-1] + 0.28 * np.mean(flat[-k:])), int((flat >= 0.58).sum())

    def _continuity_pair(self, left: Tracklet, right: Tracklet) -> Tuple[float, float, float, bool]:
        l0, l1 = self._endpoint_rows(left)
        r0, r1 = self._endpoint_rows(right)
        if l0 is None or l1 is None or r0 is None or r1 is None:
            return 0.0, 0.0, 0.0, False
        if left.end <= right.start:
            gap = float(right.start - left.end)
            prev, nxt = l1, r0
        elif right.end <= left.start:
            gap = float(left.start - right.end)
            prev, nxt = r1, l0
        else:
            gap = 0.0
            prev, nxt = (l1, r0) if left.end <= right.end else (r1, l0)
        if gap > self.same_camera_gap:
            return 0.0, 0.0, gap, False
        px, py = self._center(prev)
        nx, ny = self._center(nxt)
        ph = self._height(prev)
        pw = self._width(prev)
        nh = self._height(nxt)
        nw = self._width(nxt)
        distance = float(np.hypot(nx - px, ny - py) / ph)
        spatial = max(0.0, 1.0 - distance / self.same_camera_distance)
        scale_ratio = min(ph, nh) / max(ph, nh)
        width_ratio = min(pw, nw) / max(pw, nw)
        scale = 0.5 * scale_ratio + 0.5 * width_ratio
        temporal = 1.0 if gap == 0 else max(0.0, 1.0 - gap / self.same_camera_gap)
        continuity = 0.45 * temporal + 0.40 * spatial + 0.15 * scale
        return float(continuity), float(spatial), gap, self.overlap(left, right)

    @staticmethod
    def endpoint_iou(left: Tracklet, right: Tracklet) -> float:
        rows_left = sorted(left.observations, key=lambda x: float(x.get("timestamp", 0.0)))
        rows_right = sorted(right.observations, key=lambda x: float(x.get("timestamp", 0.0)))
        if not rows_left or not rows_right:
            return 0.0
        best = 0.0
        for a, b in ((rows_left[-1], rows_right[0]), (rows_right[-1], rows_left[0])):
            ax1, ay1, ax2, ay2 = map(float, a.get("bbox") or [0, 0, 0, 0])
            bx1, by1, bx2, by2 = map(float, b.get("bbox") or [0, 0, 0, 0])
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            if union > 0:
                best = max(best, inter / union)
        return float(best)

    def pair_edge(self, left: Tracklet, right: Tracklet, faces: Dict[str, List[Feature]]) -> EdgeV6 | None:
        if left.key == right.key:
            return None
        same_camera = left.camera == right.camera
        body, support = self.appearance(left.features, right.features)
        face, face_support = self.face_pair(faces.get(left.key, []), faces.get(right.key, []))
        continuity = spatial = 0.0
        overlap = False
        if same_camera:
            continuity, spatial, _, overlap = self._continuity_pair(left, right)
        iou = self.endpoint_iou(left, right) if same_camera else 0.0
        reasons: List[str] = []
        accepted = False
        score = body

        if face >= self.face_strong and face_support >= self.face_support:
            score = max(score, 0.65 * body + 0.35 * face)
            reasons.append("face")
        elif face >= self.face_medium and body >= self.body_medium:
            score = max(score, 0.70 * body + 0.30 * face)
            reasons.append("face_support")

        if same_camera:
            strong_cont = continuity >= 0.60 and spatial >= 0.55
            moderate_cont = continuity >= self.same_camera_min_continuity and spatial >= 0.45
            overlap_ok = overlap and (iou >= self.same_camera_overlap or spatial >= 0.65)
            if strong_cont and body >= self.same_camera_body:
                accepted = True
                reasons.append("recent_lost_track")
            elif moderate_cont and body >= 0.55 and support >= 1:
                accepted = True
                reasons.append("temporal_spatial_reassociation")
            elif overlap_ok and body >= 0.52:
                accepted = True
                reasons.append("fragment_overlap")
            elif face >= self.face_strong and spatial >= 0.45:
                accepted = True
                reasons.append("face_reassociation")
            elif body >= self.body_strong and support >= 2:
                accepted = True
                reasons.append("same_camera_appearance")
        else:
            if face >= self.face_strong and face_support >= self.face_support:
                accepted = True
                reasons.append("cross_camera_face")
            elif body >= self.body_strong and support >= 2:
                accepted = True
                reasons.append("cross_camera_body")
            elif body >= self.body_medium and support >= 4:
                accepted = True
                reasons.append("cross_camera_multi_observation")

        if not accepted:
            return None

        if face >= self.face_strong:
            score = max(score, face)
        elif same_camera and continuity > 0:
            score = max(score, 0.66 * body + 0.24 * continuity + 0.10 * spatial)

        return EdgeV6(
            left.key,
            right.key,
            float(body),
            float(face),
            float(continuity),
            float(spatial),
            float(continuity),
            float(score),
            int(max(support, face_support)),
            same_camera,
            "+".join(reasons),
        )

    @staticmethod
    def _overlap_conflict(a: Tracklet, b: Tracklet) -> bool:
        if a.camera != b.camera:
            return False
        start = max(a.start, b.start)
        end = min(a.end, b.end)
        return end - start > 0.75

    def _component_conflict(self, left_members: List[str], right_members: List[str], tracks: Dict[str, Tracklet]) -> bool:
        return any(self._overlap_conflict(tracks[a], tracks[b]) for a in left_members for b in right_members)

    def build_edges(self, tracks: Dict[str, Tracklet], faces: Dict[str, List[Feature]]) -> None:
        values = sorted(tracks.values(), key=lambda x: (x.start, x.camera, x.key))
        raw: List[EdgeV6] = []
        for i, left in enumerate(values):
            for right in values[i + 1:]:
                edge = self.pair_edge(left, right, faces)
                if edge is not None:
                    raw.append(edge)

        by_key: Dict[str, List[EdgeV6]] = {key: [] for key in tracks}
        for edge in raw:
            by_key[edge.left].append(edge)
            by_key[edge.right].append(edge)

        kept: List[EdgeV6] = []
        for edge in raw:
            if edge.same_camera:
                kept.append(edge)
                continue
            left_candidates = sorted(by_key[edge.left], key=lambda x: x.score, reverse=True)
            right_candidates = sorted(by_key[edge.right], key=lambda x: x.score, reverse=True)
            left_best = left_candidates[0].score if left_candidates else 0.0
            right_best = right_candidates[0].score if right_candidates else 0.0
            left_margin = left_best - (left_candidates[1].score if len(left_candidates) > 1 else 0.0)
            right_margin = right_best - (right_candidates[1].score if len(right_candidates) > 1 else 0.0)
            if edge.score + 1e-6 >= left_best and edge.score + 1e-6 >= right_best and max(left_margin, right_margin) >= self.cross_margin:
                kept.append(edge)

        self.edges = sorted(kept, key=lambda x: x.score, reverse=True)

    def _cluster(self, tracks: Dict[str, Tracklet]) -> Dict[str, List[str]]:
        keys = sorted(tracks)
        dsu = DSU(keys)
        members: Dict[str, List[str]] = {key: [key] for key in keys}
        for edge in self.edges:
            left_root = dsu.find(edge.left)
            right_root = dsu.find(edge.right)
            if left_root == right_root:
                continue
            left_members = members[left_root]
            right_members = members[right_root]
            if self._component_conflict(left_members, right_members, tracks):
                continue
            if edge.score < self.cluster_min_edge and not edge.same_camera:
                continue
            root = dsu.union(left_root, right_root)
            if root == left_root:
                members[root] = left_members + right_members
                members.pop(right_root, None)
            else:
                members[root] = right_members + left_members
                members.pop(left_root, None)
        return {dsu.find(k): sorted(v) for k, v in members.items()}

    @staticmethod
    def _canonical(members: List[str], tracks: Dict[str, Tracklet]) -> str:
        return min((tracks[key] for key in members), key=lambda x: (x.start, x.camera, x.key)).key

    def _add_identity(self, gid: str, track: Tracklet, trusted: bool) -> None:
        self.identities.setdefault(gid, Identity(gid)).add(track, trusted, self.gallery, self.promote)

    def _add_face(self, gid: str, values: List[Feature], trusted: bool) -> None:
        if not values:
            return
        target = self.face_trusted if trusted else self.face_candidate
        bank = target.setdefault(gid, [])
        for item in values:
            if trusted and item.quality < self.promote:
                continue
            if bank:
                best = float(np.max(np.stack([x.vector for x in bank]) @ unit(item.vector)))
                if best >= self.face_novelty and item.quality <= max(x.quality for x in bank) + 0.02:
                    continue
            bank.append(item)
        bank.sort(key=lambda x: x.quality, reverse=True)
        del bank[self.face_gallery if trusted else max(4, self.face_gallery // 2):]

    def _edge_for_key(self, key: str, members: List[str]) -> EdgeV6 | None:
        pool = [e for e in self.edges if (e.left == key and e.right in members) or (e.right == key and e.left in members)]
        return max(pool, key=lambda x: x.score) if pool else None

    def run(self, tracks: Dict[str, Tracklet], faces: Dict[str, List[Feature]]):
        self.build_edges(tracks, faces)
        components = self._cluster(tracks)
        self.provisional_count = len(components)
        ordered = sorted(components.values(), key=lambda xs: min((tracks[k].start, tracks[k].camera, k) for k in xs))

        for index, members in enumerate(ordered, 1):
            provisional = f"P{index:04d}"
            gid = self.gid()
            canonical = self._canonical(members, tracks)
            self.provisional[provisional] = gid
            self._add_identity(gid, tracks[canonical], trusted=True)
            for key in members:
                if key != canonical:
                    self._add_identity(gid, tracks[key], trusted=True)
                self.mapping[key] = gid
                self.provisional[key] = provisional

            if len(members) > 1:
                self.identity_merges += len(members) - 1

            for key in members:
                edge = self._edge_for_key(key, members)
                if edge is None:
                    state = "new_person"
                    reason = "new_identity"
                    body = face = temporal = spatial = score = margin = 0.0
                    support = 0
                    merged = False
                else:
                    state = "confirmed_existing"
                    reason = edge.reason
                    body = edge.body
                    face = edge.face
                    temporal = edge.temporal
                    spatial = edge.spatial
                    score = edge.score
                    support = edge.support
                    merged = len(members) > 1
                    if "face" in reason:
                        self.face_assisted += 1
                    if "body" in reason or body >= self.body_medium:
                        self.body_assisted += 1
                    if temporal > 0:
                        self.temporal_assisted += 1
                    if edge.same_camera and ("reassociation" in reason or "fragment" in reason or "recent_lost_track" in reason):
                        self.reassociated += 1
                        self.same_camera_reassociated += 1
                    if not edge.same_camera and "cross_camera" in reason:
                        self.cross_camera_reidentified += 1
                    candidates = sorted([e.score for e in self.edges if e.left == key or e.right == key], reverse=True)
                    margin = score - (candidates[1] if len(candidates) > 1 else 0.0)

                self.decisions.append(
                    DecisionV6(
                        key=key,
                        gid=gid,
                        state=state,
                        reason=reason,
                        score=float(score),
                        margin=float(margin),
                        body=float(body),
                        face=float(face),
                        temporal=float(temporal),
                        spatial=float(spatial),
                        support=int(support),
                        camera=tracks[key].camera,
                        provisional=provisional,
                        merged=merged,
                    )
                )
                self._add_face(gid, faces.get(key, []), trusted=(state == "confirmed_existing" or len(members) > 1))

        return dict(self.mapping), list(self.decisions)

    def summary(self, tracks: Dict[str, Tracklet]) -> dict:
        groups: Dict[str, set] = {}
        group_tracks: Dict[str, list] = {}
        for key, gid in self.mapping.items():
            groups.setdefault(gid, set()).add(tracks[key].camera)
            group_tracks.setdefault(gid, []).append(key)
        multi = {gid: sorted(cams) for gid, cams in groups.items() if len(cams) > 1}
        fragmented = {gid: sorted(keys) for gid, keys in group_tracks.items() if len(keys) > 1}
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
            "fragmented_identities": fragmented,
            "fragmented_identity_count": len(fragmented),
            "face_assisted": self.face_assisted,
            "body_assisted": self.body_assisted,
            "temporal_assisted": self.temporal_assisted,
            "edge_count": len(self.edges),
            "trusted_body": sum(len(x.trusted) for x in self.identities.values()),
            "candidate_body": sum(len(x.candidate) for x in self.identities.values()),
            "trusted_face": sum(len(x) for x in self.face_trusted.values()),
            "candidate_face": sum(len(x) for x in self.face_candidate.values()),
        }
