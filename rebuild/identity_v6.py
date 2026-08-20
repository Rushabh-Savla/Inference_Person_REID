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
        left, right = self.find(left), self.find(right)
        if left == right:
            return left
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return left


class GlobalIdentityV6:
    """Persistent global identity association.

    A tracklet is an observation, never an identity. V6 keeps the useful V5
    gallery idea, adds explicit same-camera fragment reassociation, delays new
    identity creation for ambiguous observations, and searches Global IDs
    directly for cross-camera recovery instead of requiring a global mutual-best
    tracklet edge.
    """

    def __init__(self, cfg: dict):
        self.body_strong = float(cfg.get("body_strong", 0.70))
        self.body_medium = float(cfg.get("body_medium", 0.61))
        self.partial_strong = float(cfg.get("partial_strong", 0.66))
        self.same_camera_body = float(cfg.get("same_camera_body", 0.48))
        self.face_strong = float(cfg.get("face_strong", 0.84))
        self.face_medium = float(cfg.get("face_medium", 0.76))
        self.same_camera_gap = float(cfg.get("same_camera_gap_sec", 15.0))
        self.same_camera_distance = float(cfg.get("same_camera_distance", 5.0))
        self.same_camera_min = float(cfg.get("same_camera_min_continuity", 0.34))
        self.gallery = int(cfg.get("gallery", 32))
        self.promote = float(cfg.get("promote_quality", 0.68))
        self.face_gallery = int(cfg.get("face_gallery", 10))
        self.face_novelty = float(cfg.get("face_novelty", 0.985))
        self.cross_margin = float(cfg.get("cross_margin", 0.025))
        self.merge_body = float(cfg.get("merge_body", 0.80))
        self.merge_face = float(cfg.get("merge_face", 0.90))
        self.merge_support = int(cfg.get("merge_support", 3))

        self.identities: Dict[str, Identity] = {}
        self.face_trusted: Dict[str, List[Feature]] = {}
        self.face_candidate: Dict[str, List[Feature]] = {}
        self.mapping: Dict[str, str] = {}
        self.provisional: Dict[str, str] = {}
        self.decisions: List[DecisionV6] = []
        self.edges: List[EdgeV6] = []

        self.next_id = 1
        self.provisional_count = 0
        self.identity_merges = 0
        self.reassociated = 0
        self.same_camera_reassociated = 0
        self.cross_camera_reidentified = 0
        self.face_assisted = 0
        self.body_assisted = 0
        self.temporal_assisted = 0
        self.face_observations = 0
        self.face_high_quality = 0
        self.face_matches = 0
        self.face_conflicts = 0
        self.pending_count = 0

    def gid(self) -> str:
        value = f"G{self.next_id:06d}"
        self.next_id += 1
        return value

    @staticmethod
    def overlap(left: Tracklet, right: Tracklet) -> bool:
        return not (left.end < right.start or right.end < left.start)

    @staticmethod
    def _endpoints(track: Tracklet) -> Tuple[dict | None, dict | None]:
        if not track.observations:
            return None, None
        rows = sorted(track.observations, key=lambda x: float(x.get("timestamp", 0.0)))
        return rows[0], rows[-1]

    @staticmethod
    def _center(row: dict) -> Tuple[float, float]:
        x1, y1, x2, y2 = map(float, row.get("bbox") or [0, 0, 0, 0])
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @staticmethod
    def _height(row: dict) -> float:
        box = row.get("bbox") or [0, 0, 0, 0]
        return max(1.0, float(box[3]) - float(box[1]))

    @staticmethod
    def _width(row: dict) -> float:
        box = row.get("bbox") or [0, 0, 0, 0]
        return max(1.0, float(box[2]) - float(box[0]))

    @staticmethod
    def _kind_pair(left: List[Feature], right: List[Feature], kind_left: str, kind_right: str) -> Tuple[float, int]:
        a = [unit(x.vector) for x in left if x.kind == kind_left]
        b = [unit(x.vector) for x in right if x.kind == kind_right]
        if not a or not b:
            return 0.0, 0
        sims = np.stack(a) @ np.stack(b).T
        flat = np.sort(sims.reshape(-1))
        k = min(3, len(flat))
        return float(0.72 * flat[-1] + 0.28 * np.mean(flat[-k:])), int((flat >= 0.60).sum())

    @classmethod
    def appearance(cls, left: List[Feature], right: List[Feature]) -> Tuple[float, int, bool]:
        if not left or not right:
            return 0.0, 0, False

        pairs = [
            ("full", "full", 0.46),
            ("upper", "upper", 0.28),
            ("lower", "lower", 0.16),
            ("upper", "full", 0.08),
            ("full", "upper", 0.08),
            ("lower", "full", 0.06),
            ("full", "lower", 0.06),
            ("light", "full", 0.05),
            ("full", "light", 0.05),
        ]
        values = []
        partial = False
        for left_kind, right_kind, weight in pairs:
            score, support = cls._kind_pair(left, right, left_kind, right_kind)
            if score <= 0:
                continue
            values.append((score, support, weight, left_kind, right_kind))
            if "upper" in (left_kind, right_kind) or "lower" in (left_kind, right_kind):
                partial = True

        if not values:
            return 0.0, 0, False

        values.sort(key=lambda x: x[0], reverse=True)
        top = values[:4]
        total_weight = sum(x[2] for x in top)
        score = sum(x[0] * x[2] for x in top) / max(total_weight, 1e-8)
        support = sum(x[1] for x in top)
        return float(score), int(support), partial

    @staticmethod
    def face_pair(left: List[Feature], right: List[Feature]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        a = np.stack([unit(x.vector) for x in left])
        b = np.stack([unit(x.vector) for x in right])
        flat = np.sort((a @ b.T).reshape(-1))
        k = min(3, len(flat))
        return float(0.72 * flat[-1] + 0.28 * np.mean(flat[-k:])), int((flat >= 0.58).sum())

    @classmethod
    def continuity(cls, left: Tracklet, right: Tracklet, gap_limit: float, distance_limit: float) -> Tuple[float, float, float]:
        if left.camera != right.camera or cls.overlap(left, right):
            return 0.0, 0.0, 0.0
        _, left_end = cls._endpoints(left)
        right_start, _ = cls._endpoints(right)
        if left_end is None or right_start is None:
            return 0.0, 0.0, 0.0
        gap = abs(float(right.start - left.end))
        if gap > gap_limit:
            return 0.0, 0.0, gap
        px, py = cls._center(left_end)
        nx, ny = cls._center(right_start)
        ph = cls._height(left_end)
        rh = cls._height(right_start)
        pw = cls._width(left_end)
        rw = cls._width(right_start)
        distance = float(np.hypot(nx - px, ny - py) / ph)
        spatial = max(0.0, 1.0 - distance / max(distance_limit, 1e-6))
        scale = 0.5 * (min(ph, rh) / max(ph, rh)) + 0.5 * (min(pw, rw) / max(pw, rw))
        temporal = max(0.0, 1.0 - gap / gap_limit)
        score = 0.48 * temporal + 0.37 * spatial + 0.15 * scale
        return float(score), float(spatial), gap

    def _identity_body(self, track: Tracklet, identity: Identity) -> Tuple[float, int, bool]:
        return self.appearance(track.features, identity.values())

    def _identity_face(self, track_key: str, gid: str, faces: Dict[str, List[Feature]]) -> Tuple[float, int]:
        return self.face_pair(faces.get(track_key, []), self.face_trusted.get(gid, []))

    @staticmethod
    def _camera_conflict(track: Tracklet, identity: Identity, tracks: Dict[str, Tracklet]) -> bool:
        if track.camera not in identity.cameras:
            return False
        for key in identity.tracks:
            other = tracks.get(key)
            if other is not None and other.camera == track.camera and GlobalIdentityV6.overlap(track, other):
                return True
        return False

    def _rank_identity(self, track: Tracklet, tracks: Dict[str, Tracklet], faces: Dict[str, List[Feature]]) -> List[dict]:
        ranked: List[dict] = []
        for gid, identity in self.identities.items():
            if self._camera_conflict(track, identity, tracks):
                continue
            body, support, partial = self._identity_body(track, identity)
            face, face_support = self._identity_face(track.key, gid, faces)
            same_support = 0.0
            same_spatial = 0.0
            same_gap = 0.0
            if track.camera in identity.cameras and identity.tracks:
                best_cont = 0.0
                for old_key in identity.tracks:
                    old = tracks.get(old_key)
                    if old is None:
                        continue
                    cont, spatial, gap = self.continuity(old, track, self.same_camera_gap, self.same_camera_distance)
                    if cont > best_cont:
                        best_cont, same_spatial, same_gap = cont, spatial, gap
                same_support = best_cont

            score = body
            reason = "body"
            if face >= self.face_strong:
                score = max(score, 0.60 * body + 0.40 * face)
                reason = "face+body"
            elif face >= self.face_medium and body >= self.body_medium:
                score = max(score, 0.72 * body + 0.28 * face)
                reason = "face_support"
            if same_support > 0:
                rescue = 0.68 * body + 0.22 * same_support + 0.10 * same_spatial
                if rescue > score:
                    score = rescue
                    reason = "same_camera_reassociation"

            ranked.append({
                "gid": gid,
                "score": float(score),
                "body": float(body),
                "face": float(face),
                "support": int(max(support, face_support)),
                "partial": bool(partial),
                "temporal": float(same_support),
                "spatial": float(same_spatial),
                "gap": float(same_gap),
                "reason": reason,
            })
        ranked.sort(key=lambda x: (x["score"], x["support"], x["body"], x["face"]), reverse=True)
        return ranked

    def _accept_existing(self, row: dict, second: float, track: Tracklet) -> bool:
        margin = float(row["score"] - second)
        score = row["score"]
        face = row["face"]
        body = row["body"]
        support = row["support"]
        temporal = row["temporal"]
        partial = row["partial"]
        if face >= self.face_strong and row["face"] >= self.face_medium:
            return True
        if temporal >= self.same_camera_min and body >= self.same_camera_body and margin >= 0.02:
            return True
        if partial and score >= self.partial_strong and support >= 2 and margin >= self.cross_margin:
            return True
        if score >= self.body_strong:
            return margin >= self.cross_margin or support >= 3
        return score >= self.body_medium and support >= 3 and margin >= max(self.cross_margin, 0.03)

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
        limit = self.face_gallery if trusted else max(4, self.face_gallery // 2)
        del bank[limit:]

    def _add_track(self, gid: str, track: Tracklet, trusted: bool) -> None:
        self.identities.setdefault(gid, Identity(gid)).add(track, trusted, self.gallery, self.promote)

    def _new_provisional(self, track: Tracklet, index: int) -> str:
        key = f"P{index:04d}"
        self.provisional[key] = track.key
        return key

    def _fragment_candidate(self, track: Tracklet, assigned: Dict[str, str], tracks: Dict[str, Tracklet], faces: Dict[str, List[Feature]]) -> Tuple[str | None, dict | None]:
        candidates = []
        for key, gid in assigned.items():
            other = tracks[key]
            if other.camera != track.camera or self.overlap(other, track):
                continue
            body, support, partial = self.appearance(other.features, track.features)
            face, face_support = self.face_pair(faces.get(key, []), faces.get(track.key, []))
            cont, spatial, gap = self.continuity(other, track, self.same_camera_gap, self.same_camera_distance)
            score = body
            reason = "fragment_appearance"
            if cont >= self.same_camera_min:
                score = max(score, 0.66 * body + 0.24 * cont + 0.10 * spatial)
                reason = "recent_lost_track"
            if face >= self.face_strong:
                score = max(score, face)
                reason = "face_reassociation"
            candidates.append({"gid": gid, "score": score, "body": body, "face": face, "support": max(support, face_support), "temporal": cont, "spatial": spatial, "gap": gap, "partial": partial, "reason": reason})
        candidates.sort(key=lambda x: (x["score"], x["support"]), reverse=True)
        if not candidates:
            return None, None
        best = candidates[0]
        second = candidates[1]["score"] if len(candidates) > 1 else 0.0
        if best["score"] >= self.partial_strong and best["score"] - second >= self.cross_margin:
            return best["gid"], best
        return None, best

    def _merge_identities(self, tracks: Dict[str, Tracklet], faces: Dict[str, List[Feature]]) -> None:
        gids = list(self.identities)
        changed = True
        while changed:
            changed = False
            gids = list(self.identities)
            for i, left_gid in enumerate(gids):
                for right_gid in gids[i + 1:]:
                    if left_gid not in self.identities or right_gid not in self.identities:
                        continue
                    left = self.identities[left_gid]
                    right = self.identities[right_gid]
                    if self._identity_overlap(left, right, tracks):
                        continue
                    body, support, partial = self.appearance(left.values(), right.values())
                    face = self._gallery_face_similarity(left_gid, right_gid)
                    if face >= self.merge_face or (body >= self.merge_body and support >= self.merge_support):
                        self._merge_pair(left_gid, right_gid)
                        self.identity_merges += 1
                        changed = True
                        break
                if changed:
                    break

    @staticmethod
    def _identity_overlap(left: Identity, right: Identity, tracks: Dict[str, Tracklet]) -> bool:
        for a in left.tracks:
            ta = tracks.get(a)
            if ta is None:
                continue
            for b in right.tracks:
                tb = tracks.get(b)
                if tb is None:
                    continue
                if GlobalIdentityV6.overlap(ta, tb) and ta.camera == tb.camera:
                    return True
        return False

    def _gallery_face_similarity(self, left_gid: str, right_gid: str) -> float:
        a = self.face_trusted.get(left_gid, [])
        b = self.face_trusted.get(right_gid, [])
        return self.face_pair(a, b)[0]

    def _merge_pair(self, left_gid: str, right_gid: str) -> None:
        left = self.identities[left_gid]
        right = self.identities[right_gid]
        canonical, other = (left_gid, right_gid) if int(left_gid[1:]) < int(right_gid[1:]) else (right_gid, left_gid)
        target = self.identities[canonical]
        source = self.identities[other]
        target.add(Tracklet(source.gid, 0, 0, 1.0), False, self.gallery, 0.0) if False else None
        for feature in source.trusted:
            if not any(feature.kind == x.kind and float(np.dot(unit(feature.vector), unit(x.vector))) >= 0.985 for x in target.trusted):
                target.trusted.append(feature)
        for feature in source.candidate:
            target.candidate.append(feature)
        target.trusted = target.trim(target.trusted, self.gallery)
        target.candidate = target.trim(target.candidate, max(6, self.gallery // 2))
        target.tracks = list(dict.fromkeys(target.tracks + source.tracks))
        target.cameras.update(source.cameras)
        self.face_trusted.setdefault(canonical, []).extend(self.face_trusted.get(other, []))
        self.face_candidate.setdefault(canonical, []).extend(self.face_candidate.get(other, []))
        self.face_trusted[canonical] = sorted(self.face_trusted[canonical], key=lambda x: x.quality, reverse=True)[: self.face_gallery]
        self.face_candidate[canonical] = sorted(self.face_candidate[canonical], key=lambda x: x.quality, reverse=True)[: max(4, self.face_gallery // 2)]
        for key, gid in list(self.mapping.items()):
            if gid == other:
                self.mapping[key] = canonical
        self.identities.pop(other, None)

    def _record(self, track: Tracklet, gid: str, state: str, reason: str, row: dict | None, provisional: str, merged: bool = False) -> None:
        if row is None:
            self.decisions.append(DecisionV6(track.key, gid, state, reason, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, track.camera, provisional, merged))
            return
        second = float(row.get("second", 0.0))
        self.decisions.append(DecisionV6(track.key, gid, state, reason, float(row["score"]), float(row["score"] - second), float(row["body"]), float(row["face"]), float(row["temporal"]), float(row["spatial"]), int(row["support"]), track.camera, provisional, merged))

    def run(self, tracks: Dict[str, Tracklet], faces: Dict[str, List[Feature]]):
        self.mapping.clear()
        self.identities.clear()
        self.face_trusted.clear()
        self.face_candidate.clear()
        self.decisions.clear()
        self.provisional.clear()
        self.edges.clear()
        self.identity_merges = 0
        self.reassociated = 0
        self.same_camera_reassociated = 0
        self.cross_camera_reidentified = 0
        self.face_assisted = 0
        self.body_assisted = 0
        self.temporal_assisted = 0
        self.face_observations = sum(bool(v) for v in faces.values())
        self.face_high_quality = sum(sum(float(x.quality) >= self.promote for x in v) for v in faces.values())
        self.face_matches = 0
        self.face_conflicts = 0
        self.pending_count = 0

        ordered = sorted(tracks.values(), key=lambda x: (-x.evidence(), x.start, x.camera, x.key))
        pending: List[Tracklet] = []
        assigned: Dict[str, str] = {}

        for track in ordered:
            ranked = self._rank_identity(track, tracks, faces)
            best = ranked[0] if ranked else None
            second = ranked[1]["score"] if len(ranked) > 1 else 0.0
            if best is not None and self._accept_existing(best, second, track):
                gid = best["gid"]
                self.mapping[track.key] = gid
                assigned[track.key] = gid
                self._add_track(gid, track, trusted=best["score"] >= self.body_strong or best["face"] >= self.face_strong or best["temporal"] >= self.same_camera_min)
                self._add_face(gid, faces.get(track.key, []), trusted=True)
                provisional = next((p for p, key in self.provisional.items() if key == gid), gid)
                row = dict(best)
                row["second"] = second
                reason = best["reason"]
                if best["face"] >= self.face_strong:
                    self.face_matches += 1
                    self.face_assisted += 1
                    reason = "face_assisted"
                elif best["temporal"] >= self.same_camera_min:
                    self.temporal_assisted += 1
                    self.reassociated += 1
                    self.same_camera_reassociated += 1
                    reason = "recent_lost_track"
                else:
                    self.body_assisted += 1
                if track.camera != next(iter(self.identities[gid].cameras - {track.camera}), track.camera) and len(self.identities[gid].cameras) > 1:
                    self.cross_camera_reidentified += 1
                self._record(track, gid, "confirmed_existing", reason, row, provisional)
            else:
                provisional = self._new_provisional(track, len(self.provisional) + 1)
                pending.append(track)
                self._record(track, "PENDING", "unknown", "pending_evidence", None, provisional)

        self.pending_count = len(pending)

        # Re-run pending observations against the fully enriched gallery before
        # creating any new permanent identity. This is the key difference from
        # V6's old component-first design.
        progressed = True
        while pending and progressed:
            progressed = False
            next_pending: List[Tracklet] = []
            for track in pending:
                ranked = self._rank_identity(track, tracks, faces)
                best = ranked[0] if ranked else None
                second = ranked[1]["score"] if len(ranked) > 1 else 0.0
                if best is not None and self._accept_existing(best, second, track):
                    gid = best["gid"]
                    self.mapping[track.key] = gid
                    assigned[track.key] = gid
                    self._add_track(gid, track, trusted=best["score"] >= self.body_strong or best["face"] >= self.face_strong)
                    self._add_face(gid, faces.get(track.key, []), trusted=True)
                    row = dict(best)
                    row["second"] = second
                    reason = "face_assisted" if best["face"] >= self.face_strong else ("recent_lost_track" if best["temporal"] >= self.same_camera_min else "body_assisted")
                    if "face" in reason:
                        self.face_matches += 1
                        self.face_assisted += 1
                    if "recent_lost_track" in reason:
                        self.reassociated += 1
                        self.same_camera_reassociated += 1
                        self.temporal_assisted += 1
                    else:
                        self.body_assisted += 1
                    self._record(track, gid, "confirmed_existing", reason, row, next((p for p, key in self.provisional.items() if key == gid), gid))
                    progressed = True
                else:
                    next_pending.append(track)
            pending = next_pending

        # Pairwise pending grouping is intentionally conservative. It prevents a
        # difficult sitting observation from becoming permanent solely because its
        # first comparison was weak, while still allowing multiple observations of
        # the same unknown person to seed one identity.
        if pending:
            keys = [x.key for x in pending]
            dsu = DSU(keys)
            edges: List[EdgeV6] = []
            for i, left in enumerate(pending):
                for right in pending[i + 1:]:
                    body, support, partial = self.appearance(left.features, right.features)
                    face, face_support = self.face_pair(faces.get(left.key, []), faces.get(right.key, []))
                    temporal, spatial, _ = self.continuity(left, right, self.same_camera_gap, self.same_camera_distance)
                    same = left.camera == right.camera
                    score = body
                    reason = "pending_body"
                    accept = False
                    if face >= self.face_strong and face_support >= 1:
                        score = max(score, face)
                        accept = True
                        reason = "pending_face"
                    elif same and temporal >= self.same_camera_min and body >= self.same_camera_body:
                        score = max(score, 0.68 * body + 0.22 * temporal + 0.10 * spatial)
                        accept = True
                        reason = "pending_temporal"
                    elif body >= self.body_strong and support >= 3:
                        accept = True
                    elif partial and body >= self.partial_strong and support >= 2:
                        accept = True
                    if accept:
                        edges.append(EdgeV6(left.key, right.key, float(body), float(face), float(temporal), float(spatial), float(temporal), float(score), int(max(support, face_support)), same, reason))
            edges.sort(key=lambda e: e.score, reverse=True)
            members = {k: [k] for k in keys}
            for edge in edges:
                a, b = dsu.find(edge.left), dsu.find(edge.right)
                if a == b:
                    continue
                if same_camera_overlap_conflict(members[a], members[b], tracks):
                    continue
                root = dsu.union(a, b)
                if root == a:
                    members[root] = members[a] + members[b]
                    members.pop(b, None)
                else:
                    members[root] = members[b] + members[a]
                    members.pop(a, None)
            self.provisional_count = len(members)
            for index, group in enumerate(sorted(members.values(), key=lambda xs: min((tracks[k].start, tracks[k].camera, k) for k in xs)), 1):
                gid = self.gid()
                provisional = f"PNEW{index:04d}"
                self.provisional[provisional] = gid
                self._add_track(gid, tracks[group[0]], trusted=True)
                self._add_face(gid, faces.get(group[0], []), trusted=True)
                self.mapping[group[0]] = gid
                for key in group[1:]:
                    self._add_track(gid, tracks[key], trusted=True)
                    self._add_face(gid, faces.get(key, []), trusted=True)
                    self.mapping[key] = gid
                if len(group) > 1:
                    self.identity_merges += len(group) - 1
                for key in group:
                    if key == group[0]:
                        continue
                    self._record(tracks[key], gid, "promoted", "new_identity_group", None, provisional, True)

        # The final merge pass repairs fragmentation created earlier in the run,
        # but only at a substantially stronger evidence level than ordinary search.
        self._merge_identities(tracks, faces)

        # Canonicalize the decisions to the final merged GID.
        for index, decision in enumerate(self.decisions):
            final_gid = self.mapping.get(decision.key, decision.gid)
            if final_gid != decision.gid:
                self.decisions[index] = DecisionV6(decision.key, final_gid, decision.state, "identity_merge" if decision.state != "unknown" else decision.reason, decision.score, decision.margin, decision.body, decision.face, decision.temporal, decision.spatial, decision.support, decision.camera, decision.provisional, True)

        return dict(self.mapping), list(self.decisions)

    def summary(self, tracks: Dict[str, Tracklet]) -> dict:
        groups: Dict[str, set] = {}
        track_groups: Dict[str, List[str]] = {}
        reasons: Dict[str, int] = {}
        for key, gid in self.mapping.items():
            groups.setdefault(gid, set()).add(tracks[key].camera)
            track_groups.setdefault(gid, []).append(key)
        for decision in self.decisions:
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
        multi = {gid: sorted(cams) for gid, cams in groups.items() if len(cams) > 1}
        fragmented = {gid: sorted(keys) for gid, keys in track_groups.items() if len(keys) > 1}
        return {
            "tracklets": len(tracks),
            "global_ids": len(set(self.mapping.values())),
            "multi_camera": multi,
            "multi_camera_count": len(multi),
            "reasons": reasons,
            "new_identities": sum(x.state in {"promoted", "new_person"} for x in self.decisions),
            "reidentified": sum(x.state == "confirmed_existing" for x in self.decisions),
            "same_camera_reassociations": self.same_camera_reassociated,
            "recent_lost_track_reassociations": self.reassociated,
            "cross_camera_reidentifications": self.cross_camera_reidentified,
            "identity_merges": self.identity_merges,
            "provisional_identities": len(self.provisional),
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
            "face_observations": self.face_observations,
            "face_high_quality": self.face_high_quality,
            "face_matches": self.face_matches,
            "face_conflicts": self.face_conflicts,
            "pending": self.pending_count,
        }


def same_camera_overlap_conflict(left: List[str], right: List[str], tracks: Dict[str, Tracklet]) -> bool:
    for a in left:
        for b in right:
            ta, tb = tracks[a], tracks[b]
            if ta.camera == tb.camera and GlobalIdentityV6.overlap(ta, tb):
                return True
    return False
