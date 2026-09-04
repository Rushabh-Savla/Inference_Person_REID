from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from rebuild.identity_v3 import Feature, Identity, Tracklet, unit
from rebuild.identity_v2 import geometry_similarity


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


def _sim(a: Feature, b: Feature) -> float:
    return float(np.dot(unit(a.vector), unit(b.vector)))


def _groups(items: List[Feature]) -> Dict[str, List[Feature]]:
    out: Dict[str, List[Feature]] = {}
    for item in items:
        out.setdefault(item.kind, []).append(item)
    return out


class GlobalIdentityV6:
    """Persistent GID-first association.

    Tracklets are observations, not people. The engine performs sequential search
    against a persistent global gallery, keeps ambiguous observations pending, uses
    same-camera temporal/spatial reassociation as supporting evidence, protects the
    trusted gallery, and only performs a conservative final identity-merge pass.
    """

    def __init__(self, cfg: dict):
        self.match_threshold = float(cfg.get("match_threshold", 0.61))
        self.match_margin = float(cfg.get("match_margin", 0.035))
        self.strong_threshold = float(cfg.get("strong_threshold", 0.74))
        self.support_required = int(cfg.get("support_required", 2))
        self.accumulated_body = float(cfg.get("accumulated_body", 0.56))
        self.accumulated_support = int(cfg.get("accumulated_support", 3))
        self.partial_threshold = float(cfg.get("partial_threshold", 0.62))
        self.partial_support = int(cfg.get("partial_support", 2))
        self.face_threshold = float(cfg.get("face_threshold", 0.78))
        self.face_strong = float(cfg.get("face_strong", 0.90))
        self.face_quality = float(cfg.get("face_quality", 0.60))
        self.face_margin = float(cfg.get("face_margin", 0.04))
        self.same_camera_gap = float(cfg.get("same_camera_gap_sec", 15.0))
        self.same_camera_distance = float(cfg.get("same_camera_distance", 5.0))
        self.same_camera_body = float(cfg.get("same_camera_body", 0.48))
        self.same_camera_min = float(cfg.get("same_camera_min_continuity", 0.30))
        self.gallery = int(cfg.get("gallery", 32))
        self.promote = float(cfg.get("promote_quality", 0.68))
        self.novelty = float(cfg.get("novelty", 0.985))
        self.face_gallery = int(cfg.get("face_gallery", 10))
        self.face_novelty = float(cfg.get("face_novelty", 0.985))
        self.merge_body = float(cfg.get("merge_body", 0.80))
        self.merge_face = float(cfg.get("merge_face", 0.90))
        self.merge_support = int(cfg.get("merge_support", 3))

        self.identities: Dict[str, Identity] = {}
        self.face_trusted: Dict[str, List[Feature]] = {}
        self.face_candidate: Dict[str, List[Feature]] = {}
        self.mapping: Dict[str, str] = {}
        self.provisional: Dict[str, str] = {}
        self.decisions: List[DecisionV6] = []
        self.edges: List[dict] = []
        self.next_id = 1

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
        self.provisional_count = 0

    def gid(self) -> str:
        value = f"G{self.next_id:06d}"
        self.next_id += 1
        return value

    @staticmethod
    def overlap(left: Tracklet, right: Tracklet) -> bool:
        return not (left.end < right.start or right.end < left.start)

    @staticmethod
    def _center(row: dict) -> Tuple[float, float]:
        x1, y1, x2, y2 = map(float, row.get("bbox") or [0, 0, 0, 0])
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @staticmethod
    def _height(row: dict) -> float:
        box = row.get("bbox") or [0, 0, 0, 0]
        return max(1.0, float(box[3]) - float(box[1]))

    @staticmethod
    def _endpoint(track: Tracklet, last: bool) -> dict | None:
        if not track.observations:
            return None
        rows = sorted(track.observations, key=lambda x: float(x.get("timestamp", 0.0)))
        return rows[-1] if last else rows[0]

    def continuity(self, left: Tracklet, right: Tracklet) -> Tuple[float, float, float]:
        if left.camera != right.camera or self.overlap(left, right):
            return 0.0, 0.0, 0.0
        prev, nxt = (left, right) if left.end <= right.start else (right, left)
        gap = max(0.0, nxt.start - prev.end)
        if gap > self.same_camera_gap:
            return 0.0, 0.0, gap
        a = self._endpoint(prev, True)
        b = self._endpoint(nxt, False)
        if a is None or b is None:
            return 0.0, 0.0, gap
        ax, ay = self._center(a)
        bx, by = self._center(b)
        distance = float(np.hypot(bx - ax, by - ay) / self._height(a))
        spatial = max(0.0, 1.0 - distance / self.same_camera_distance)
        temporal = max(0.0, 1.0 - gap / self.same_camera_gap)
        return float(0.55 * temporal + 0.45 * spatial), float(spatial), float(gap)

    @staticmethod
    def _compatible(query: str, gallery: str) -> bool:
        if query == gallery:
            return True
        if {query, gallery} <= {"full", "light"}:
            return True
        if query in {"upper", "lower"} and gallery in {"full", "light"}:
            return True
        if gallery in {"upper", "lower"} and query in {"full", "light"}:
            return True
        return False

    def body_score(self, query: List[Feature], gallery: List[Feature]) -> Tuple[float, int, bool]:
        if not query or not gallery:
            return 0.0, 0, False
        qg, gg = _groups(query), _groups(gallery)
        pairs: List[Tuple[str, str, float, int]] = []
        for qk, qv in qg.items():
            for gk, gv in gg.items():
                if not self._compatible(qk, gk):
                    continue
                sims = np.asarray([[_sim(a, b) for b in gv] for a in qv], dtype=np.float32)
                flat = np.sort(sims.reshape(-1))
                if flat.size == 0:
                    continue
                best = float(flat[-1])
                top = float(np.mean(flat[-min(3, len(flat)):]))
                pairs.append((qk, gk, 0.72 * best + 0.28 * top, int((flat >= 0.60).sum())))
        if not pairs:
            return 0.0, 0, False
        exact = [(s, n, q) for q, g, s, n in pairs if q == g]
        pool = exact if exact else [(s, n, q) for q, _, s, n in pairs]
        pool.sort(reverse=True)
        base, base_support = pool[0][0], pool[0][1]
        partial = bool(any(q in {"upper", "lower"} for q, _, _, _ in pairs)) and not any(q in {"full", "light"} for q, _, _, _ in pairs)
        parts = [s for q, _, s, _ in pairs if q in {"upper", "lower"}]
        final = 0.85 * max(parts) + 0.15 * base if partial and parts else base
        return float(final), max(base_support, sum(n for _, _, _, n in pairs)), partial

    @staticmethod
    def face_score(query: List[Feature], gallery: List[Feature]) -> Tuple[float, int]:
        if not query or not gallery:
            return 0.0, 0
        sims = np.asarray([[_sim(a, b) for b in gallery] for a in query], dtype=np.float32)
        flat = np.sort(sims.reshape(-1))
        if flat.size == 0:
            return 0.0, 0
        top = np.mean(flat[-min(3, len(flat)):])
        return float(0.72 * flat[-1] + 0.28 * top), int((flat >= 0.55).sum())

    def _conflict(self, track: Tracklet, identity: Identity, tracks: Dict[str, Tracklet]) -> bool:
        if track.camera not in identity.cameras:
            return False
        return any(
            (item := tracks.get(key)) is not None
            and item.camera == track.camera
            and self.overlap(track, item)
            for key in identity.tracks
        )

    def _geometry(self, track: Tracklet, identity: Identity) -> float:
        if not identity.geometry or track.shape <= 0:
            return 0.5
        return float(geometry_similarity(track.shape, float(np.median(identity.geometry))))

    @staticmethod
    def _face_quality(items: List[Feature]) -> float:
        return float(max((x.quality for x in items), default=0.0))

    def _latest_same_camera(self, track: Tracklet, identity: Identity, tracks: Dict[str, Tracklet]) -> Tracklet | None:
        candidates = [tracks[k] for k in identity.tracks if k in tracks and tracks[k].camera == track.camera and tracks[k].end <= track.start]
        return max(candidates, key=lambda x: x.end) if candidates else None

    def _rank(self, track: Tracklet, tracks: Dict[str, Tracklet], faces: Dict[str, List[Feature]]) -> List[dict]:
        rows: List[dict] = []
        qface = faces.get(track.key, [])
        qface_quality = self._face_quality(qface)
        for gid, identity in self.identities.items():
            if self._conflict(track, identity, tracks):
                continue
            body, support, partial = self.body_score(track.features, identity.values())
            shape = self._geometry(track, identity)
            prior = self._latest_same_camera(track, identity, tracks)
            temporal = spatial = gap = 0.0
            if prior is not None:
                temporal, spatial, gap = self.continuity(prior, track)
            face, face_support = self.face_score(qface, self.face_trusted.get(gid, []) + self.face_candidate.get(gid, []))
            score = 0.94 * body + 0.06 * shape
            reason = "body_gallery"
            if temporal > 0.0:
                score = 0.82 * score + 0.13 * temporal + 0.05 * spatial
                if body >= self.same_camera_body:
                    reason = "recent_lost_track"
            if face >= self.face_threshold and qface_quality >= self.face_quality:
                score = max(score, 0.88 * face + 0.12 * max(body, 0.5))
                reason = "face_assisted"
            rows.append({
                "gid": gid, "score": float(score), "body": float(body), "face": float(face),
                "temporal": float(temporal), "spatial": float(spatial), "gap": float(gap),
                "support": int(support), "face_support": int(face_support), "partial": bool(partial),
                "shape": float(shape), "reason": reason, "face_quality": float(qface_quality),
            })
        return sorted(rows, key=lambda x: x["score"], reverse=True)

    def _accept(self, best: dict, second: float) -> Tuple[bool, str]:
        margin = best["score"] - second
        if best["face"] >= self.face_strong and best["face_quality"] >= self.face_quality:
            return True, "face_assisted"
        if best["temporal"] >= self.same_camera_min and best["spatial"] >= 0.35 and best["body"] >= self.same_camera_body:
            return True, "recent_lost_track"
        if best["body"] >= self.strong_threshold and (margin >= self.match_margin or best["support"] >= self.support_required):
            return True, "body_strong"
        if best["body"] >= self.match_threshold and best["support"] >= self.support_required and margin >= self.match_margin:
            return True, "body_gallery"
        if best["body"] >= self.accumulated_body and best["support"] >= self.accumulated_support and margin >= 0.02:
            return True, "body_accumulated"
        if best["partial"] and best["body"] >= self.partial_threshold and best["support"] >= self.partial_support and margin >= 0.02:
            return True, "partial_body"
        if best["face"] >= self.face_threshold and best["face_quality"] >= self.face_quality and best["body"] >= 0.42 and margin >= 0.02:
            return True, "face_body_assist"
        return False, "pending"

    def _add_track(self, gid: str, track: Tracklet, trusted: bool) -> None:
        self.identities.setdefault(gid, Identity(gid)).add(track, trusted, self.gallery, self.promote)

    def _add_face(self, gid: str, values: List[Feature], trusted: bool) -> None:
        if not values:
            return
        target = self.face_trusted if trusted else self.face_candidate
        bank = target.setdefault(gid, [])
        for item in values:
            if item.quality < self.face_quality:
                continue
            if bank:
                best = float(np.max(np.stack([x.vector for x in bank]) @ unit(item.vector)))
                if best >= self.face_novelty and item.quality <= max(x.quality for x in bank) + 0.02:
                    continue
            bank.append(item)
        bank.sort(key=lambda x: x.quality, reverse=True)
        limit = self.face_gallery if trusted else max(4, self.face_gallery // 2)
        del bank[limit:]

    def _record(self, track: Tracklet, gid: str, state: str, reason: str, row: dict | None, provisional: str, merged: bool = False) -> None:
        if row is None:
            self.decisions.append(DecisionV6(track.key, gid, state, reason, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, track.camera, provisional, merged))
            return
        second = float(row.get("second", 0.0))
        self.decisions.append(DecisionV6(track.key, gid, state, reason, float(row["score"]), float(row["score"] - second), float(row["body"]), float(row["face"]), float(row["temporal"]), float(row["spatial"]), int(row["support"]), track.camera, provisional, merged))

    def _new_gid(self, track: Tracklet, faces: Dict[str, List[Feature]], provisional: str) -> str:
        gid = self.gid()
        self.identities[gid] = Identity(gid)
        self.face_trusted[gid] = []
        self.face_candidate[gid] = []
        trusted = track.evidence() >= 0.68
        self._add_track(gid, track, trusted=trusted)
        self._add_face(gid, faces.get(track.key, []), trusted=trusted)
        self.mapping[track.key] = gid
        self.provisional[provisional] = gid
        return gid

    def _can_merge(self, left: str, right: str, tracks: Dict[str, Tracklet]) -> bool:
        a, b = self.identities[left], self.identities[right]
        for lk in a.tracks:
            lt = tracks.get(lk)
            if lt is None:
                continue
            for rk in b.tracks:
                rt = tracks.get(rk)
                if rt is not None and lt.camera == rt.camera and self.overlap(lt, rt):
                    return False
        return True

    def _merge_pair(self, winner: str, loser: str) -> None:
        a, b = self.identities[winner], self.identities[loser]
        a.trusted = a.trim(a.trusted + b.trusted, self.gallery)
        a.candidate = a.trim(a.candidate + b.candidate, max(6, self.gallery // 2))
        a.tracks = list(dict.fromkeys(a.tracks + b.tracks))
        a.cameras.update(b.cameras)
        a.geometry = (a.geometry + b.geometry)[-self.gallery:]
        for name, limit in (("face_trusted", self.face_gallery), ("face_candidate", max(4, self.face_gallery // 2))):
            target = getattr(self, name)
            target[winner] = sorted(target.get(winner, []) + target.get(loser, []), key=lambda x: x.quality, reverse=True)[:limit]
            target.pop(loser, None)
        for key, gid in list(self.mapping.items()):
            if gid == loser:
                self.mapping[key] = winner
        self.identities.pop(loser, None)
        self.identity_merges += 1

    def _merge_pass(self, tracks: Dict[str, Tracklet]) -> None:
        changed = True
        while changed:
            changed = False
            ids = list(self.identities)
            for i, left in enumerate(ids):
                if left not in self.identities:
                    continue
                for right in ids[i + 1:]:
                    if right not in self.identities or not self._can_merge(left, right, tracks):
                        continue
                    a, b = self.identities[left], self.identities[right]
                    body, support, _ = self.body_score(a.values(), b.values())
                    face, _ = self.face_score(self.face_trusted.get(left, []) + self.face_candidate.get(left, []), self.face_trusted.get(right, []) + self.face_candidate.get(right, []))
                    if (body >= self.merge_body and support >= self.merge_support) or face >= self.merge_face:
                        winner, loser = (left, right) if int(left[1:]) < int(right[1:]) else (right, left)
                        self._merge_pair(winner, loser)
                        changed = True
                        break
                if changed:
                    break

    def run(self, tracks: Dict[str, Tracklet], faces: Dict[str, List[Feature]]):
        self.mapping.clear()
        self.identities.clear()
        self.face_trusted.clear()
        self.face_candidate.clear()
        self.provisional.clear()
        self.decisions.clear()
        self.identity_merges = self.reassociated = self.same_camera_reassociated = 0
        self.cross_camera_reidentified = self.face_assisted = self.body_assisted = self.temporal_assisted = 0
        self.pending_count = self.provisional_count = 0
        self.face_observations = sum(bool(v) for v in faces.values())
        self.face_high_quality = sum(sum(x.quality >= self.promote for x in v) for v in faces.values())
        self.face_matches = self.face_conflicts = 0

        ordered = sorted(tracks.values(), key=lambda x: (x.start, x.camera, x.key))
        pending: List[Tracklet] = []

        for track in ordered:
            ranked = self._rank(track, tracks, faces)
            best = ranked[0] if ranked else None
            second = ranked[1]["score"] if len(ranked) > 1 else 0.0
            if best is None:
                pending.append(track)
                provisional = f"P{len(pending):04d}"
                self.provisional[provisional] = "PENDING"
                self._record(track, "PENDING", "unknown", "pending_evidence", None, provisional)
                continue
            accepted, reason = self._accept(best, second)
            if not accepted:
                pending.append(track)
                provisional = f"P{len(pending):04d}"
                self.provisional[provisional] = "PENDING"
                self._record(track, "PENDING", "unknown", "pending_evidence", {**best, "second": second}, provisional)
                continue

            gid = best["gid"]
            prior_cameras = set(self.identities[gid].cameras)
            self.mapping[track.key] = gid
            trusted = reason in {"body_strong", "face_assisted", "recent_lost_track"} and track.evidence() >= 0.60
            self._add_track(gid, track, trusted=trusted)
            self._add_face(gid, faces.get(track.key, []), trusted=trusted)
            self._record(track, gid, "confirmed_existing", reason, {**best, "second": second}, next((p for p, g in self.provisional.items() if g == gid), gid))
            if best["body"] >= self.accumulated_body:
                self.body_assisted += 1
            if "face" in reason:
                self.face_assisted += 1
                self.face_matches += 1
            if reason == "recent_lost_track":
                self.reassociated += 1
                self.same_camera_reassociated += 1
                self.temporal_assisted += 1
            if prior_cameras and track.camera not in prior_cameras:
                self.cross_camera_reidentified += 1

        changed = True
        while pending and changed:
            changed = False
            remain: List[Tracklet] = []
            for track in pending:
                ranked = self._rank(track, tracks, faces)
                best = ranked[0] if ranked else None
                second = ranked[1]["score"] if len(ranked) > 1 else 0.0
                if best is None:
                    remain.append(track)
                    continue
                accepted, reason = self._accept(best, second)
                if not accepted:
                    remain.append(track)
                    continue
                gid = best["gid"]
                prior_cameras = set(self.identities[gid].cameras)
                self.mapping[track.key] = gid
                trusted = reason in {"body_strong", "face_assisted", "recent_lost_track"} and track.evidence() >= 0.60
                self._add_track(gid, track, trusted=trusted)
                self._add_face(gid, faces.get(track.key, []), trusted=trusted)
                self._record(track, gid, "confirmed_existing", reason, {**best, "second": second}, next((p for p, g in self.provisional.items() if g == gid), gid))
                if "face" in reason:
                    self.face_assisted += 1
                    self.face_matches += 1
                if reason == "recent_lost_track":
                    self.reassociated += 1
                    self.same_camera_reassociated += 1
                    self.temporal_assisted += 1
                if prior_cameras and track.camera not in prior_cameras:
                    self.cross_camera_reidentified += 1
                changed = True
            pending = remain

        pending = sorted(pending, key=lambda x: (-x.evidence(), x.start, x.camera, x.key))
        while pending:
            remain: List[Tracklet] = []
            for track in pending:
                ranked = self._rank(track, tracks, faces)
                best = ranked[0] if ranked else None
                second = ranked[1]["score"] if len(ranked) > 1 else 0.0
                if best is None:
                    remain.append(track)
                    continue
                accepted, reason = self._accept(best, second)
                if not accepted:
                    remain.append(track)
                    continue
                gid = best["gid"]
                prior_cameras = set(self.identities[gid].cameras)
                self.mapping[track.key] = gid
                trusted = reason in {"body_strong", "face_assisted", "recent_lost_track"} and track.evidence() >= 0.60
                self._add_track(gid, track, trusted=trusted)
                self._add_face(gid, faces.get(track.key, []), trusted=trusted)
                self._record(track, gid, "confirmed_existing", reason, {**best, "second": second}, next((p for p, g in self.provisional.items() if g == gid), gid))
                if "face" in reason:
                    self.face_assisted += 1
                    self.face_matches += 1
                if reason == "recent_lost_track":
                    self.reassociated += 1
                    self.same_camera_reassociated += 1
                    self.temporal_assisted += 1
                if prior_cameras and track.camera not in prior_cameras:
                    self.cross_camera_reidentified += 1
            if len(remain) == len(pending):
                seed = remain.pop(0)
                provisional = f"PNEW{self.next_id:04d}_{seed.key}"
                gid = self._new_gid(seed, faces, provisional)
                self._record(seed, gid, "promoted", "new_identity", None, provisional)
                pending = remain
            else:
                pending = remain

        self.pending_count = 0
        self.provisional_count = len(self.identities)

        self._merge_pass(tracks)
        for i, row in enumerate(self.decisions):
            final_gid = self.mapping.get(row.key, row.gid)
            if final_gid != row.gid:
                self.decisions[i] = DecisionV6(row.key, final_gid, row.state, "identity_merge", row.score, row.margin, row.body, row.face, row.temporal, row.spatial, row.support, row.camera, row.provisional, True)
        return dict(self.mapping), list(self.decisions)

    def summary(self, tracks: Dict[str, Tracklet]) -> dict:
        groups: Dict[str, set] = {}
        members: Dict[str, List[str]] = {}
        reasons: Dict[str, int] = {}
        for key, gid in self.mapping.items():
            groups.setdefault(gid, set()).add(tracks[key].camera)
            members.setdefault(gid, []).append(key)
        for item in self.decisions:
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
        fragmented = {gid: sorted(keys) for gid, keys in members.items() if len(keys) > 1}
        multi = {gid: sorted(cams) for gid, cams in groups.items() if len(cams) > 1}
        return {
            "tracklets": len(tracks),
            "global_ids": len(set(self.mapping.values())),
            "multi_camera": multi,
            "multi_camera_count": len(multi),
            "reasons": reasons,
            "new_identities": sum(x.state == "promoted" for x in self.decisions),
            "reidentified": sum(x.state == "confirmed_existing" for x in self.decisions),
            "same_camera_reassociations": self.same_camera_reassociated,
            "recent_lost_track_reassociations": self.reassociated,
            "cross_camera_reidentifications": self.cross_camera_reidentified,
            "identity_merges": self.identity_merges,
            "provisional_identities": self.provisional_count,
            "pending": self.pending_count,
            "fragmented_identities": fragmented,
            "fragmented_identity_count": len(fragmented),
            "face_assisted": self.face_assisted,
            "face_matches": self.face_matches,
            "face_observations": self.face_observations,
            "face_high_quality": self.face_high_quality,
            "face_conflicts": self.face_conflicts,
            "body_assisted": self.body_assisted,
            "temporal_assisted": self.temporal_assisted,
            "edge_count": len(self.edges),
            "trusted_body": sum(len(x.trusted) for x in self.identities.values()),
            "candidate_body": sum(len(x.candidate) for x in self.identities.values()),
        }
