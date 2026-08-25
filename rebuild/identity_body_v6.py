from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from rebuild.identity_v2 import geometry_similarity, unit
from rebuild.identity_v3 import Feature, Identity, Tracklet


@dataclass(frozen=True)
class DecisionBodyV6:
    key: str
    gid: str
    state: str
    reason: str
    score: float
    margin: float
    body: float
    temporal: float
    spatial: float
    support: int
    camera: str
    provisional: str
    merged: bool = False


def sim(a: Feature, b: Feature) -> float:
    return float(np.dot(unit(a.vector), unit(b.vector)))


def groups(items: List[Feature]) -> Dict[str, List[Feature]]:
    out: Dict[str, List[Feature]] = {}
    for item in items:
        out.setdefault(item.kind, []).append(item)
    return out


class GlobalIdentityBodyV6:
    """Persistent body-only GID association.

    A tracklet is an observation, never an identity. The engine keeps one shared
    gallery of diverse body observations, delays new-ID creation while evidence
    is weak, uses same-camera lost-track continuity, and performs only a
    conservative identity merge after gallery enrichment.
    """

    def __init__(self, cfg: dict):
        self.threshold = float(cfg.get("match_threshold", 0.60))
        self.margin = float(cfg.get("match_margin", 0.035))
        self.strong = float(cfg.get("strong_threshold", 0.72))
        self.support = int(cfg.get("support_required", 2))
        self.accumulated = float(cfg.get("accumulated_body", 0.56))
        self.accumulated_support = int(cfg.get("accumulated_support", 3))
        self.partial = float(cfg.get("partial_threshold", 0.58))
        self.partial_support = int(cfg.get("partial_support", 2))
        self.gallery = int(cfg.get("gallery", 32))
        self.candidate_gallery = int(cfg.get("candidate_gallery", 12))
        self.promote = float(cfg.get("promote_quality", 0.68))
        self.novelty = float(cfg.get("novelty", 0.985))
        self.gap = float(cfg.get("same_camera_gap_sec", 15.0))
        self.distance = float(cfg.get("same_camera_distance", 5.0))
        self.continuity_min = float(cfg.get("same_camera_min_continuity", 0.35))
        self.merge_body = float(cfg.get("merge_body", 0.82))
        self.merge_support = int(cfg.get("merge_support", 3))
        self.seed_count = int(cfg.get("seed_count", 3))

        self.identities: Dict[str, Identity] = {}
        self.mapping: Dict[str, str] = {}
        self.provisional: Dict[str, str] = {}
        self.decisions: List[DecisionBodyV6] = []
        self.merge_count = 0
        self.same_camera_reassociated = 0
        self.cross_camera_reidentified = 0
        self.temporal_assisted = 0
        self.body_assisted = 0
        self.next_id = 1

    def gid(self) -> str:
        value = f"G{self.next_id:06d}"
        self.next_id += 1
        return value

    def order_key(self, track: Tracklet) -> Tuple[float, str, str]:
        """Ordering hook; subclasses may use a calibrated global clock."""
        return float(track.start), track.camera, track.key

    @staticmethod
    def overlap(a: Tracklet, b: Tracklet) -> bool:
        return not (a.end < b.start or b.end < a.start)

    @staticmethod
    def center(row: dict) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(v) for v in row.get("bbox", [0, 0, 0, 0])]
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @staticmethod
    def height(row: dict) -> float:
        b = row.get("bbox", [0, 0, 0, 0])
        return max(1.0, float(b[3]) - float(b[1]))

    @staticmethod
    def endpoint(track: Tracklet, last: bool) -> dict | None:
        if not track.observations:
            return None
        rows = sorted(track.observations, key=lambda x: float(x.get("timestamp", 0.0)))
        return rows[-1] if last else rows[0]

    def continuity(self, prior: Tracklet, query: Tracklet) -> Tuple[float, float, float]:
        if prior.camera != query.camera or self.overlap(prior, query):
            return 0.0, 0.0, 0.0
        gap = max(0.0, query.start - prior.end)
        if gap > self.gap:
            return 0.0, 0.0, gap
        a = self.endpoint(prior, True)
        b = self.endpoint(query, False)
        if a is None or b is None:
            return 0.0, 0.0, gap
        ax, ay = self.center(a)
        bx, by = self.center(b)
        distance = float(np.hypot(bx - ax, by - ay) / self.height(a))
        spatial = max(0.0, 1.0 - distance / self.distance)
        temporal = max(0.0, 1.0 - gap / self.gap)
        return float(0.55 * temporal + 0.45 * spatial), float(spatial), float(gap)

    @staticmethod
    def compatible(query: str, gallery: str) -> bool:
        if query == gallery:
            return True
        if query in {"upper", "lower"} and gallery in {"full", "light"}:
            return True
        if gallery in {"upper", "lower"} and query in {"full", "light"}:
            return True
        if {query, gallery} <= {"full", "light"}:
            return True
        return False

    def body_score(self, query: List[Feature], gallery: List[Feature]) -> Tuple[float, int, bool]:
        if not query or not gallery:
            return 0.0, 0, False
        qg, gg = groups(query), groups(gallery)
        pairs = []
        for qk, qv in qg.items():
            for gk, gv in gg.items():
                if not self.compatible(qk, gk):
                    continue
                matrix = np.asarray([[sim(a, b) for b in gv] for a in qv], dtype=np.float32)
                flat = np.sort(matrix.reshape(-1))
                best = float(flat[-1])
                top = float(np.mean(flat[-min(3, len(flat)):]))
                support = int((flat >= 0.60).sum())
                pairs.append((qk, gk, 0.72 * best + 0.28 * top, support))
        if not pairs:
            return 0.0, 0, False

        exact = [x for x in pairs if x[0] == x[1]]
        pool = exact if exact else pairs
        pool = sorted(pool, key=lambda x: x[2], reverse=True)
        base = pool[0][2]
        support = max(x[3] for x in pool)
        partial_query = any(x[0] in {"upper", "lower"} for x in pairs) and not any(x[0] in {"full", "light"} for x in pairs)
        part_scores = [x[2] for x in pairs if x[0] in {"upper", "lower"}]
        if partial_query and part_scores:
            base = 0.85 * max(part_scores) + 0.15 * base
        support = max(support, sum(x[3] for x in pairs))
        return float(base), int(support), bool(partial_query)

    def conflict(self, query: Tracklet, identity: Identity, tracks: Dict[str, Tracklet]) -> bool:
        if query.camera not in identity.cameras:
            return False
        for key in identity.tracks:
            item = tracks.get(key)
            if item is not None and item.camera == query.camera and self.overlap(query, item):
                return True
        return False

    def geometry(self, query: Tracklet, identity: Identity) -> float:
        if query.shape <= 0 or not identity.geometry:
            return 0.5
        return float(geometry_similarity(query.shape, float(np.median(identity.geometry))))

    def latest_same_camera(self, query: Tracklet, identity: Identity, tracks: Dict[str, Tracklet]) -> Tracklet | None:
        values = [tracks[k] for k in identity.tracks if k in tracks and tracks[k].camera == query.camera and tracks[k].end <= query.start]
        return max(values, key=lambda x: x.end) if values else None

    def rank(self, track: Tracklet, tracks: Dict[str, Tracklet]) -> List[dict]:
        rows = []
        for gid, identity in self.identities.items():
            if self.conflict(track, identity, tracks):
                continue
            body, support, partial = self.body_score(track.features, identity.values())
            shape = self.geometry(track, identity)
            prior = self.latest_same_camera(track, identity, tracks)
            temporal = spatial = gap = 0.0
            if prior is not None:
                temporal, spatial, gap = self.continuity(prior, track)
            score = 0.94 * body + 0.06 * shape
            reason = "body_gallery"
            if temporal > 0.0 and body >= 0.48:
                score = 0.82 * score + 0.13 * temporal + 0.05 * spatial
                reason = "recent_lost_track"
            rows.append({
                "gid": gid, "score": float(score), "body": float(body),
                "temporal": float(temporal), "spatial": float(spatial),
                "gap": float(gap), "support": int(support),
                "partial": bool(partial), "reason": reason,
            })
        return sorted(rows, key=lambda x: x["score"], reverse=True)

    def accept(self, best: dict, second: float) -> Tuple[bool, str]:
        margin = best["score"] - second
        if best["temporal"] >= self.continuity_min and best["spatial"] >= 0.35 and best["body"] >= 0.48:
            return True, "recent_lost_track"
        if best["body"] >= self.strong and (margin >= self.margin or best["support"] >= self.support):
            return True, "body_strong"
        if best["body"] >= self.threshold and best["support"] >= self.support and margin >= self.margin:
            return True, "body_gallery"
        if best["body"] >= self.accumulated and best["support"] >= self.accumulated_support and margin >= 0.02:
            return True, "body_accumulated"
        if best["partial"] and best["body"] >= self.partial and best["support"] >= self.partial_support and margin >= 0.02:
            return True, "partial_body"
        return False, "pending"

    def add_track(self, gid: str, track: Tracklet, trusted: bool) -> None:
        self.identities[gid].add(track, trusted, self.gallery, self.promote)

    def new_identity(self, track: Tracklet, provisional: str) -> str:
        gid = self.gid()
        self.identities[gid] = Identity(gid)
        self.add_track(gid, track, trusted=track.evidence() >= self.promote)
        self.mapping[track.key] = gid
        self.provisional[provisional] = gid
        return gid

    def merge_allowed(self, a: Identity, b: Identity, tracks: Dict[str, Tracklet]) -> bool:
        for ak in a.tracks:
            left = tracks.get(ak)
            if left is None:
                continue
            for bk in b.tracks:
                right = tracks.get(bk)
                if right is not None and left.camera == right.camera and self.overlap(left, right):
                    return False
        return True

    def merge_pair(self, winner: str, loser: str) -> None:
        a, b = self.identities[winner], self.identities[loser]
        a.trusted = a.trim(a.trusted + b.trusted, self.gallery)
        a.candidate = a.trim(a.candidate + b.candidate, self.candidate_gallery)
        a.tracks = list(dict.fromkeys(a.tracks + b.tracks))
        a.cameras.update(b.cameras)
        a.geometry = (a.geometry + b.geometry)[-self.gallery:]
        for key, gid in list(self.mapping.items()):
            if gid == loser:
                self.mapping[key] = winner
        self.identities.pop(loser, None)
        self.merge_count += 1

    def merge_pass(self, tracks: Dict[str, Tracklet]) -> None:
        changed = True
        while changed:
            changed = False
            ids = list(self.identities)
            for i, left_id in enumerate(ids):
                if left_id not in self.identities:
                    continue
                for right_id in ids[i + 1:]:
                    if right_id not in self.identities:
                        continue
                    left, right = self.identities[left_id], self.identities[right_id]
                    if not self.merge_allowed(left, right, tracks):
                        continue
                    score, support, _ = self.body_score(left.values(), right.values())
                    if score < self.merge_body or support < self.merge_support:
                        continue
                    winner, loser = (left_id, right_id) if int(left_id[1:]) < int(right_id[1:]) else (right_id, left_id)
                    self.merge_pair(winner, loser)
                    changed = True
                    break
                if changed:
                    break

    def record(self, track: Tracklet, gid: str, state: str, reason: str, row: dict | None, provisional: str, merged: bool = False) -> None:
        if row is None:
            self.decisions.append(DecisionBodyV6(track.key, gid, state, reason, 0.0, 0.0, 0.0, 0.0, 0.0, 0, track.camera, provisional, merged))
            return
        second = float(row.get("second", 0.0))
        self.decisions.append(DecisionBodyV6(track.key, gid, state, reason, float(row["score"]), float(row["score"] - second), float(row["body"]), float(row["temporal"]), float(row["spatial"]), int(row["support"]), track.camera, provisional, merged))

    def run(self, tracks: Dict[str, Tracklet]):
        self.identities.clear(); self.mapping.clear(); self.provisional.clear(); self.decisions.clear()
        self.next_id = 1; self.merge_count = 0; self.same_camera_reassociated = 0; self.cross_camera_reidentified = 0
        self.temporal_assisted = 0; self.body_assisted = 0

        ordered = sorted([x for x in tracks.values() if x.count() > 0], key=self.order_key)
        pending: List[Tracklet] = []
        for track in ordered:
            ranked = self.rank(track, tracks)
            best = ranked[0] if ranked else None
            second = ranked[1]["score"] if len(ranked) > 1 else 0.0
            if best is None:
                pending.append(track)
                self.record(track, "PENDING", "unknown", "pending_evidence", None, f"P{len(pending):04d}")
                continue
            accepted, reason = self.accept(best, second)
            if not accepted:
                pending.append(track)
                self.record(track, "PENDING", "unknown", "pending_evidence", {**best, "second": second}, f"P{len(pending):04d}")
                continue
            gid = best["gid"]
            prior_cameras = set(self.identities[gid].cameras)
            self.mapping[track.key] = gid
            trusted = reason in {"body_strong", "recent_lost_track"} and track.evidence() >= 0.60
            self.add_track(gid, track, trusted)
            self.record(track, gid, "confirmed_existing", reason, {**best, "second": second}, gid)
            self.body_assisted += 1
            if reason == "recent_lost_track":
                self.same_camera_reassociated += 1; self.temporal_assisted += 1
            if prior_cameras and track.camera not in prior_cameras:
                self.cross_camera_reidentified += 1

        changed = True
        while changed and pending:
            changed = False
            remain: List[Tracklet] = []
            for track in pending:
                ranked = self.rank(track, tracks)
                best = ranked[0] if ranked else None
                second = ranked[1]["score"] if len(ranked) > 1 else 0.0
                if best is not None:
                    accepted, reason = self.accept(best, second)
                    if accepted:
                        gid = best["gid"]; prior_cameras = set(self.identities[gid].cameras)
                        self.mapping[track.key] = gid
                        trusted = reason in {"body_strong", "recent_lost_track"} and track.evidence() >= 0.60
                        self.add_track(gid, track, trusted)
                        self.record(track, gid, "confirmed_existing", reason, {**best, "second": second}, gid)
                        self.body_assisted += 1; changed = True
                        if reason == "recent_lost_track": self.same_camera_reassociated += 1; self.temporal_assisted += 1
                        if prior_cameras and track.camera not in prior_cameras: self.cross_camera_reidentified += 1
                        continue
                remain.append(track)
            pending = remain

        pending = sorted(pending, key=lambda x: (-x.evidence(), x.start, x.camera, x.key))
        while pending:
            ranked_any = False
            remain = []
            for track in pending:
                ranked = self.rank(track, tracks)
                best = ranked[0] if ranked else None
                second = ranked[1]["score"] if len(ranked) > 1 else 0.0
                if best is not None:
                    accepted, reason = self.accept(best, second)
                    if accepted:
                        gid = best["gid"]; prior_cameras = set(self.identities[gid].cameras)
                        self.mapping[track.key] = gid; self.add_track(gid, track, reason == "body_strong")
                        self.record(track, gid, "confirmed_existing", reason, {**best, "second": second}, gid)
                        self.body_assisted += 1; ranked_any = True
                        if reason == "recent_lost_track": self.same_camera_reassociated += 1; self.temporal_assisted += 1
                        if prior_cameras and track.camera not in prior_cameras: self.cross_camera_reidentified += 1
                        continue
                remain.append(track)
            if ranked_any:
                pending = remain; continue
            seed = remain.pop(0)
            provisional = f"PNEW{self.next_id:04d}"
            gid = self.new_identity(seed, provisional)
            self.record(seed, gid, "promoted", "new_identity", None, provisional)
            pending = remain

        self.merge_pass(tracks)
        final = {}
        for key, gid in self.mapping.items():
            final[key] = gid
        for row in self.decisions:
            row_gid = final.get(row.key, row.gid)
            if row_gid != row.gid:
                idx = self.decisions.index(row)
                self.decisions[idx] = DecisionBodyV6(row.key, row_gid, row.state, "identity_merge", row.score, row.margin, row.body, row.temporal, row.spatial, row.support, row.camera, row.provisional, True)
        return final, list(self.decisions)

    def summary(self, tracks: Dict[str, Tracklet]) -> dict:
        groups: Dict[str, List[str]] = {}
        cameras: Dict[str, set] = {}
        reasons: Dict[str, int] = {}
        for key, gid in self.mapping.items():
            groups.setdefault(gid, []).append(key); cameras.setdefault(gid, set()).add(tracks[key].camera)
        for item in self.decisions: reasons[item.reason] = reasons.get(item.reason, 0) + 1
        fragmented = {gid: sorted(keys) for gid, keys in groups.items() if len(keys) > 1}
        multi = {gid: sorted(cams) for gid, cams in cameras.items() if len(cams) > 1}
        return {
            "tracklets": len(tracks),
            "global_ids": len(set(self.mapping.values())),
            "new_identities": sum(x.state == "promoted" for x in self.decisions),
            "reidentified": sum(x.state == "confirmed_existing" for x in self.decisions),
            "same_camera_reassociations": self.same_camera_reassociated,
            "recent_lost_track_reassociations": self.same_camera_reassociated,
            "cross_camera_reidentifications": self.cross_camera_reidentified,
            "identity_merges": self.merge_count,
            "provisional_identities": sum(x.gid.startswith("G") and x.state == "unknown" for x in self.decisions),
            "fragmented_identity_count": len(fragmented),
            "fragmented_identities": fragmented,
            "multi_camera": multi,
            "body_assisted": self.body_assisted,
            "temporal_assisted": self.temporal_assisted,
            "face_assisted": 0,
            "reasons": reasons,
            "trusted_body": sum(len(x.trusted) for x in self.identities.values()),
            "candidate_body": sum(len(x.candidate) for x in self.identities.values()),
        }
