from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from rebuild.identity_v3 import Feature, Identity, Tracklet, geometry_similarity


@dataclass(frozen=True)
class V4Decision:
    key: str
    gid: str
    reason: str
    score: float
    margin: float
    body: float
    face: float
    support: int
    face_support: int
    second: float
    camera: str


class GlobalIdentityV4:
    """Persistent body + face global identity gallery."""

    def __init__(self, cfg):
        self.threshold = float(cfg.get("match_threshold", 0.61))
        self.margin = float(cfg.get("match_margin", 0.035))
        self.strong = float(cfg.get("strong_threshold", 0.74))
        self.support = int(cfg.get("support", 2))
        self.gallery = int(cfg.get("gallery", 24))
        self.face_threshold = float(cfg.get("face_threshold", 0.70))
        self.face_strong = float(cfg.get("face_strong_threshold", 0.80))
        self.face_margin = float(cfg.get("face_margin", 0.06))
        self.face_quality_min = float(cfg.get("face_quality", 0.48))
        self.face_strong_quality = float(cfg.get("face_strong_quality", 0.64))
        self.body_conflict = float(cfg.get("body_conflict", 0.52))
        self.new_count = int(cfg.get("new_count", 3))
        self.novelty = float(cfg.get("novelty", 0.985))
        self.promote = float(cfg.get("promote_quality", 0.70))
        self.identities: Dict[str, Identity] = {}
        self.face_trusted: Dict[str, List[Feature]] = {}
        self.face_candidate: Dict[str, List[Feature]] = {}
        self.mapping: Dict[str, str] = {}
        self.decisions: List[V4Decision] = []
        self.next_id = 1

    def gid(self):
        value = f"G{self.next_id:06d}"
        self.next_id += 1
        return value

    @staticmethod
    def overlap(left: Tracklet, right: Tracklet) -> bool:
        return not (left.end < right.start or right.end < left.start)

    def conflict(self, query: Tracklet, identity: Identity, tracks: Dict[str, Tracklet]) -> bool:
        if query.camera not in identity.cameras:
            return False
        for key in identity.tracks:
            item = tracks.get(key)
            if item is not None and item.camera == query.camera and self.overlap(query, item):
                return True
        return False

    @staticmethod
    def body_pair(left: List[Feature], right: List[Feature]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        groups = {}
        kinds = {x.kind for x in left} | {x.kind for x in right}
        for kind in kinds:
            a = [x for x in left if x.kind == kind]
            b = [x for x in right if x.kind == kind]
            if not a or not b:
                continue
            sims = np.stack([x.vector for x in a]) @ np.stack([x.vector for x in b]).T
            flat = np.sort(sims.reshape(-1))
            groups[kind] = (
                0.72 * float(flat[-1]) + 0.28 * float(np.mean(flat[-min(3, len(flat)):])),
                int((flat >= 0.63).sum()),
            )
        if not groups:
            return 0.0, 0
        full = groups.get("full") or groups.get("light")
        upper = groups.get("upper")
        lower = groups.get("lower")
        if full:
            score = full[0]
            support = full[1]
            parts = [x[0] for x in (upper, lower) if x]
            if parts:
                score = 0.82 * score + 0.18 * max(parts)
                support += sum(groups[k][1] for k in ("upper", "lower") if k in groups)
            return float(score), int(support)
        parts = [x for x in (upper, lower) if x]
        if not parts:
            return 0.0, 0
        vals = sorted((x[0] for x in parts), reverse=True)
        return float(0.85 * vals[0] + 0.15 * (vals[1] if len(vals) > 1 else vals[0])), int(sum(x[1] for x in parts))

    @staticmethod
    def face_pair(left: List[Feature], right: List[Feature]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        sims = np.stack([x.vector for x in left]) @ np.stack([x.vector for x in right]).T
        flat = np.sort(sims.reshape(-1))
        best = float(flat[-1])
        top = float(np.mean(flat[-min(3, len(flat)):] ))
        return float(0.72 * best + 0.28 * top), int((flat >= 0.55).sum())

    @staticmethod
    def face_quality(features: List[Feature]) -> float:
        if not features:
            return 0.0
        return float(max(x.quality for x in features))

    def add_face(self, gid: str, values: List[Feature], trusted: bool):
        if not values:
            return
        target = self.face_trusted.setdefault(gid, []) if trusted else self.face_candidate.setdefault(gid, [])
        for item in values:
            if item.quality < self.face_quality_min:
                continue
            if target:
                best = float(np.max(np.stack([x.vector for x in target]) @ item.vector))
                if best >= self.novelty:
                    continue
            target.append(item)
        bank = self.gallery if trusted else max(8, self.gallery // 2)
        target.sort(key=lambda x: x.quality, reverse=True)
        del target[bank:]

    def search_face(self, gid: str, query: List[Feature]) -> Tuple[float, int]:
        values = self.face_trusted.get(gid, []) + self.face_candidate.get(gid, [])
        return self.face_pair(query, values)

    def search(self, track: Tracklet, faces: List[Feature], tracks: Dict[str, Tracklet]):
        ranked = []
        fq = self.face_quality(faces)
        for gid, identity in self.identities.items():
            if self.conflict(track, identity, tracks):
                continue
            body, support = self.body_pair(track.features, identity.values())
            face, face_support = self.search_face(gid, faces)
            shape = 0.5
            if identity.geometry and track.shape > 0:
                shape = geometry_similarity(track.shape, float(np.median(identity.geometry)))
            if fq < self.face_quality_min:
                face = 0.0
                face_support = 0
            if face > 0.0 and body > 0.0:
                if face >= self.face_strong and fq >= self.face_strong_quality:
                    fused = 0.68 * face + 0.24 * body + 0.08 * shape
                else:
                    fused = 0.68 * body + 0.24 * face + 0.08 * shape
            elif face > 0.0:
                fused = 0.90 * face + 0.10 * shape
            else:
                fused = 0.90 * body + 0.10 * shape
            ranked.append((float(fused), support, face, face_support, body, gid, shape))
        ranked.sort(reverse=True)
        return ranked

    def _accept(self, item, second, fq):
        score, support, face, face_support, body, _, _ = item
        margin = score - second
        face_ok = fq >= self.face_quality_min and face >= self.face_threshold and face_support >= 1
        face_strong = fq >= self.face_strong_quality and face >= self.face_strong and face_support >= 1
        body_ok = body >= self.threshold and support >= self.support
        body_strong = body >= self.strong and support >= self.support
        fused_ok = score >= self.threshold and margin >= self.margin
        if face_strong and (body <= 0.0 or body >= self.body_conflict):
            return True, "face_body_strong"
        if face_ok and body <= 0.0 and margin >= self.face_margin:
            return True, "face_only"
        if face_ok and body >= self.body_conflict and fused_ok:
            return True, "face_body"
        if body_strong and margin >= self.margin:
            return True, "body_strong"
        if body_ok and fused_ok:
            return True, "body_gallery"
        return False, "uncertain"

    def assign(self, track: Tracklet, faces: List[Feature], tracks: Dict[str, Tracklet]):
        ranked = self.search(track, faces, tracks)
        fq = self.face_quality(faces)
        if ranked:
            best = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else 0.0
            accepted, reason = self._accept(best, second, fq)
            score, support, face, face_support, body, gid, _ = best
            if accepted:
                self.mapping[track.key] = gid
                identity = self.identities[gid]
                trusted = bool((body >= self.strong and support >= self.support) or
                               (face >= self.face_strong and fq >= self.face_strong_quality))
                identity.add(track, trusted, self.gallery, self.promote)
                if faces:
                    self.add_face(gid, faces, trusted)
                decision = V4Decision(track.key, gid, reason, score, score - second,
                                      body, face, support, face_support, second, track.camera)
                self.decisions.append(decision)
                return decision
            if track.count() < self.new_count and not faces:
                decision = V4Decision(track.key, "PENDING", "pending", score, score - second,
                                      body, face, support, face_support, second, track.camera)
                self.decisions.append(decision)
                return decision

        if track.count() >= self.new_count or (faces and fq >= self.face_strong_quality):
            gid = self.gid()
            self.identities[gid] = Identity(gid)
            self.face_trusted.setdefault(gid, [])
            self.face_candidate.setdefault(gid, [])
            self.identities[gid].add(track, True, self.gallery, self.promote)
            if faces:
                self.add_face(gid, faces, fq >= self.face_strong_quality)
            self.mapping[track.key] = gid
            decision = V4Decision(track.key, gid, "new", 1.0, 1.0,
                                  1.0 if track.features else 0.0, 1.0 if faces else 0.0,
                                  track.count(), len(faces), 0.0, track.camera)
            self.decisions.append(decision)
            return decision

        best_score = ranked[0][0] if ranked else 0.0
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        decision = V4Decision(track.key, "PENDING", "pending", best_score, best_score - second,
                              ranked[0][4] if ranked else 0.0, ranked[0][2] if ranked else 0.0,
                              ranked[0][1] if ranked else 0, ranked[0][3] if ranked else 0,
                              second, track.camera)
        self.decisions.append(decision)
        return decision

    def run(self, tracks: Dict[str, Tracklet], faces_by_track: Dict[str, List[Feature]]):
        usable = {k: v for k, v in tracks.items() if v.count() >= self.new_count or faces_by_track.get(k)}
        order = sorted(usable.values(), key=lambda x: (-x.evidence(), x.camera, x.start, x.key))
        for track in order:
            self.assign(track, faces_by_track.get(track.key, []), usable)
        pending = [x for x in tracks.values() if x.key not in self.mapping and x.count() > 0]
        for track in sorted(pending, key=lambda x: (-x.evidence(), x.camera, x.key)):
            self.assign(track, faces_by_track.get(track.key, []), tracks)
        for track in sorted(tracks.values(), key=lambda x: (x.start, x.camera, x.key)):
            if track.key in self.mapping:
                gid = self.mapping[track.key]
                self.identities[gid].add(track, False, self.gallery, self.promote)
                self.add_face(gid, faces_by_track.get(track.key, []), False)
        return dict(self.mapping), list(self.decisions)

    def summary(self, tracks: Dict[str, Tracklet]):
        groups = {}
        for key, gid in self.mapping.items():
            groups.setdefault(gid, set()).add(tracks[key].camera)
        multi = {gid: sorted(cams) for gid, cams in groups.items() if len(cams) > 1}
        reasons = {}
        for item in self.decisions:
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
        face_count = sum(1 for x in self.decisions if x.face >= self.face_threshold and x.gid != "PENDING")
        body_only = sum(1 for x in self.decisions if x.body >= self.threshold and x.face < self.face_threshold and x.gid != "PENDING")
        conflicts = sum(1 for x in self.decisions if x.reason == "uncertain" and x.body >= self.body_conflict and x.face >= self.face_threshold)
        return {
            "tracklets": len(tracks),
            "global_ids": len(set(self.mapping.values())),
            "multi_camera": multi,
            "multi_camera_count": len(multi),
            "reasons": reasons,
            "trusted_body": sum(len(x.trusted) for x in self.identities.values()),
            "candidate_body": sum(len(x.candidate) for x in self.identities.values()),
            "trusted_face": sum(len(x) for x in self.face_trusted.values()),
            "candidate_face": sum(len(x) for x in self.face_candidate.values()),
            "face_assisted": face_count,
            "body_only": body_only,
            "conflicts": conflicts,
        }
