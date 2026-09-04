from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from rebuild.identity_v3 import Feature, Identity, Tracklet, geometry_similarity


@dataclass(frozen=True)
class DecisionV5:
    key: str
    gid: str
    reason: str
    score: float
    margin: float
    body: float
    face: float
    support: int
    stable_support: int
    face_support: int
    face_margin: float
    second: float
    camera: str


class GlobalIdentityV5:
    """Body-first persistent ReID with multi-observation rescue and face assistance."""

    def __init__(self, cfg: dict):
        self.threshold = float(cfg.get("match_threshold", 0.61))
        self.margin = float(cfg.get("match_margin", 0.035))
        self.strong = float(cfg.get("strong_threshold", 0.74))
        self.support = int(cfg.get("support", 2))
        self.gallery = int(cfg.get("gallery", 24))
        self.novelty = float(cfg.get("novelty", 0.985))
        self.promote = float(cfg.get("promote_quality", 0.70))
        self.new_count = int(cfg.get("new_count", 3))
        self.rescue_body = float(cfg.get("rescue_body", 0.56))
        self.rescue_support = int(cfg.get("rescue_support", 4))
        self.rescue_stable = int(cfg.get("rescue_stable", 2))
        self.rescue_margin = float(cfg.get("rescue_margin", 0.04))
        self.evidence_threshold = float(cfg.get("evidence_threshold", 0.55))
        self.stable_threshold = float(cfg.get("stable_threshold", 0.61))
        self.face_threshold = float(cfg.get("face_threshold", 0.78))
        self.face_rescue = float(cfg.get("face_rescue", 0.84))
        self.face_margin = float(cfg.get("face_margin", 0.04))
        self.face_min_quality = float(cfg.get("face_quality", 0.60))
        self.face_strong_quality = float(cfg.get("face_strong_quality", 0.66))
        self.face_gallery = int(cfg.get("face_gallery", 8))
        self.face_novelty = float(cfg.get("face_novelty", 0.985))
        self.merge_body = float(cfg.get("merge_body", 0.74))
        self.merge_support = int(cfg.get("merge_support", 3))
        self.merge_face = float(cfg.get("merge_face", 0.90))
        self.identities: Dict[str, Identity] = {}
        self.face_trusted: Dict[str, List[Feature]] = {}
        self.face_candidate: Dict[str, List[Feature]] = {}
        self.mapping: Dict[str, str] = {}
        self.decisions: List[DecisionV5] = []
        self.next_id = 1
        self.merge_count = 0
        self.merged_pairs: List[Tuple[str, str]] = []

    def gid(self) -> str:
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
    def pair(left: List[Feature], right: List[Feature]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        groups = {}
        for kind in {x.kind for x in left} | {x.kind for x in right}:
            a = [x for x in left if x.kind == kind]
            b = [x for x in right if x.kind == kind]
            if not a or not b:
                continue
            sims = np.stack([x.vector for x in a]) @ np.stack([x.vector for x in b]).T
            flat = np.sort(sims.reshape(-1))
            groups[kind] = (0.72 * float(flat[-1]) + 0.28 * float(np.mean(flat[-min(3, len(flat)):])), int((flat >= 0.63).sum()))
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
        parts = [x for x in (upper, lower) if x]
        if not parts:
            return 0.0, 0
        parts = sorted(parts, key=lambda x: x[0], reverse=True)
        return float(0.85 * parts[0][0] + 0.15 * parts[min(1, len(parts) - 1)][0]), int(sum(x[1] for x in parts))

    @staticmethod
    def face_pair(left: List[Feature], right: List[Feature]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        sims = np.stack([x.vector for x in left]) @ np.stack([x.vector for x in right]).T
        flat = np.sort(sims.reshape(-1))
        return float(0.72 * flat[-1] + 0.28 * np.mean(flat[-min(3, len(flat)):])), int((flat >= 0.55).sum())

    @staticmethod
    def quality(values: List[Feature]) -> float:
        return float(max((x.quality for x in values), default=0.0))

    def add_face(self, gid: str, values: List[Feature], trusted: bool) -> None:
        if not values:
            return
        target = self.face_trusted.setdefault(gid, []) if trusted else self.face_candidate.setdefault(gid, [])
        for item in values:
            if item.quality < self.face_min_quality:
                continue
            if target:
                best = float(np.max(np.stack([x.vector for x in target]) @ item.vector))
                if best >= self.face_novelty and item.quality <= max(x.quality for x in target) + 0.02:
                    continue
            target.append(item)
        target.sort(key=lambda x: x.quality, reverse=True)
        limit = self.face_gallery if trusted else max(4, self.face_gallery // 2)
        del target[limit:]

    @staticmethod
    def observation_support(track: Tracklet, identity_values: List[Feature], low: float, stable: float) -> Tuple[int, int]:
        if not track.features or not identity_values:
            return 0, 0
        by_kind: Dict[str, List[np.ndarray]] = {}
        for item in identity_values:
            by_kind.setdefault(item.kind, []).append(item.vector)
        total = stable_count = 0
        for item in track.features:
            bank = by_kind.get(item.kind) or [x.vector for x in identity_values]
            score = float(np.max(np.stack(bank) @ item.vector))
            total += int(score >= low)
            stable_count += int(score >= stable)
        return total, stable_count

    def search(self, track: Tracklet, faces: List[Feature], tracks: Dict[str, Tracklet]) -> List[dict]:
        rows = []
        for gid, identity in self.identities.items():
            if self.conflict(track, identity, tracks):
                continue
            values = identity.values()
            raw_body, pair_support = self.pair(track.features, values)
            obs_support, stable_support = self.observation_support(track, values, self.evidence_threshold, self.stable_threshold)
            shape = 0.5
            if identity.geometry and track.shape > 0:
                shape = geometry_similarity(track.shape, float(np.median(identity.geometry)))
            body = float(0.93 * raw_body + 0.07 * shape) if raw_body > 0 else 0.0
            rows.append({"gid": gid, "body": body, "support": max(pair_support, obs_support),
                         "obs_support": obs_support, "stable_support": stable_support,
                         "face": 0.0, "face_support": 0, "face_margin": 0.0})

        use_face = bool(faces) and self.quality(faces) >= self.face_min_quality
        face_rank = []
        if use_face:
            for row in rows:
                bank = self.face_trusted.get(row["gid"], []) + self.face_candidate.get(row["gid"], [])
                score, support = self.face_pair(faces, bank)
                row["face"], row["face_support"] = float(score), int(support)
                face_rank.append((float(score), row["gid"]))
            face_rank.sort(reverse=True)
            for i, (score, gid) in enumerate(face_rank):
                second = face_rank[i + 1][0] if i + 1 < len(face_rank) else 0.0
                for row in rows:
                    if row["gid"] == gid:
                        row["face_margin"] = float(score - second)
                        break

        for row in rows:
            score = row["body"]
            if row["face"] >= self.face_threshold and row["face_support"] >= 1:
                score = min(0.97, score + 0.08 * max(0.0, row["face"] - 0.60))
            if row["body"] <= 0.0 and row["face"] > 0.0:
                score = 0.86 * row["face"]
            row["score"] = float(score)
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows

    def accept(self, best: dict, second: float, faces: List[Feature]) -> Tuple[bool, str]:
        body, face = best["body"], best["face"]
        margin = best["score"] - second
        fq, fm = self.quality(faces), best["face_margin"]
        if body >= self.strong and (margin >= self.margin or body >= 0.78):
            return True, "body_strong"
        if body >= self.threshold and best["support"] >= self.support and margin >= self.margin:
            if face >= self.face_rescue and fq >= self.face_strong_quality and fm >= self.face_margin:
                return True, "body_face_confirmed"
            return True, "body_gallery"
        if body >= self.rescue_body and best["obs_support"] >= self.rescue_support and best["stable_support"] >= self.rescue_stable and margin >= self.rescue_margin:
            return True, "body_accumulated"
        if face >= self.face_rescue and fq >= self.face_strong_quality and best["face_support"] >= 1 and fm >= self.face_margin and (body <= 0.0 or body >= 0.42):
            return True, "face_rescue"
        if face >= self.face_threshold and fq >= self.face_min_quality and fm >= self.face_margin and body >= 0.54 and margin >= 0.02:
            return True, "body_face_assist"
        return False, "uncertain"

    def assign(self, track: Tracklet, faces: List[Feature], tracks: Dict[str, Tracklet]) -> DecisionV5:
        ranked = self.search(track, faces, tracks)
        best = ranked[0] if ranked else None
        second = ranked[1]["score"] if len(ranked) > 1 else 0.0
        if best is not None:
            accepted, reason = self.accept(best, second, faces)
            if accepted:
                gid = best["gid"]
                self.mapping[track.key] = gid
                trusted = bool((best["body"] >= self.strong and best["support"] >= self.support) or
                               (best["face"] >= self.face_rescue and self.quality(faces) >= self.face_strong_quality))
                self.identities[gid].add(track, trusted, self.gallery, self.promote)
                self.add_face(gid, faces, trusted)
                decision = DecisionV5(track.key, gid, reason, best["score"], best["score"] - second,
                                      best["body"], best["face"], best["support"], best["stable_support"],
                                      best["face_support"], best["face_margin"], second, track.camera)
                self.decisions.append(decision)
                return decision

        if track.count() < self.new_count:
            decision = DecisionV5(track.key, "PENDING", "pending", best["score"] if best else 0.0,
                                  (best["score"] - second) if best else 0.0, best["body"] if best else 0.0,
                                  best["face"] if best else 0.0, best["support"] if best else 0,
                                  best["stable_support"] if best else 0, best["face_support"] if best else 0,
                                  best["face_margin"] if best else 0.0, second, track.camera)
            self.decisions.append(decision)
            return decision

        gid = self.gid()
        self.identities[gid] = Identity(gid)
        self.face_trusted[gid], self.face_candidate[gid] = [], []
        self.identities[gid].add(track, True, self.gallery, self.promote)
        self.add_face(gid, faces, True)
        self.mapping[track.key] = gid
        decision = DecisionV5(track.key, gid, "new", 1.0, 1.0, best["body"] if best else 0.0,
                              best["face"] if best else 0.0, track.count(), track.count(), len(faces),
                              best["face_margin"] if best else 0.0, second, track.camera)
        self.decisions.append(decision)
        return decision

    def _identity_overlap(self, left: Identity, right: Identity, tracks: Dict[str, Tracklet]) -> bool:
        for lk in left.tracks:
            lt = tracks.get(lk)
            if lt is None:
                continue
            for rk in right.tracks:
                rt = tracks.get(rk)
                if rt is not None and rt.camera == lt.camera and self.overlap(lt, rt):
                    return True
        return False

    def _merge_face(self, left: List[Feature], right: List[Feature], limit: int) -> List[Feature]:
        values = list(left)
        for item in right:
            if values:
                best = float(np.max(np.stack([x.vector for x in values]) @ item.vector))
                if best >= self.face_novelty and item.quality <= max(x.quality for x in values) + 0.02:
                    continue
            values.append(item)
        values.sort(key=lambda x: x.quality, reverse=True)
        return values[:limit]

    def _merge(self, winner: str, loser: str) -> None:
        a, b = self.identities[winner], self.identities[loser]
        a.trusted = a.trim(a.trusted + b.trusted, self.gallery)
        a.candidate = a.trim(a.candidate + b.candidate, max(6, self.gallery // 2))
        a.tracks = list(dict.fromkeys(a.tracks + b.tracks))
        a.cameras.update(b.cameras)
        a.geometry = (a.geometry + b.geometry)[-self.gallery:]
        self.face_trusted[winner] = self._merge_face(self.face_trusted.get(winner, []), self.face_trusted.get(loser, []), self.face_gallery)
        self.face_candidate[winner] = self._merge_face(self.face_candidate.get(winner, []), self.face_candidate.get(loser, []), max(4, self.face_gallery // 2))
        for key, gid in list(self.mapping.items()):
            if gid == loser:
                self.mapping[key] = winner
        self.identities.pop(loser, None)
        self.face_trusted.pop(loser, None)
        self.face_candidate.pop(loser, None)
        self.merge_count += 1
        self.merged_pairs.append((loser, winner))

    def merge_identities(self, tracks: Dict[str, Tracklet]) -> None:
        changed = True
        while changed:
            changed = False
            gids = sorted(self.identities)
            for i, left_gid in enumerate(gids):
                if left_gid not in self.identities:
                    continue
                for right_gid in gids[i + 1:]:
                    if right_gid not in self.identities:
                        continue
                    left, right = self.identities[left_gid], self.identities[right_gid]
                    if self._identity_overlap(left, right, tracks):
                        continue
                    body, support = self.pair(left.values(), right.values())
                    face_left = self.face_trusted.get(left_gid, []) + self.face_candidate.get(left_gid, [])
                    face_right = self.face_trusted.get(right_gid, []) + self.face_candidate.get(right_gid, [])
                    face, _ = self.face_pair(face_left, face_right)
                    if (body >= self.merge_body and support >= self.merge_support) or face >= self.merge_face:
                        left_size = len(left.trusted) + len(left.candidate)
                        right_size = len(right.trusted) + len(right.candidate)
                        winner, loser = (left_gid, right_gid) if left_size >= right_size else (right_gid, left_gid)
                        self._merge(winner, loser)
                        changed = True
                        break
                if changed:
                    break

    def run(self, tracks: Dict[str, Tracklet], faces_by_track: Dict[str, List[Feature]]):
        usable = {k: v for k, v in tracks.items() if v.count() >= self.new_count or faces_by_track.get(k)}
        order = sorted(usable.values(), key=lambda x: (-x.evidence(), x.camera, x.start, x.key))
        for track in order:
            self.assign(track, faces_by_track.get(track.key, []), usable)
        pending = [x for x in tracks.values() if x.key not in self.mapping and x.count() > 0]
        for track in sorted(pending, key=lambda x: (-x.evidence(), x.camera, x.key)):
            self.assign(track, faces_by_track.get(track.key, []), tracks)

        for track in sorted(tracks.values(), key=lambda x: (x.evidence(), x.start, x.camera, x.key)):
            if track.key in self.mapping:
                continue
            ranked = self.search(track, faces_by_track.get(track.key, []), tracks)
            if not ranked:
                continue
            best = ranked[0]
            second = ranked[1]["score"] if len(ranked) > 1 else 0.0
            if (best["body"] >= self.rescue_body and best["obs_support"] >= self.rescue_support and
                    best["stable_support"] >= self.rescue_stable and best["score"] - second >= self.rescue_margin):
                gid = best["gid"]
                self.mapping[track.key] = gid
                self.identities[gid].add(track, False, self.gallery, self.promote)
                self.add_face(gid, faces_by_track.get(track.key, []), False)
                self.decisions.append(DecisionV5(track.key, gid, "body_second_chance", best["score"], best["score"] - second,
                                                 best["body"], best["face"], best["support"], best["stable_support"],
                                                 best["face_support"], best["face_margin"], second, track.camera))
        self.merge_identities(tracks)
        return dict(self.mapping), list(self.decisions)

    def summary(self, tracks: Dict[str, Tracklet]) -> dict:
        groups: Dict[str, set] = {}
        for key, gid in self.mapping.items():
            groups.setdefault(gid, set()).add(tracks[key].camera)
        multi = {gid: sorted(cams) for gid, cams in groups.items() if len(cams) > 1}
        reasons: Dict[str, int] = {}
        for item in self.decisions:
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
        accepted = [x for x in self.decisions if x.gid != "PENDING"]
        buckets = {"<0.50": 0, "0.50-0.55": 0, "0.55-0.60": 0, "0.60-0.65": 0,
                   "0.65-0.70": 0, "0.70-0.75": 0, ">=0.75": 0}
        for item in self.decisions:
            value = item.body
            if value < 0.50: buckets["<0.50"] += 1
            elif value < 0.55: buckets["0.50-0.55"] += 1
            elif value < 0.60: buckets["0.55-0.60"] += 1
            elif value < 0.65: buckets["0.60-0.65"] += 1
            elif value < 0.70: buckets["0.65-0.70"] += 1
            elif value < 0.75: buckets["0.70-0.75"] += 1
            else: buckets[">=0.75"] += 1
        return {
            "tracklets": len(tracks),
            "global_ids": len(set(self.mapping.values())),
            "multi_camera": multi,
            "multi_camera_count": len(multi),
            "reasons": reasons,
            "reidentified": sum(x.reason != "new" and x.gid != "PENDING" for x in accepted),
            "face_assisted": sum("face" in x.reason for x in accepted),
            "near_threshold": sum(self.rescue_body - 0.03 <= x.body < self.threshold for x in self.decisions if x.gid == "PENDING"),
            "score_buckets": buckets,
            "identity_merges": self.merge_count,
            "trusted_body": sum(len(x.trusted) for x in self.identities.values()),
            "candidate_body": sum(len(x.candidate) for x in self.identities.values()),
            "trusted_face": sum(len(x) for x in self.face_trusted.values()),
            "candidate_face": sum(len(x) for x in self.face_candidate.values()),
        }
