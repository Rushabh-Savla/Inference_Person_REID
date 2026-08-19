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
    """V3 body matching retained as the primary signal with adaptive evidence and face rescue."""

    def __init__(self, cfg: dict):
        self.threshold = float(cfg.get("match_threshold", 0.61))
        self.margin = float(cfg.get("match_margin", 0.035))
        self.strong = float(cfg.get("strong_threshold", 0.74))
        self.support = int(cfg.get("support", 2))
        self.gallery = int(cfg.get("gallery", 24))
        self.novelty = float(cfg.get("novelty", 0.985))
        self.promote = float(cfg.get("promote_quality", 0.70))
        self.new_count = int(cfg.get("new_count", 3))

        # Adaptive body evidence. These do not replace the V3 threshold;
        # they provide a second path when several observations agree.
        self.rescue_body = float(cfg.get("rescue_body", 0.56))
        self.rescue_support = int(cfg.get("rescue_support", 4))
        self.rescue_stable = int(cfg.get("rescue_stable", 2))
        self.rescue_margin = float(cfg.get("rescue_margin", 0.04))
        self.evidence_threshold = float(cfg.get("evidence_threshold", 0.55))
        self.stable_threshold = float(cfg.get("stable_threshold", 0.61))

        # Face is a rescue/boost signal. It never penalizes a qualified body match.
        self.face_threshold = float(cfg.get("face_threshold", 0.78))
        self.face_rescue = float(cfg.get("face_rescue", 0.84))
        self.face_margin = float(cfg.get("face_margin", 0.04))
        self.face_quality = float(cfg.get("face_quality", 0.60))
        self.face_strong_quality = float(cfg.get("face_strong_quality", 0.66))
        self.face_gallery = int(cfg.get("face_gallery", 8))
        self.face_novelty = float(cfg.get("face_novelty", 0.985))

        # Conservative post-pass merge for fragmentation that survived the
        # normal track-to-global search.
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
        groups: Dict[str, Tuple[float, int]] = {}
        kinds = {x.kind for x in left} | {x.kind for x in right}
        for kind in kinds:
            a = [x for x in left if x.kind == kind]
            b = [x for x in right if x.kind == kind]
            if not a or not b:
                continue
            sims = np.stack([x.vector for x in a]) @ np.stack([x.vector for x in b]).T
            flat = np.sort(sims.reshape(-1))
            best = float(flat[-1])
            top = float(np.mean(flat[-min(3, len(flat)):]))
            groups[kind] = (0.72 * best + 0.28 * top, int((flat >= 0.63).sum()))
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
        values = sorted((x[0] for x in parts), reverse=True)
        return float(0.85 * values[0] + 0.15 * (values[1] if len(values) > 1 else values[0])), int(sum(x[1] for x in parts))

    @staticmethod
    def face_pair(left: List[Feature], right: List[Feature]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        sims = np.stack([x.vector for x in left]) @ np.stack([x.vector for x in right]).T
        flat = np.sort(sims.reshape(-1))
        best = float(flat[-1])
        top = float(np.mean(flat[-min(3, len(flat)):]))
        return float(0.72 * best + 0.28 * top), int((flat >= 0.55).sum())

    @staticmethod
    def face_best(values: List[Feature]) -> float:
        return float(max((x.quality for x in values), default=0.0))

    def add_face(self, gid: str, values: List[Feature], trusted: bool) -> None:
        if not values:
            return
        target = self.face_trusted.setdefault(gid, []) if trusted else self.face_candidate.setdefault(gid, [])
        for item in values:
            if item.quality < self.face_quality:
                continue
            if target:
                best = float(np.max(np.stack([x.vector for x in target]) @ item.vector))
                if best >= self.face_novelty and item.quality <= max(x.quality for x in target) + 0.02:
                    continue
            target.append(item)
        target.sort(key=lambda x: x.quality, reverse=True)
        del target[self.face_gallery if trusted else max(4, self.face_gallery // 2):]

    @staticmethod
    def observation_support(track: Tracklet, identity_values: List[Feature], low: float, stable: float) -> Tuple[int, int]:
        if not track.features or not identity_values:
            return 0, 0
        total = 0
        stable_count = 0
        by_kind: Dict[str, List[np.ndarray]] = {}
        for item in identity_values:
            by_kind.setdefault(item.kind, []).append(item.vector)
        for item in track.features:
            gallery = by_kind.get(item.kind)
            if not gallery:
                gallery = [x.vector for x in identity_values]
            if not gallery:
                continue
            score = float(np.max(np.stack(gallery) @ item.vector))
            total += int(score >= low)
            stable_count += int(score >= stable)
        return total, stable_count

    def search(self, track: Tracklet, faces: List[Feature], tracks: Dict[str, Tracklet]):
        body_rows = []
        for gid, identity in self.identities.items():
            if self.conflict(track, identity, tracks):
                continue
            values = identity.values()
            body, pair_support = self.pair(track.features, values)
            obs_support, stable_support = self.observation_support(
                track, values, self.evidence_threshold, self.stable_threshold
            )
            shape = 0.5
            if identity.geometry and track.shape > 0:
                shape = geometry_similarity(track.shape, float(np.median(identity.geometry)))
            body_score = float(0.93 * body + 0.07 * shape) if body > 0 else 0.0
            body_rows.append({
                "gid": gid,
                "body": body_score,
                "raw_body": body,
                "support": max(pair_support, obs_support),
                "obs_support": obs_support,
                "stable_support": stable_support,
                "face": 0.0,
                "face_support": 0,
                "shape": shape,
            })

        face_available = bool(faces) and self.face_best(faces) >= self.face_quality
        face_rows = []
        if face_available:
            for row in body_rows:
                gid = row["gid"]
                values = self.face_trusted.get(gid, []) + self.face_candidate.get(gid, [])
                score, support = self.face_pair(faces, values)
                row["face"] = float(score)
                row["face_support"] = int(support)
                face_rows.append((float(score), gid))
            face_rows.sort(reverse=True)

        face_rank = {gid: idx for idx, (_, gid) in enumerate(face_rows)}
        for row in body_rows:
            if row["gid"] in face_rank and len(face_rows) > 1:
                idx = face_rank[row["gid"]]
                second = face_rows[idx + 1][0] if idx + 1 < len(face_rows) else 0.0
                row["face_margin"] = float(row["face"] - second)
            else:
                row["face_margin"] = float(row["face"])

            # Body remains primary. A reliable matching face only gives a bounded boost.
            score = row["body"]
            if row["face"] >= self.face_threshold and row["face_support"] >= 1:
                score = min(0.97, score + 0.08 * max(0.0, row["face"] - 0.60))
            if row["body"] <= 0.0 and row["face"] > 0.0:
                score = 0.86 * row["face"]
            row["score"] = float(score)

        body_rows.sort(key=lambda x: x["score"], reverse=True)
        return body_rows

    def _accept(self, best: dict, second: float, track: Tracklet, face_quality: float) -> Tuple[bool, str]:
        score = float(best["score"])
        body = float(best["body"])
        face = float(best["face"])
        margin = score - second
        face_margin = float(best.get("face_margin", 0.0))

        # A very strong body match is independent of weak/missing face evidence.
        if body >= self.strong and (margin >= self.margin or body >= self.78):
            return True, "body_strong"

        # Preserve the V3 operating point when the body gallery agrees.
        if body >= self.threshold and best["support"] >= self.support and margin >= self.margin:
            if face >= self.face_rescue and face_quality >= self.face_strong_quality and face_margin >= self.face_margin:
                return True, "body_face_confirmed"
            return True, "body_gallery"

        # Several consistent moderate observations can rescue a borderline track.
        if (
            body >= self.rescue_body
            and best["obs_support"] >= self.rescue_support
            and best["stable_support"] >= self.rescue_stable
            and margin >= self.rescue_margin
        ):
            return True, "body_accumulated"

        # Strong, high-quality face can rescue a weak body when it is itself well separated.
        if (
            face >= self.face_rescue
            and face_quality >= self.face_strong_quality
            and best["face_support"] >= 1
            and face_margin >= self.face_margin
            and (body <= 0.0 or body >= 0.42)
        ):
            return True, "face_rescue"

        # Face assists an otherwise borderline body candidate, but never suppresses it.
        if (
            face >= self.face_threshold
            and face_quality >= self.face_quality
            and face_margin >= self.face_margin
            and body >= 0.54
            and margin >= 0.02
        ):
            return True, "body_face_assist"

        return False, "uncertain"

    def assign(self, track: Tracklet, faces: List[Feature], tracks: Dict[str, Tracklet]) -> DecisionV5:
        ranked = self.search(track, faces, tracks)
        fq = self.face_best(faces)
        best = ranked[0] if ranked else None
        second = ranked[1]["score"] if len(ranked) > 1 else 0.0

        if best is not None:
            accepted, reason = self._accept(best, second, track, fq)
            if accepted:
                gid = best["gid"]
                self.mapping[track.key] = gid
                trusted = bool(
                    (best["body"] >= self.strong and best["support"] >= self.support)
                    or (best["face"] >= self.face_rescue and fq >= self.face_strong_quality)
                )
                self.identities[gid].add(track, trusted, self.gallery, self.promote)
                if faces:
                    self.add_face(gid, faces, trusted)
                decision = DecisionV5(
                    track.key, gid, reason, float(best["score"]), float(best["score"] - second),
                    float(best["body"]), float(best["face"]), int(best["support"]),
                    int(best["stable_support"]), int(best["face_support"]),
                    float(best.get("face_margin", 0.0)), float(second), track.camera,
                )
                self.decisions.append(decision)
                return decision

        # New-ID creation remains conservative. A borderline candidate is logged,
        # but repeated evidence is allowed to win before a new ID is created.
        if track.count() < self.new_count:
            decision = DecisionV5(
                track.key, "PENDING", "pending", float(best["score"] if best else 0.0),
                float((best["score"] - second) if best else 0.0),
                float(best["body"] if best else 0.0), float(best["face"] if best else 0.0),
                int(best["support"] if best else 0), int(best["stable_support"] if best else 0),
                int(best["face_support"] if best else 0),
                float(best.get("face_margin", 0.0) if best else 0.0), float(second), track.camera,
            )
            self.decisions.append(decision)
            return decision

        gid = self.gid()
        self.identities[gid] = Identity(gid)
        self.face_trusted.setdefault(gid, [])
        self.face_candidate.setdefault(gid, [])
        self.identities[gid].add(track, True, self.gallery, self.promote)
        if faces:
            self.add_face(gid, faces, True)
        decision = DecisionV5(
            track.key, gid, "new", 1.0, 1.0,
            float(best["body"] if best else 0.0), float(best["face"] if best else 0.0),
            track.count(), track.count(), len(faces), float(best.get("face_margin", 0.0) if best else 0.0),
            float(second), track.camera,
        )
        self.mapping[track.key] = gid
        self.decisions.append(decision)
        return decision

    def _identity_overlap(self, left: Identity, right: Identity, tracks: Dict[str, Tracklet]) -> bool:
        common = set(left.cameras) & set(right.cameras)
        if not common:
            return False
        for lk in left.tracks:
            lt = tracks.get(lk)
            if lt is None or lt.camera not in common:
                continue
            for rk in right.tracks:
                rt = tracks.get(rk)
                if rt is None or rt.camera != lt.camera:
                    continue
                if self.overlap(lt, rt):
                    return True
        return False

    def _merge_pair(self, winner: str, loser: str) -> None:
        a = self.identities[winner]
        b = self.identities[loser]
        a.trusted = a.trim(a.trusted + b.trusted, self.gallery)
        a.candidate = a.trim(a.candidate + b.candidate, max(6, self.gallery // 2))
        a.tracks = list(dict.fromkeys(a.tracks + b.tracks))
        a.cameras.update(b.cameras)
        a.geometry = (a.geometry + b.geometry)[-self.gallery:]
        self.face_trusted[winner] = self._merge_face(
            self.face_trusted.get(winner, []), self.face_trusted.get(loser, []), self.face_gallery
        )
        self.face_candidate[winner] = self._merge_face(
            self.face_candidate.get(winner, []), self.face_candidate.get(loser, []), max(4, self.face_gallery // 2)
        )
        for key, gid in list(self.mapping.items()):
            if gid == loser:
                self.mapping[key] = winner
        self.identities.pop(loser, None)
        self.face_trusted.pop(loser, None)
        self.face_candidate.pop(loser, None)
        self.merge_count += 1
        self.merged_pairs.append((loser, winner))

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
                    left = self.identities[left_gid]
                    right = self.identities[right_gid]
                    if self._identity_overlap(left, right, tracks):
                        continue
                    body, support = self.pair(left.values(), right.values())
                    face, _ = self.face_pair(
                        self.face_trusted.get(left_gid, []) + self.face_candidate.get(left_gid, []),
                        self.face_trusted.get(right_gid, []) + self.face_candidate.get(right_gid, []),
                    )
                    if (body >= self.merge_body and support >= self.merge_support) or face >= self.merge_face:
                        # Keep the identity with the larger trusted gallery as the stable root.
                        if len(left.trusted) + len(left.candidate) >= len(right.trusted) + len(right.candidate):
                            self._merge_pair(left_gid, right_gid)
                        else:
                            self._merge_pair(right_gid, left_gid)
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

        # Give weaker tracklets a second chance after all high-evidence identities exist.
        for track in sorted(tracks.values(), key=lambda x: (x.evidence(), x.start, x.camera, x.key)):
            if track.key in self.mapping:
                continue
            ranked = self.search(track, faces_by_track.get(track.key, []), tracks)
            if not ranked:
                continue
            best = ranked[0]
            second = ranked[1]["score"] if len(ranked) > 1 else 0.0
            if (
                best["body"] >= self.rescue_body
                and best["obs_support"] >= self.rescue_support
                and best["stable_support"] >= self.rescue_stable
                and best["score"] - second >= self.rescue_margin
            ):
                gid = best["gid"]
                self.mapping[track.key] = gid
                self.identities[gid].add(track, False, self.gallery, self.promote)
                self.add_face(gid, faces_by_track.get(track.key, []), False)
                self.decisions.append(DecisionV5(
                    track.key, gid, "body_second_chance", float(best["score"]), float(best["score"] - second),
                    float(best["body"]), float(best["face"]), int(best["support"]), int(best["stable_support"]),
                    int(best["face_support"]), float(best.get("face_margin", 0.0)), float(second), track.camera,
                ))

        # Conservative identity-level merge catches fragmented sitting/standing/re-entry tracklets.
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
        face_assisted = sum("face" in x.reason for x in accepted)
        strong_body = sum(x.body >= self.strong for x in accepted)
        near = sum(
            1 for x in self.decisions
            if x.gid == "PENDING" and self.rescue_body - 0.03 <= x.body < self.threshold
        )
        score_buckets = {"<0.50": 0, "0.50-0.55": 0, "0.55-0.60": 0, "0.60-0.65": 0, "0.65-0.70": 0, "0.70-0.75": 0, ">=0.75": 0}
        for item in self.decisions:
            value = item.body
            if value < 0.50:
                score_buckets["<0.50"] += 1
            elif value < 0.55:
                score_buckets["0.50-0.55"] += 1
            elif value < 0.60:
                score_buckets["0.55-0.60"] += 1
            elif value < 0.65:
                score_buckets["0.60-0.65"] += 1
            elif value < 0.70:
                score_buckets["0.65-0.70"] += 1
            elif value < 0.75:
                score_buckets["0.70-0.75"] += 1
            else:
                score_buckets[">=0.75"] += 1
        return {
            "tracklets": len(tracks),
            "global_ids": len(set(self.mapping.values())),
            "multi_camera": multi,
            "multi_camera_count": len(multi),
            "reasons": reasons,
            "reidentified": sum(x.reason != "new" and x.gid != "PENDING" for x in self.decisions),
            "face_assisted": face_assisted,
            "strong_body_matches": strong_body,
            "near_threshold_pending": near,
            "score_buckets": score_buckets,
            "identity_merges": self.merge_count,
            "trusted_body": sum(len(x.trusted) for x in self.identities.values()),
            "candidate_body": sum(len(x.candidate) for x in self.identities.values()),
            "trusted_face": sum(len(x) for x in self.face_trusted.values()),
            "candidate_face": sum(len(x) for x in self.face_candidate.values()),
        }
