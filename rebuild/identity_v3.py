from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from rebuild.identity_v2 import geometry_similarity, unit


@dataclass
class Feature:
    vector: np.ndarray
    kind: str
    quality: float
    camera: str
    stamp: float
    meta: dict = field(default_factory=dict)


@dataclass
class Tracklet:
    camera: str
    track_id: int
    segment: int
    fps: float
    start: float = 0.0
    end: float = 0.0
    shape: float = 0.0
    features: List[Feature] = field(default_factory=list)
    observations: List[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.camera}:{self.track_id}:{self.segment}"

    def add(self, vector: np.ndarray, kind: str, quality: float, meta: dict, novelty: float, limit: int) -> bool:
        vector = unit(vector)
        if self.features:
            same = [x.vector for x in self.features if x.kind == kind]
            if same:
                best = float(np.max(np.stack(same) @ vector))
                old = max(x.quality for x in self.features if x.kind == kind)
                if best >= novelty and quality <= old + 0.02:
                    return False
        stamp = float(meta.get("timestamp", 0.0))
        first = not self.observations
        self.features.append(Feature(vector, kind, float(quality), self.camera, stamp, dict(meta)))
        self.features = sorted(self.features, key=lambda x: x.quality, reverse=True)[: max(1, int(limit))]
        self.observations.append(dict(meta))
        if first:
            self.start = stamp
            self.end = stamp
        else:
            self.start = min(self.start, stamp)
            self.end = max(self.end, stamp)
        return True

    def count(self) -> int:
        return len(self.features)

    def evidence(self) -> float:
        if not self.features:
            return 0.0
        q = np.asarray([x.quality for x in self.features], dtype=np.float32)
        span = max(0.0, self.end - self.start)
        return float(0.75 * q.mean() + 0.15 * min(1.0, len(self.features) / 12.0) + 0.10 * min(1.0, span / 8.0))


@dataclass
class Identity:
    gid: str
    trusted: List[Feature] = field(default_factory=list)
    candidate: List[Feature] = field(default_factory=list)
    tracks: List[str] = field(default_factory=list)
    cameras: set[str] = field(default_factory=set)
    geometry: List[float] = field(default_factory=list)

    def add(self, track: Tracklet, trusted: bool, bank: int, quality: float) -> None:
        target = self.trusted if trusted else self.candidate
        for item in track.features:
            if item.quality < quality:
                continue
            same = [x.vector for x in target if x.kind == item.kind]
            if same and float(np.max(np.stack(same) @ item.vector)) > 0.985:
                continue
            target.append(item)
        target.sort(key=lambda x: x.quality, reverse=True)
        del target[bank:]
        if track.key not in self.tracks:
            self.tracks.append(track.key)
        self.cameras.add(track.camera)
        if track.shape > 0:
            self.geometry.append(track.shape)
            self.geometry = self.geometry[-bank:]


@dataclass(frozen=True)
class Decision:
    key: str
    gid: str
    reason: str
    score: float
    margin: float
    support: int
    second: float
    camera: str


class GlobalIdentityV3:
    """Persistent global identity search with trusted multi-view galleries."""

    def __init__(self, threshold: float = 0.61, margin: float = 0.035,
                 strong: float = 0.74, support: int = 2, gallery: int = 24,
                 novelty: float = 0.985, promote: float = 0.70,
                 new_count: int = 3):
        self.threshold = float(threshold)
        self.margin = float(margin)
        self.strong = float(strong)
        self.support = int(support)
        self.gallery = int(gallery)
        self.novelty = float(novelty)
        self.promote = float(promote)
        self.new_count = int(new_count)
        self.identities: Dict[str, Identity] = {}
        self.mapping: Dict[str, str] = {}
        self.decisions: List[Decision] = []
        self.next_id = 1

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
                score = 0.82 * score + 0.18 * float(max(parts))
                support += sum(groups[k][1] for k in ("upper", "lower") if k in groups)
            return float(score), int(support)
        parts = [x for x in (upper, lower) if x]
        if not parts:
            return 0.0, 0
        vals = sorted([x[0] for x in parts], reverse=True)
        return float(0.85 * vals[0] + 0.15 * (vals[1] if len(vals) > 1 else vals[0])), int(sum(x[1] for x in parts))

    def score(self, track: Tracklet, identity: Identity) -> Tuple[float, int]:
        return self.pair(track.features, identity.trusted or identity.candidate)

    def search(self, track: Tracklet, tracks: Dict[str, Tracklet]):
        found = []
        for gid, identity in self.identities.items():
            if self.conflict(track, identity, tracks):
                continue
            score, support = self.score(track, identity)
            shape = 0.5
            if identity.geometry and track.shape > 0:
                shape = geometry_similarity(track.shape, float(np.median(identity.geometry)))
            final = float(0.90 * score + 0.10 * shape)
            found.append((final, support, gid, shape))
        found.sort(reverse=True)
        return found

    def assign(self, track: Tracklet, tracks: Dict[str, Tracklet]) -> Decision:
        ranked = self.search(track, tracks)
        if ranked:
            best, support, gid, _ = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else 0.0
            margin = best - second
            accept = best >= self.threshold and ((support >= self.support and margin >= self.margin) or best >= self.strong)
            if accept:
                self.mapping[track.key] = gid
                self.identities[gid].add(track, True, self.gallery, self.promote)
                result = Decision(track.key, gid, "reidentified", best, margin, support, second, track.camera)
                self.decisions.append(result)
                return result

        if track.count() >= self.new_count:
            gid = self.gid()
            self.identities[gid] = Identity(gid)
            self.identities[gid].add(track, True, self.gallery, self.promote)
            self.mapping[track.key] = gid
            result = Decision(track.key, gid, "new", 1.0, 1.0, track.count(), 0.0, track.camera)
            self.decisions.append(result)
            return result

        result = Decision(track.key, "PENDING", "pending", ranked[0][0] if ranked else 0.0,
                          (ranked[0][0] - ranked[1][0]) if len(ranked) > 1 else 0.0,
                          ranked[0][1] if ranked else 0,
                          ranked[1][0] if len(ranked) > 1 else 0.0,
                          track.camera)
        self.decisions.append(result)
        return result

    def run(self, tracks: Dict[str, Tracklet]) -> Tuple[Dict[str, str], List[Decision]]:
        usable = {k: v for k, v in tracks.items() if v.count() >= self.new_count}
        order = sorted(usable.values(), key=lambda x: (-x.evidence(), x.camera, x.start, x.key))
        for track in order:
            self.assign(track, usable)

        pending = [x for x in tracks.values() if x.key not in self.mapping and x.count() > 0]
        for track in sorted(pending, key=lambda x: (-x.evidence(), x.camera, x.key)):
            self.assign(track, tracks)

        for track in sorted(tracks.values(), key=lambda x: (x.start, x.camera, x.key)):
            if track.key in self.mapping:
                self.identities[self.mapping[track.key]].add(track, False, self.gallery, self.promote)
        return dict(self.mapping), list(self.decisions)

    def gallery_map(self) -> Dict[str, np.ndarray]:
        out = {}
        for gid, identity in self.identities.items():
            values = [x.vector for x in identity.trusted]
            if values:
                out[gid] = np.stack(values).astype(np.float32)
        return out

    def summary(self, tracks: Dict[str, Tracklet]) -> dict:
        groups: Dict[str, set] = {}
        for key, gid in self.mapping.items():
            groups.setdefault(gid, set()).add(tracks[key].camera)
        multi = {gid: sorted(cams) for gid, cams in groups.items() if len(cams) > 1}
        reasons: Dict[str, int] = {}
        for item in self.decisions:
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
        return {"tracklets": len(tracks), "global_ids": len(set(self.mapping.values())),
                "multi_camera": multi, "multi_camera_count": len(multi),
                "reasons": reasons,
                "gallery_features": sum(len(x.trusted) for x in self.identities.values()),
                "candidate_features": sum(len(x.candidate) for x in self.identities.values())}
