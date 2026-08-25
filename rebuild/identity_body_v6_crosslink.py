from __future__ import annotations

from typing import Dict, List

from rebuild.identity_body_v6_verified import GlobalIdentityBodyV6Verified
from rebuild.identity_v3 import Tracklet


class GlobalIdentityBodyV6CrossLink(GlobalIdentityBodyV6Verified):
    """V6 verifier with a final high-confidence cross-camera link constraint.

    The earlier verifier deliberately refused to alter already-strong V6
    appearance scores. That is unsafe for the observed 222/224 swap: a wrong
    strong appearance score can still beat the correct identity while the
    synchronized one-to-one track pairing knows the opposite.

    This layer therefore applies the validated cross-camera link constraint
    after the normal V6 ranking for every cross-camera candidate, including
    strong body matches. It remains conservative because a link must first
    survive the complete-corpus one-to-one matching and both endpoint margin
    tests in GlobalIdentityBodyV6Verified.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cross_link_override_bonus = float(cfg.get("cross_camera_link_override_bonus", 0.28))
        self.cross_link_override_penalty = float(cfg.get("cross_camera_link_override_penalty", 0.28))
        self.cross_link_override_min = float(cfg.get("cross_camera_link_override_min", 0.56))

    def rank(self, track: Tracklet, tracks: Dict[str, Tracklet]) -> List[dict]:
        rows = super().rank(track, tracks)
        if not rows:
            return rows

        link = self.cross_links.get(track.key)
        if link is None:
            return rows

        target = tracks.get(link["other"])
        if target is None:
            return rows
        if float(link["score"]) < self.cross_link_override_min:
            return rows

        adjusted: List[dict] = []
        for row in rows:
            item = dict(row)
            identity = self.identities[item["gid"]]
            if track.camera in identity.cameras:
                adjusted.append(item)
                continue

            target_matches = target.key in identity.tracks
            if target_matches:
                item["score"] += self.cross_link_override_bonus
                item["cross_link_override"] = "support"
            else:
                item["score"] -= self.cross_link_override_penalty
                item["cross_link_override"] = "conflict"
            adjusted.append(item)

        return sorted(adjusted, key=lambda x: x["score"], reverse=True)
