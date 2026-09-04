from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from rebuild.v6_local_global import LocalGlobalResolver
from rebuild.identity_v3 import Tracklet


class SafeLocalGlobalResolver(LocalGlobalResolver):
    """Local/global resolver with a hard same-camera uniqueness guard.

    If the local V6 pass ever assigns the same local identity to overlapping
    tracks in the same camera, those tracks are split before cross-camera
    matching. A global identity is never allowed to contain two simultaneous
    observations from one camera.
    """

    @staticmethod
    def _overlap(left: Tracklet, right: Tracklet) -> bool:
        return not (left.end < right.start or right.end < left.start)

    def _split_conflicted_local_ids(self, local_mapping: Dict[str, str], tracks: Dict[str, Tracklet]) -> Dict[str, str]:
        groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for key, gid in local_mapping.items():
            groups[(tracks[key].camera, gid)].append(key)

        out = dict(local_mapping)
        for (camera, gid), keys in groups.items():
            if len(keys) < 2:
                continue
            conflict = False
            for i, left in enumerate(keys):
                for right in keys[i + 1:]:
                    if self._overlap(tracks[left], tracks[right]):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                continue
            for index, key in enumerate(sorted(keys), start=1):
                out[key] = f"{gid}__split_{index:03d}"
        return out

    def resolve(self, local_mapping: Dict[str, str], tracks: Dict[str, Tracklet], cameras: List[str]):
        safe_mapping = self._split_conflicted_local_ids(local_mapping, tracks)
        return super().resolve(safe_mapping, tracks, cameras)
