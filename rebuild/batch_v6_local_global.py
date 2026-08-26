from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rebuild.batch_v6 import BatchPipelineV6
from rebuild.identity_body_v6 import GlobalIdentityBodyV6
from rebuild.v6_local_global_safe import SafeLocalGlobalResolver
from rebuild.v6_local_global import colour_signature


class BatchPipelineV6LocalGlobal(BatchPipelineV6):
    """V6 NVIDIA ReID with camera-local identity isolation and global reconciliation.

    Each camera is solved independently with the unchanged V6 identity engine.
    Only then are local identities compared across cameras. This prevents one
    camera's mistaken global assignment from contaminating another camera.
    """

    def add_body(self, key, camera, track_id, segment, bbox, stamp, score, image, feats):
        signature = colour_signature(image)
        super().add_body(key, camera, track_id, segment, bbox, stamp, score, image, feats)
        track = self.tracks.get(key)
        if track is not None and signature is not None:
            old = getattr(track, "colour_signature", None)
            if old is None:
                track.colour_signature = signature
            else:
                mixed = 0.75 * np.asarray(old, dtype=np.float32) + 0.25 * signature
                mixed /= np.linalg.norm(mixed) + 1e-12
                track.colour_signature = mixed.astype(np.float32)

    def local_assign(self, tracks: Dict[str, object], cameras: List[str]):
        local_mapping: Dict[str, str] = {}
        engines: Dict[str, GlobalIdentityBodyV6] = {}
        for camera in cameras:
            subset = {key: track for key, track in tracks.items() if track.camera == camera}
            engine = GlobalIdentityBodyV6(self.cfg["identity_v6"])
            mapping, _ = engine.run(subset)
            local_mapping.update(mapping)
            engines[camera] = engine
            print(f"[local] {camera}: tracklets={len(subset)} local_ids={len(set(mapping.values()))}")
        return local_mapping, engines

    def save_local_debug(self, local_mapping, global_mapping, components, edges):
        payload = {
            "local_mapping": local_mapping,
            "global_mapping": global_mapping,
            "components": components,
            "edges": edges,
        }
        (self.out / "local_global_v6.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def node_camera(node_key: str) -> str:
        return node_key.split("::", 1)[0]

    def run(self, values):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")
        cameras = [camera for camera, _ in sources]
        print(f"[v6-local-global] {self.extractor.describe()}")
        print("[v6-local-global] FACE: OFF")
        print(f"[v6-local-global] cameras: {', '.join(cameras)}")
        print("[v6-local-global] pass 1: detect + track + NVIDIA body embeddings")
        for camera, path in sources:
            self.collect(camera, path)
        self.save_cache()

        print("[v6-local-global] pass 2: independent per-camera V6 identity solving")
        local_mapping, _ = self.local_assign(self.tracks, cameras)

        print("[v6-local-global] pass 3: camera-local cross-camera reconciliation")
        resolver = SafeLocalGlobalResolver(self.cfg["identity_v6"])
        global_mapping, components, edges = resolver.resolve(local_mapping, self.tracks, cameras)
        self.save_local_debug(local_mapping, global_mapping, components, edges)
        print(f"[v6-local-global] cross-camera links accepted: {len(edges)}")
        print(f"[v6-local-global] final global IDs: {len(set(global_mapping.values()))}")

        print("[v6-local-global] pass 4: render final global mapping")
        self.render(global_mapping)

        multi = {}
        for gid, members in components.items():
            cams = sorted({self.node_camera(k) for k in members})
            if len(cams) > 1:
                multi[gid] = cams
        print("MULTI-CAMERA IDS:")
        for gid, cams in sorted(multi.items()):
            print(f"  {gid}: {', '.join(cams)}")
        print(f"outputs: {self.out}")
        return global_mapping
