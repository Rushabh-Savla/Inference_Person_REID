from __future__ import annotations

import json

import cv2

from rebuild.batch_v5 import BatchPipelineV5
from rebuild.identity_v6 import GlobalIdentityV6


class BatchPipelineV6(BatchPipelineV5):
    """V6 batch pipeline: preserve V5 feature collection, replace identity state layer."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.engine = GlobalIdentityV6(self.cfg["identity_v6"])
        self.cache = self.out / "cache_v6"
        self.crops = self.cache / "crops"
        self.faces_dir = self.cache / "faces"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.crops.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)

    def save_debug(self, decisions):
        with (self.out / "identity_debug_v6.jsonl").open("w", encoding="utf-8") as handle:
            for item in decisions:
                handle.write(json.dumps(item.__dict__) + "\n")
        edges = [item.__dict__ for item in self.engine.edges]
        (self.out / "identity_edges_v6.json").write_text(json.dumps(edges, indent=2), encoding="utf-8")

    def save_gallery(self):
        body = {}
        face = {}
        meta = {}
        for gid, identity in self.engine.identities.items():
            if identity.trusted:
                body[gid] = __import__("numpy").stack([x.vector for x in identity.trusted]).astype("float32")
            trusted_faces = self.engine.face_trusted.get(gid, [])
            if trusted_faces:
                face[gid] = __import__("numpy").stack([x.vector for x in trusted_faces]).astype("float32")
            meta[gid] = {
                "tracks": identity.tracks,
                "cameras": sorted(identity.cameras),
                "trusted_body": len(identity.trusted),
                "candidate_body": len(identity.candidate),
                "trusted_face": len(trusted_faces),
                "candidate_face": len(self.engine.face_candidate.get(gid, [])),
            }
        import numpy as np
        np.savez_compressed(self.out / "global_body_gallery_v6.npz", **body)
        np.savez_compressed(self.out / "global_face_gallery_v6.npz", **face)
        (self.out / "global_gallery_v6.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def render(self, mapping):
        for camera, meta in self.meta.items():
            cap = cv2.VideoCapture(meta["source"])
            out = self.out / f"{camera}_v6.mp4"
            writer = cv2.VideoWriter(
                str(out),
                cv2.VideoWriter_fourcc(*"mp4v"),
                meta["fps"],
                (meta["width"], meta["height"]),
            )
            rows = {}
            with (self.cache / f"{camera}.detections.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    rows.setdefault(int(item["frame"]), []).append(item)
            frame = 0
            try:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    frame += 1
                    for item in rows.get(frame, []):
                        gid = mapping.get(item["tracklet_key"], "UNKNOWN")
                        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            image, gid, (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA,
                        )
                    cv2.putText(
                        image, camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    writer.write(image)
            finally:
                cap.release()
                writer.release()
            print(f"[v6] wrote {out}")

    def print_summary(self):
        data = self.engine.summary(self.tracks)
        print("\n===== V6 IDENTITY RESULT =====")
        print(f"tracklets: {data['tracklets']}")
        print(f"global IDs: {data['global_ids']}")
        print(f"new identities: {data['new_identities']}")
        print(f"reidentified tracks: {data['reidentified']}")
        print(f"same-camera reidentifications: {data['same_camera_reassociations']}")
        print(f"recent-lost-track reassociations: {data['recent_lost_track_reassociations']}")
        print(f"cross-camera reidentifications: {data['cross_camera_reidentifications']}")
        print(f"identity merges: {data['identity_merges']}")
        print(f"provisional identities: {data['provisional_identities']}")
        print(f"fragmented identities before/after: track groups={data['fragmented_identity_count']}")
        print(f"face-assisted: {data['face_assisted']}")
        print(f"body-assisted: {data['body_assisted']}")
        print(f"temporal-assisted: {data['temporal_assisted']}")
        print(f"identity edges: {data['edge_count']}")
        print(f"reasons: {json.dumps(data['reasons'], sort_keys=True)}")
        for gid, tracks in sorted(data["fragmented_identities"].items()):
            print(f"  {gid}: {', '.join(tracks)}")
        for gid, cams in sorted(data["multi_camera"].items()):
            print(f"  {gid}: {', '.join(cams)}")
        print(f"outputs: {self.out}")

    def run(self, values):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")
        print(f"[v6] Body ReID: {self.extractor.describe()}")
        print(f"[v6] Face ReID: {self.face.describe()}")
        print(f"[v6] cameras: {len(sources)}")
        print("[v6] pass 1: detect + track + continuous body/face feature collection")
        for camera, path in sources:
            self.collect(camera, path)
        self.save_cache()
        print("[v6] pass 2: provisional track associations + lost-track continuity + global identity clustering")
        mapping, decisions = self.engine.run(self.tracks, self.faces)
        self.save_debug(decisions)
        self.save_gallery()
        self.print_summary()
        print("[v6] pass 3: render from saved detections")
        self.render(mapping)
        return mapping
