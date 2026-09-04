from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from rebuild.batch_v4 import BatchPipelineV4
from rebuild.identity_v5 import GlobalIdentityV5


class BatchPipelineV5(BatchPipelineV4):
    """V5: keep V4 feature collection, replace only the identity decision layer."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.cache = self.out / "cache_v5"
        self.crops = self.cache / "crops"
        self.faces_dir = self.cache / "faces"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.crops.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.engine = GlobalIdentityV5(self.cfg["identity_v5"])

    def save_cache(self):
        info = []
        packed = {}
        face_info = {}
        for key, track in sorted(self.tracks.items()):
            features = []
            for feature in track.features:
                features.append({
                    "kind": feature.kind,
                    "quality": feature.quality,
                    "camera": feature.camera,
                    "timestamp": feature.stamp,
                    "meta": feature.meta,
                })
            info.append({
                "key": key,
                "camera": track.camera,
                "track_id": track.track_id,
                "segment": track.segment,
                "start": track.start,
                "end": track.end,
                "shape": track.shape,
                "count": track.count(),
                "features": features,
            })
            if track.features:
                packed[key] = np.stack([x.vector for x in track.features]).astype(np.float32)
            face_info[key] = [
                {"quality": x.quality, "timestamp": x.stamp, "meta": x.meta}
                for x in self.faces.get(key, [])
            ]
        (self.cache / "tracklets_v5.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        np.savez_compressed(self.cache / "tracklets_v5.npz", **packed)
        (self.cache / "faces_v5.json").write_text(json.dumps(face_info, indent=2), encoding="utf-8")
        face_pack = {
            key: np.stack([x.vector for x in values]).astype(np.float32)
            for key, values in self.faces.items() if values
        }
        np.savez_compressed(self.cache / "faces_v5.npz", **face_pack)
        (self.cache / "video_meta_v5.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def save_debug(self, decisions):
        with (self.out / "identity_debug_v5.jsonl").open("w", encoding="utf-8") as handle:
            for item in decisions:
                handle.write(json.dumps(item.__dict__) + "\n")
        (self.out / "identity_merges_v5.json").write_text(
            json.dumps(self.engine.merged_pairs, indent=2), encoding="utf-8"
        )

    def save_gallery(self):
        body = {}
        face = {}
        meta = {}
        for gid, identity in self.engine.identities.items():
            if identity.trusted:
                body[gid] = np.stack([x.vector for x in identity.trusted]).astype(np.float32)
            trusted_faces = self.engine.face_trusted.get(gid, [])
            if trusted_faces:
                face[gid] = np.stack([x.vector for x in trusted_faces]).astype(np.float32)
            meta[gid] = {
                "tracks": identity.tracks,
                "cameras": sorted(identity.cameras),
                "trusted_body": len(identity.trusted),
                "candidate_body": len(identity.candidate),
                "trusted_face": len(trusted_faces),
                "candidate_face": len(self.engine.face_candidate.get(gid, [])),
            }
        np.savez_compressed(self.out / "global_body_gallery_v5.npz", **body)
        np.savez_compressed(self.out / "global_face_gallery_v5.npz", **face)
        (self.out / "global_gallery_v5.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def render(self, mapping):
        for camera, meta in self.meta.items():
            cap = cv2.VideoCapture(meta["source"])
            out = self.out / f"{camera}_v5.mp4"
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
            print(f"[v5] wrote {out}")

    def print_summary(self):
        data = self.engine.summary(self.tracks)
        print("\n===== V5 ADAPTIVE REID RESULT =====")
        print(f"tracklets: {data['tracklets']}")
        print(f"global IDs: {data['global_ids']}")
        print(f"reidentified: {data['reidentified']}")
        print(f"multi-camera IDs: {data['multi_camera_count']}")
        print(f"identity merges: {data['identity_merges']}")
        print(f"reasons: {json.dumps(data['reasons'], sort_keys=True)}")
        print(f"near-threshold pending: {data['near_threshold']}")
        print(f"score buckets: {json.dumps(data['score_buckets'], sort_keys=True)}")
        print(f"trusted body features: {data['trusted_body']}")
        print(f"candidate body features: {data['candidate_body']}")
        print(f"trusted face features: {data['trusted_face']}")
        print(f"candidate face features: {data['candidate_face']}")
        print(f"face-assisted matches: {data['face_assisted']}")
        for gid, cams in sorted(data["multi_camera"].items()):
            print(f"  {gid}: {', '.join(cams)}")
        print(f"outputs: {self.out}")

    def run(self, values):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")
        print(f"[v5] Body ReID: {self.extractor.describe()}")
        print(f"[v5] Face ReID: {self.face.describe()}")
        print(f"[v5] cameras: {len(sources)}")
        print("[v5] pass 1: detect + track + continuous body/face feature collection")
        for camera, path in sources:
            self.collect(camera, path)
        self.save_cache()
        print("[v5] pass 2: V3-compatible body matching + adaptive evidence + face rescue + identity merge")
        mapping, decisions = self.engine.run(self.tracks, self.faces)
        self.save_debug(decisions)
        self.save_gallery()
        self.print_summary()
        print("[v5] pass 3: render from saved detections")
        self.render(mapping)
        return mapping
