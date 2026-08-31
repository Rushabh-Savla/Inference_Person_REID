from __future__ import annotations

import json
import cv2

from rebuild.batch_state_invariant import BatchPipelineStateInvariant
from rebuild.batch_v6 import BatchPipelineV6


class BatchPipelineStateInvariantSafe(BatchPipelineStateInvariant):
    """Fresh-run V6 with quality-aware evidence and clear GID rendering."""

    def sources(self, values):
        """Accept explicit CAMERA=RTSP_URL inputs while preserving file paths."""
        if values:
            out = []
            used = set()
            for value in values:
                text = str(value)
                if "=" in text:
                    name, source = text.split("=", 1)
                    if name and source:
                        if name in used:
                            raise SystemExit(f"Duplicate camera name: {name}")
                        used.add(name)
                        out.append((name, source))
                        continue
                camera = text.rsplit("/", 1)[-1] or text
                if camera in used:
                    raise SystemExit(f"Duplicate camera name: {camera}")
                used.add(camera)
                out.append((camera, text))
            return out
        return super().sources(values)

    @staticmethod
    def short_gid(gid: str) -> str:
        """Display G000012 as G12 while preserving the canonical internal GID."""
        text = str(gid)
        if text.startswith("G") and text[1:].isdigit():
            return f"G{int(text[1:])}"
        return text

    @staticmethod
    def gid_colour(gid: str):
        """Stable BGR palette; G2 is pink and G3 is blue."""
        palette = {
            1: (66, 199, 125),
            2: (203, 74, 221),
            3: (255, 90, 30),
            4: (0, 165, 255),
            5: (211, 85, 186),
            6: (0, 215, 255),
            7: (255, 191, 0),
            8: (80, 80, 220),
        }
        text = str(gid)
        number = int(text[1:]) if text.startswith("G") and text[1:].isdigit() else 0
        return palette.get(number, palette[((number - 1) % len(palette)) + 1])

    @staticmethod
    def label_text_colour(background):
        b, g, r = background
        luminance = 0.114 * b + 0.587 * g + 0.299 * r
        return (0, 0, 0) if luminance >= 165 else (255, 255, 255)

    def add_body(self, key, camera, track_id, segment, bbox, stamp, score, image, feats, multi):
        """Bridge state-invariant multimodel input to the base V6 feature API."""
        measured = float(__import__("rebuild.identity_v2", fromlist=["quality"]).quality(image))
        BatchPipelineV6.add_body(
            self,
            key,
            camera,
            track_id,
            segment,
            bbox,
            stamp,
            measured,
            image,
            feats,
        )

        track = self.tracks[key]
        if not hasattr(track, "state_bank"):
            track.state_bank = {
                "resnet": {kind: [] for kind in self.VARIANTS},
                "swin": {kind: [] for kind in self.VARIANTS},
                "solider": {kind: [] for kind in self.VARIANTS},
            }
        if not hasattr(track, "colour_bank"):
            track.colour_bank = []

        for observation in reversed(track.observations):
            if abs(float(observation.get("timestamp", -1.0)) - float(stamp)) <= 1e-6:
                observation["detection_score"] = float(score)
                observation["crop_quality"] = measured
                break

        for kind, vector in feats.items():
            if kind in track.state_bank["resnet"]:
                track.state_bank["resnet"][kind].append(vector)
                track.state_bank["resnet"][kind] = track.state_bank["resnet"][kind][-48:]

        for model in ("swin", "solider"):
            for kind, vectors in multi.get(model, {}).items():
                track.state_bank[model][kind].extend(vectors)
                track.state_bank[model][kind] = track.state_bank[model][kind][-48:]

        signature = self.colour_signature(image)
        if signature is not None:
            track.colour_bank.append(signature)
            track.colour_bank = track.colour_bank[-24:]
            mixed = __import__("numpy").mean(__import__("numpy").stack(track.colour_bank), axis=0)
            track.colour_signature = mixed / (__import__("numpy").linalg.norm(mixed) + 1e-12)

    @staticmethod
    def colour_signature(image):
        import numpy as np
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        if h < 40 or w < 20:
            return None
        x1, x2 = int(0.16 * w), int(0.84 * w)
        y1, y2 = int(0.08 * h), int(0.72 * h)
        torso = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1].astype(np.float32) / 255.0
        val = hsv[..., 2].astype(np.float32) / 255.0
        hue = hsv[..., 0].astype(np.float32) / 180.0
        hue_hist, _ = np.histogram(hue, bins=12, range=(0.0, 1.0), weights=sat + 0.05)
        neutral = sat < 0.28
        value_hist, _ = np.histogram(val[neutral], bins=5, range=(0.0, 1.0))
        desc = np.concatenate([hue_hist, value_hist]).astype(np.float32)
        return desc / (np.linalg.norm(desc) + 1e-12)

    def render(self, mapping):
        """Render final detections with short, stable, colour-coded GID labels."""
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
                        if gid == "UNKNOWN" or not str(gid).startswith("G"):
                            colour = (145, 145, 145)
                            label = str(gid)
                        else:
                            colour = self.gid_colour(gid)
                            label = self.short_gid(gid)
                        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                        scale, thickness = 0.68, 2
                        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                        pad_x, pad_y = 9, 7
                        bx1 = max(0, x1)
                        by2 = max(th + base + 2, y1)
                        by1 = max(0, by2 - th - base - 2 * pad_y)
                        bx2 = min(image.shape[1] - 1, bx1 + tw + 2 * pad_x)
                        by2 = min(image.shape[0] - 1, by2)
                        cv2.rectangle(image, (bx1, by1), (bx2, by2), colour, -1)
                        text_colour = self.label_text_colour(colour)
                        cv2.putText(image, label, (bx1 + pad_x, by2 - pad_y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, text_colour, thickness, cv2.LINE_AA)
                    cv2.putText(image, camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    writer.write(image)
            finally:
                cap.release()
                writer.release()
            print(f"[safe-v6] wrote {out}")
