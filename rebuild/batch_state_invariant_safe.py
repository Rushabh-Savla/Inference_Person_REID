from __future__ import annotations

import cv2

from rebuild.batch_state_invariant import BatchPipelineStateInvariant
from rebuild.identity_v2 import quality as cropquality


class BatchPipelineStateInvariantSafe(BatchPipelineStateInvariant):
    """State-invariant V6 with quality-aware evidence and clear GID rendering."""

    def add_body(
        self,
        key,
        camera,
        track_id,
        segment,
        bbox,
        stamp,
        score,
        image,
        feats,
        multi,
    ):
        super().add_body(
            key,
            camera,
            track_id,
            segment,
            bbox,
            stamp,
            score,
            image,
            feats,
            multi,
        )

        track = self.tracks[key]
        measured = float(cropquality(image))
        target = float(stamp)
        eps = 1e-6

        # Use actual crop quality for feature retention while preserving detector
        # confidence separately in the observation metadata.
        for feature in track.features:
            if feature.camera == camera and abs(float(feature.stamp) - target) <= eps:
                feature.quality = measured

        for observation in track.observations:
            if abs(float(observation.get("timestamp", -1.0)) - target) <= eps:
                observation["crop_quality"] = measured

        if len(track.features) > self.bank:
            track.trim(max(1, int(self.bank)))

    @staticmethod
    def short_gid(gid: str) -> str:
        """Display G000012 as G12 while preserving the canonical internal GID."""
        text = str(gid)
        if text.startswith("G") and text[1:].isdigit():
            return f"G{int(text[1:])}"
        return text

    @staticmethod
    def gid_colour(gid: str):
        """Stable BGR palette; G2 is pink and G3 is blue as requested."""
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
                    import json
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

                        scale = 0.68
                        thickness = 2
                        (tw, th), base = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
                        )
                        pad_x, pad_y = 9, 7
                        bx1 = max(0, x1)
                        by2 = max(th + base + 2, y1)
                        by1 = max(0, by2 - th - base - 2 * pad_y)
                        bx2 = min(image.shape[1] - 1, bx1 + tw + 2 * pad_x)
                        if by1 < 0:
                            by1 = 0
                            by2 = min(image.shape[0] - 1, by1 + th + base + 2 * pad_y)

                        cv2.rectangle(image, (bx1, by1), (bx2, by2), colour, -1)
                        text_colour = self.label_text_colour(colour)
                        text_x = bx1 + pad_x
                        text_y = by2 - pad_y - base
                        cv2.putText(
                            image,
                            label,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale,
                            text_colour,
                            thickness,
                            cv2.LINE_AA,
                        )

                    cv2.putText(
                        image,
                        camera,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    writer.write(image)
            finally:
                cap.release()
                writer.release()

            print(f"[safe-v6] wrote {out}")
