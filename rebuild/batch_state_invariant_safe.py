from __future__ import annotations

from rebuild.batch_state_invariant import BatchPipelineStateInvariant
from rebuild.identity_v2 import quality as cropquality


class BatchPipelineStateInvariantSafe(BatchPipelineStateInvariant):
    """State-invariant V6 with quality-aware local evidence retention.

    The detector confidence remains the detector confidence in observation
    metadata. After the parent pipeline has accepted an observation, its actual
    crop quality is applied to the corresponding stored features so the local
    V6 gallery favors useful person crops rather than merely high-confidence
    detections. This is intentionally presentation/inference-neutral: the
    multimodel state resolver and global matching thresholds are unchanged.
    """

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

        # Correct feature-selection quality using the actual crop-quality score
        # while preserving the true detector confidence separately in metadata.
        for feature in track.features:
            if feature.camera == camera and abs(float(feature.stamp) - target) <= eps:
                feature.quality = measured

        for observation in track.observations:
            if abs(float(observation.get("timestamp", -1.0)) - target) <= eps:
                observation["crop_quality"] = measured

        if len(track.features) > self.bank:
            track.trim(max(1, int(self.bank)))
