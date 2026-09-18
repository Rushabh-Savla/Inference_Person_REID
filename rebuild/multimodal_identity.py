from __future__ import annotations

from rebuild.multimodal_identity_strict import MultiModalStrict


class MultiModal(MultiModalStrict):
    """Active identity resolver: strict multimodal ReID with feature-only GID assignment."""

    MODELS = ("resnet", "swin", "solider")

    def __init__(self, cfg):
        super().__init__(cfg)
        self.face_threshold = float(self.fth)

    def observe(self, frame, rows, commit=True, recovery=False):
        values = super().observe(
            frame,
            rows,
            commit=commit,
            recovery=recovery,
        )
        if len(values) != len(rows):
            raise RuntimeError(
                "Strict ReID returned a row-count mismatch"
            )
        return {
            int(row["track_id"]): str(value)
            for row, value in zip(rows, values)
        }

    @property
    def profiles(self):
        return self.pro

    @profiles.setter
    def profiles(self, value):
        self.pro = value


__all__ = ["MultiModal"]
