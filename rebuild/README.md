# Clean ReID Rebuild

This directory is the clean replacement path for the previous experimental V5 stack.
It is intentionally isolated under `rebuild/` until the end-to-end validation on the real footage passes.

## Architecture

```text
video / RTSP
    |
    v
YOLO person detection + ByteTrack
    |
    v
local tracklets
    |
    +---- periodic high-quality crops ----> FastReID SBS R101-IBN
    |                                         |
    |                                         v
    |                                  normalized embeddings
    |
    v
finished tracklets
    |
    v
tracklet-to-tracklet reconciliation
    |
    +---- same camera: temporal non-overlap required
    |
    +---- cross camera: appearance only + reciprocal/strong match
    |
    v
Global Person IDs
    |
    v
re-render from saved detections
```

There is deliberately no Qdrant dependency in the clean batch path. The current workload is small enough that an in-memory NumPy gallery/tracklet graph is simpler, faster, and much easier to debug.

## Why this replaces the previous approach

The old project mixes detector/tracker state, online identity decisions, Qdrant persistence, reconciliation, reranking, verification, pose ensembling, per-camera overrides, and multiple experimental ReID paths. The repository itself contains a very large `reconcile.py`, a large `IdentityService`, a Qdrant store, an empty architecture document, and a tracker configuration that was later reverted. That is too much moving machinery for a three-camera proof-of-correctness system.

The clean path has one identity authority: the tracklet reconciler. A local tracker ID never becomes a global identity by itself. Global IDs are assigned only after multiple ReID observations exist.

## Current model decision

The first clean production candidate is **FastReID SBS ResNet-101-IBN on MSMT17**, because the repository already contains a verified inference backend and the matching checkpoint. Its published MSMT17 baseline is 84.8% Rank-1 / 62.8% mAP. The R50-IBN variant is 83.9% / 60.6%, so R101 is the conservative accuracy-first starting point.

NVIDIA ReIdentificationNet is being evaluated rather than blindly adopted. NVIDIA's current documentation describes a ResNet-50 embedding model trained on Market-1501 plus sampled 2023 AI City MTMC data, supports TAO fine-tuning, and reports Market-1501 Rank-1 up to 95.3% at 2048 dimensions. NVIDIA's Transformer variant uses a Swin backbone, pretrains on about 3.2M unlabeled NV PeopleCrops images, reports Market-1501 Rank-1 of 95.6% (Swin Tiny) and 96.0% (Swin Base), and NVIDIA reports better generalization for target domains that differ substantially from Market/AICity training data. It is also more computationally expensive.

Therefore the decision is:

1. Start the clean pipeline with FastReID SBS R101-IBN so the complete system can be tested immediately on the existing A6000 setup.
2. Benchmark NVIDIA ReIdentificationNet Transformer and the ResNet-50 version on the **same saved crops and tracklets**.
3. Select the final model from measured same-camera and cross-camera ROC/AUC, Rank-1, error rates, GPU memory, and latency on this actual footage.
4. If NVIDIA wins, replace only the ReID backend; the detector, tracker, tracklet store, reconciliation, and render pipeline do not change.

Sources:
- NVIDIA ReIdentificationNet: https://docs.nvidia.com/mms/text/MDX_ReIdentificationNet.html
- NVIDIA ReIdentificationNet Transformer: https://docs.nvidia.com/mms/text/MDX_ReIdentificationNet_Transformer.html
- NVIDIA perception model guide: https://docs.nvidia.com/mms/text/MDX_Perception_Overview.html
- FastReID model zoo: https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md

## Recorded-video mode

Run all videos together so all tracklets participate in one reconciliation graph:

```bash
PYTHONPATH="$PWD:$PWD/SOLIDER-REID" \
python rebuild/run.py batch \
  --videos cam_213.mp4 cam_219.mp4 cam_224.mp4
```

Outputs are written to `rebuild_outputs/`:

```text
rebuild_outputs/
├── cam_213.mp4
├── cam_219.mp4
├── cam_224.mp4
├── track_to_global.json
└── cache/
    ├── cam_213.detections.jsonl
    ├── cam_219.detections.jsonl
    ├── cam_224.detections.jsonl
    ├── cam_*.embeddings.npy
    └── tracklets.json
```

The renderer never runs YOLO again. It reads the saved frame/bbox/tracklet records and only applies final global IDs. This makes visual debugging and identity debugging use exactly the same detections.

## Live mode

The same detector, tracker, ReID extractor and global gallery are used for RTSP sources:

```bash
PYTHONPATH="$PWD:$PWD/SOLIDER-REID" \
python rebuild/run.py live \
  --sources cam_213=rtsp://... cam_219=rtsp://... \
  --show
```

Each camera is an input adapter only. There are no camera-specific thresholds or model rules.

## Validation policy

Do not judge the system by `global ID count`. The required checks are:

- same-camera track fragmentation recovered without merging simultaneous people;
- same real person across cameras receives one ID;
- different people with similar clothing remain separate;
- cross-camera links are backed by held-out evidence rather than tuning the threshold on the final output;
- ID switches and false merges are explicitly counted;
- GPU memory and embedding latency are measured.

The repository already contains a useful calibration principle: backbone comparisons should use identical crops and threshold-free metrics, and cross-camera negatives must come from operator labels rather than camera co-occurrence. The clean system follows that principle.
