# ReID V2 — Multi-View Global Gallery

V2 replaces the old single-centroid/tracklet-only association with a shared global gallery containing diverse high-quality observations.

## Core changes

- Every sampled track observation is embedded continuously at the configured interval.
- The original crop and a conservative CLAHE illumination variant are both considered.
- Tracklet galleries use quality + embedding novelty, so front/rear/side/lighting views are retained instead of only the highest-score frames.
- Global identities are built from those galleries and reconciled in multiple passes.
- Same-camera overlap remains forbidden, but elapsed time is not used as a hard rejection for ReID.
- Geometry is only an auxiliary tie-breaker based on scale-free person-box aspect ratio.
- The system never constructs a literal 3D/360-degree body model; it builds a multi-view appearance gallery from the angles the cameras actually observe.
- Live mode uses the same global gallery concept online.

## Recorded video

```bash
rm -rf rebuild_outputs

PYTHONPATH="$PWD" \
python rebuild/run.py batch \
  --videos cam_213.mp4 cam_219.mp4 cam_224.mp4 \
  2>&1 | tee /tmp/rebuild_v2.log
```

Outputs:

```text
rebuild_outputs/cam_213_v2.mp4
rebuild_outputs/cam_219_v2.mp4
rebuild_outputs/cam_224_v2.mp4
rebuild_outputs/track_to_global_v2.json
rebuild_outputs/identity_matches_v2.jsonl
rebuild_outputs/global_gallery_v2.npz
rebuild_outputs/global_gallery_v2.json
rebuild_outputs/cache/*_v2.*
```

## NVIDIA ReIdentificationNet benchmark

NVIDIA provides a deployable ResNet-50 ReIdentificationNet ONNX model. The official DeepStream documentation specifies 256x128 RGB input, the preprocessing offsets/scale, 256-dimensional features, and L2 normalization of the raw output. Download the ONNX model first, then compare it against the FastReID backend on the exact same cached crops:

```bash
mkdir -p models/nvidia_reid
wget 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/reidentificationnet/deployable_v1.2/files/resnet50_market1501_aicity156.onnx' \
  -O models/nvidia_reid/resnet50_market1501_aicity156.onnx

PYTHONPATH="$PWD" \
python rebuild/benchmark_reid.py \
  --nvidia models/nvidia_reid/resnet50_market1501_aicity156.onnx \
  --videos cam_213.mp4 cam_219.mp4 cam_224.mp4
```

The benchmark deliberately reports embedding separation and latency without inventing cross-camera ground-truth labels. True cross-camera precision/recall requires a manually verified identity-pair file.
