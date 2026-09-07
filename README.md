# Multi-Camera Person Re-Identification

Production-oriented multi-camera Person Re-Identification (ReID / MTMC) pipeline using YOLO11m, ByteTrack, NVIDIA ResNet-50, NVIDIA Swin Base, SOLIDER, attribute evidence, trajectory/geometry signals, severe-overlap protection, persistent SQLite identity state, and annotated video output.

## 1. System Architecture

```text
Video / RTSP
     │
     ▼
YOLO11m Person Detection
     │
     ▼
ByteTrack Camera-Local Tracking
     │
     ├──► Position / Trajectory History
     │
     ▼
State-aware Tracklets
     │
     ▼
Quality-filtered Body Crops
     │
     ├── Full
     ├── Light
     ├── Upper
     ├── Torso
     └── Lower
     │
     ▼
┌───────────────────────────────┐
│ ReID Feature Extraction       │
│ NVIDIA ResNet-50              │
│ NVIDIA Swin Base              │
│ SOLIDER Swin Base             │
└───────────────────────────────┘
     │
     ▼
Appearance State Banks
     │
     ▼
Protected V6 Local Identity Proposals
     │
     ├── Same-camera tracker-reset repair
     │
     ├── Severe-overlap protection
     │        └── Dense post-overlap recovery
     │
     ▼
State-invariant MTMC Resolver
     │
     ├── ReID
     ├── Attributes / Colour
     ├── Temporal compatibility
     ├── Geometry
     └── Trajectory
     │
     ▼
Persistent SQLite Identity Registry
     │
     ▼
Global IDs
     │
     ▼
Annotated MP4
```

## 2. Repository Structure

```text
.
├── rebuild/
│   ├── run.py
│   ├── config_state_invariant.yaml
│   ├── batch_state_invariant_*.py
│   ├── multimodel_state_invariant_*.py
│   ├── identity_body_v6.py
│   └── compatibility runners
├── src/
│   ├── detector.py
│   └── reid/
│       ├── nvidia_reid.py
│       ├── nvidia_swin.py
│       └── solider_reid.py
├── identity_state/
│   └── reid_state_invariant_v4.sqlite3
├── weights/
│   └── model artifacts
├── recordings/
│   └── live capture sessions
├── requirements.txt
└── README.md
```

Legacy/development directories may exist in the repository, but they are not part of the maintained runtime.

## 3. Requirements

Important package versions documented for the maintained environment:

```text
ultralytics       8.4.89
torch             2.12.1
torchvision       0.27.1
torchreid         0.2.5
onnxruntime-gpu   1.28.0
numpy             2.2.6
opencv-python     5.0.0.93
PyYAML            6.0.3
lap               0.5.13
qdrant-client     1.18.0
```

The complete dependency list is in `requirements.txt`.

System requirements include a working NVIDIA driver/GPU and, for live processing, `ffmpeg` and `ffprobe`.

## 4. Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify GPU:

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Torch:', torch.__version__)"
```

Verify ONNX Runtime:

```bash
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

`CUDAExecutionProvider` must be available.

Verify project import:

```bash
PYTHONPATH="$PWD/src:$PWD" python -c "import rebuild.run; print('rebuild.run import: OK')"
```

## 5. Model Weights

Models are not committed to Git.

Required files:

```text
weights/yolo11m.pt

weights/reid/resnet50_market1501_aicity156.onnx

weights/reid/nvidia_swin_base_1024/export_55/
    swin_base_market1501_aicity156_featuredim1024.onnx

weights/solider_swin_base_msmt17.onnx
```

Project weight source:

https://drive.google.com/drive/folders/1jCESnjj5g2WsJRLTqaMuMi9vg2OghYdP?usp=sharing

A `.pth` Swin checkpoint may exist as a training artifact, but it is not the active inference model.

Verify files:

```bash
ls -lh   weights/yolo11m.pt   weights/reid/resnet50_market1501_aicity156.onnx   weights/reid/nvidia_swin_base_1024/export_55/swin_base_market1501_aicity156_featuredim1024.onnx   weights/solider_swin_base_msmt17.onnx
```

## 6. Detection and Tracking

Configured detector/tracker path:

```yaml
model: weights/yolo11m.pt
conf: 0.55
iou: 0.60
tracker: bytetrack.yaml
```

Flow:

```text
YOLO11m → person detections → ByteTrack → local track IDs → tracklets
```

A ByteTrack ID is camera-local and temporary. It must never be treated as a permanent identity.

## 7. ReID Sampling and Multi-view Representation

Normal ReID observations:

```text
interval = 2 frames
part_interval = 4 frames
min_quality = 0.20
```

Body views:

```text
full
light
upper
torso
lower
```

The multi-view representation improves robustness to partial visibility, illumination changes, sitting, and changed framing.

## 8. ReID Models

### NVIDIA ResNet-50

```text
weights/reid/resnet50_market1501_aicity156.onnx
```

### NVIDIA Swin Base

```text
weights/reid/nvidia_swin_base_1024/export_55/
swin_base_market1501_aicity156_featuredim1024.onnx
```

### SOLIDER Swin Base

```text
weights/solider_swin_base_msmt17.onnx
```

The normal cross-camera decision requires support from multiple models.

```text
state_cross_resnet_min  = 0.44
state_cross_swin_min    = 0.44
state_cross_solider_min = 0.42
state_cross_required_models = 2
```

## 9. Identity Resolution

### Local V6 Proposal

```text
match_threshold     = 0.60
match_margin        = 0.035
strong_threshold    = 0.72
support_required    = 2
accumulated_body    = 0.56
accumulated_support = 3
partial_threshold   = 0.58
partial_support     = 2
gallery             = 64
candidate_gallery   = 24
promote_quality     = 0.68
novelty             = 0.985
```

### Same-camera fragment repair

Key settings:

```text
state_same_fused_min             = 0.48
state_same_model_min             = 0.42
state_same_max_gap_sec           = 30.0
state_same_camera_continuity_min = 0.22
state_same_chain_min             = 0.53
state_same_chain_support         = 2
state_same_chain_continuity_min  = 0.28
state_same_trajectory_min        = 0.35
```

Appearance, trajectory, position, direction, and time gap are combined.

### Cross-camera reconciliation

```text
state_cross_fused_min           = 0.50
state_cross_partial_fused_min   = 0.47
state_cross_strong              = 0.80
state_cross_required_models     = 2
state_cross_required_views      = 1
state_cross_max_gap_sec         = 30.0
```

A single cosine score should not independently create a cross-camera identity.

## 10. Attributes

Supporting evidence includes:

```text
Upper-body colour
Lower-body colour
Upper-body pattern
Lower-body pattern
Head/detail descriptor
Eye-region descriptor
Visibility
```

Thresholds:

```text
attribute_color_min   = 0.84
attribute_pattern_min = 0.64
attribute_detail_min  = 0.72
```

Attributes reinforce an already credible ReID decision; they do not independently define identity.

## 11. Severe Overlap Protection

Configured guard:

```yaml
overlap_guard:
  enabled: true
  iou_min: 0.80
  intersection_min: 0.85
  recovery_samples: 4
```

During severe overlap:

```text
Detection       → continues
ByteTrack       → continues
Trajectory      → continues
Overlap state   → recorded
ReID extraction → paused
```

The clean appearance state before overlap is preserved as the pre-overlap anchor.

After overlap:

```text
Overlap clears
    ↓
Dense recovery
    ↓
Clean crops
    ↓
ResNet + Swin + SOLIDER
    ↓
Anchor comparison
    ↓
Identity decision
```

Recovery:

```text
post_overlap_interval_frames = 1
recovery_samples              = 4
required_models               = 2
fused_min                     = 0.56
ResNet minimum                = 0.52
Swin minimum                  = 0.52
SOLIDER minimum               = 0.50
```

Core rule: overlap itself must never cause a GID change.

## 12. Trajectory and Geometry

Each observation can retain:

```text
frame
timestamp
bounding box
center point
body height
```

Trajectory history:

```text
trajectory_history_frames = 30
```

Trajectory and geometry provide temporal/spatial plausibility and support fragment repair. They should not override strong appearance evidence by themselves.

## 13. Persistent Identity State

SQLite database:

```text
identity_state/reid_state_invariant_v4.sqlite3
```

Identity configuration:

```text
model_id  = final-state-invariant-v4
bank_size = 96
state_gallery_match_min = 0.62
state_gallery_margin    = 0.02
```

Back up the database before replacing, deleting, migrating, or changing the identity-state representation. Do not reuse incompatible databases across model/preprocessing/configuration changes.

## 14. Run Recorded Videos

```bash
python rebuild/run.py batch_state_final   --config rebuild/config_state_invariant.yaml   --videos     cam_213.mp4     cam_224.mp4     cam_219.mp4
```

All supplied cameras are processed in one shared session.

## 15. Run Live RTSP

```bash
python rebuild/run.py live   --config rebuild/config_state_invariant.yaml   --sources     cam_213=rtsp://USER:PASSWORD@HOST/ch13/0     cam_224=rtsp://USER:PASSWORD@HOST/ch24/0     cam_219=rtsp://USER:PASSWORD@HOST/ch19/0
```

Use your actual RTSP URLs. `ffmpeg` and `ffprobe` must be available.

Live recordings are stored below:

```text
recordings/live_state_<timestamp>/
```

## 16. Outputs

Default output area:

```text
rebuild_outputs_state_invariant/
```

Typical contents:

```text
camera_*_output.mp4
state_invariant_debug.json
cache / metadata
```

Diagnostics may include tracklet, overlap, recovery, trajectory, and post-overlap score information.

## 17. Validation Checklist

Before real workload:

```bash
PYTHONPATH="$PWD/src:$PWD" python -c "import rebuild.run; print('rebuild.run import: OK')"

python -c "import torch; print(torch.cuda.is_available())"

python -c "import onnxruntime as ort; print(ort.get_available_providers())"

ls -lh   weights/yolo11m.pt   weights/reid/resnet50_market1501_aicity156.onnx   weights/reid/nvidia_swin_base_1024/export_55/swin_base_market1501_aicity156_featuredim1024.onnx   weights/solider_swin_base_msmt17.onnx
```

End-to-end verification must cover camera opening, detection, tracking, quality filtering, all three ReID models, local identity creation, same-camera repair, overlap protection, recovery, cross-camera matching, persistent state updates, GID consistency, and final annotated video quality.

## 18. Engineering Rules

```text
REPRODUCE FIRST
VALIDATE SECOND
OPTIMIZE THIRD
```

Evaluate identity consistency, cross-camera matching, fragment repair, false merges, overlap recovery, temporal consistency, runtime, GPU use, and output quality.

The most important separation is:

```text
Detection
≠ Tracking
≠ Tracklet
≠ ReID embedding
≠ Identity matching
≠ Global ID
```
