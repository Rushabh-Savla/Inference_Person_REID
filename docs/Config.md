# Configuration and Setup Guide

This document describes how to configure the **main-branch Person ReID repository** without assuming any particular workstation, GPU server, camera IP, username, directory layout, or operating environment.

The repository currently contains two important configuration families:

```text
rebuild/config_state_invariant.yaml
    → maintained State-Invariant V6 / Safe055 rebuild path

config.yaml
    → broader root modular stack used by main.py
```

They are related historically, but they are not interchangeable.

---

# 1. Recommended configuration for the maintained state-final pipeline

Use:

```text
rebuild/config_state_invariant.yaml
```

with:

```bash
python rebuild/run.py batch_state_final \
  --config rebuild/config_state_invariant.yaml \
  --videos <video_1> <video_2> ...
```

or:

```bash
python rebuild/run.py live \
  --config rebuild/config_state_invariant.yaml \
  --sources \
    cam_1=<source_1> \
    cam_2=<source_2>
```

---

# 2. Environment installation

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pinned runtime dependencies:

```text
ultralytics       8.4.89
torch             2.12.1
torchvision       0.27.1
torchreid         0.2.5
scipy             1.15.3
gdown             5.2.2
tensorboard       2.21.0
onnxruntime-gpu   1.28.0
lap               0.5.13
qdrant-client     1.18.0
numpy             2.2.6
opencv-python     5.0.0.93
PyYAML            6.0.3
```

FastReID is vendored in the source tree.

For the state-final ONNX path, keep the CUDA-enabled ONNX Runtime installation intact.

---

# 3. Hardware and runtime prerequisites

The repository does not hard-code one machine.

A practical deployment requires:

```text
Python 3.x environment compatible with the pinned packages
NVIDIA GPU for the intended CUDA ReID runtime
Compatible NVIDIA driver / CUDA stack
OpenCV
FFmpeg / FFprobe for live workflows
```

Verify:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

For the NVIDIA ONNX pipeline, confirm:

```text
CUDAExecutionProvider
```

is present.

---

# 4. Model artifact setup

## 4.1 State-final weights

Expected files:

```text
weights/yolo11m.pt

weights/reid/resnet50_market1501_aicity156.onnx

weights/reid/nvidia_swin_base_1024/export_55/
    swin_base_market1501_aicity156_featuredim1024.onnx

weights/solider_swin_base_msmt17.onnx
```

Project weight bundle documented by the repository:

```text
https://drive.google.com/drive/folders/1jCESnjj5g2WsJRLTqaMuMi9vg2OghYdP?usp=sharing
```

Create missing directories as needed:

```bash
mkdir -p \
  weights/reid/nvidia_swin_base_1024/export_55
```

Then place each artifact at exactly the path referenced by the configuration.

---

# 5. Detector configuration

```yaml
detector:
  model: weights/yolo11m.pt
  conf: 0.55
  iou: 0.60
  tracker: bytetrack.yaml
```

### `model`

Path to the YOLO detector weights.

### `conf`

Detection confidence threshold.

Higher values:

```text
fewer detections
higher average confidence
```

Lower values:

```text
more detections
more low-confidence boxes
potentially more false positives
```

Do not tune this from ReID failures alone.

### `iou`

NMS IoU threshold.

This is a detector post-processing parameter. It is not the same thing as the ReID similarity threshold.

### `tracker`

The state-final path uses:

```text
bytetrack.yaml
```

The resulting track IDs are camera-local.

---

# 6. ReID configuration

```yaml
reid:
  model: nvidia_reidentificationnet
  weights: weights/reid/resnet50_market1501_aicity156.onnx
  device: cuda
  max_batch: 32
  interval: 2
  part_interval: 4
  illumination_variant: true
  min_quality: 0.20
```

### `model`

Selects the feature extractor implementation expected by the rebuild stack.

### `weights`

Primary NVIDIA ReID model.

### `device`

For the maintained state-final configuration:

```text
cuda
```

should be used on a supported GPU deployment.

### `max_batch`

Maximum number of crops embedded in one batch.

Larger:

```text
better GPU utilization
higher memory usage
```

Smaller:

```text
lower peak memory
potentially lower throughput
```

### `interval`

Normal embedding interval in processed frames for the rebuild configuration.

### `part_interval`

Sampling interval for additional body views.

### `illumination_variant`

Whether the illumination-variant crop is generated.

### `min_quality`

Minimum accepted crop-quality level used by the rebuild identity pipeline.

This is not the same as YOLO confidence.

---

# 7. Cross-camera models

```yaml
cross_camera_models:
  swin_weights: weights/reid/nvidia_swin_base_1024/export_55/swin_base_market1501_aicity156_featuredim1024.onnx
  solider_weights: weights/solider_swin_base_msmt17.onnx
  swin_batch: 16
  solider_batch: 16
```

These models provide independent evidence for cross-camera matching.

The configured state-final rule is:

```text
state_cross_required_models = 2
```

Do not assume that a change to one model preserves the old threshold calibration.

---

# 8. Identity and Qdrant configuration

Qdrant is the project's persistent vector store for identity embeddings and candidate retrieval. It belongs to the identity layer, not the detector or the ReID models themselves.

The repository's Qdrant configuration follows this pattern:

```yaml
store:
  enabled: true
  path: qdrant_data
  url: http://localhost:6333
```

### `enabled`

Controls whether persistent vector storage is enabled.

### `path`

Local storage location used by the Qdrant deployment when persistent local data is enabled.

### `url`

Address of the Qdrant service used by the application.

Keep the Qdrant collection and embedding space consistent with the active ReID model configuration.

A change to the ReID model, checkpoint, preprocessing, embedding dimension, embedding tap, or identity configuration can make existing vectors incompatible and may require a new collection or a deliberate migration.

Qdrant performs vector storage and similarity retrieval; the application's identity matching/resolution logic still makes the final identity decision.

# 9. V6 local identity settings

```yaml
identity_v6:
  match_threshold: 0.60
  match_margin: 0.035
  strong_threshold: 0.72

  support_required: 2

  accumulated_body: 0.56
  accumulated_support: 3

  partial_threshold: 0.58
  partial_support: 2

  gallery: 64
  candidate_gallery: 24

  promote_quality: 0.68
  novelty: 0.985
```

These values govern local identity proposal / accumulated evidence.

Treat them as a calibrated set.

Do not raise one number casually because a video “looks wrong.”

---

# 10. Same-camera configuration

```yaml
identity_v6:
  same_camera_gap_sec: 15.0
  same_camera_distance: 5.0
  same_camera_min_continuity: 0.35

  state_same_fused_min: 0.48
  state_same_model_min: 0.42
  state_same_max_gap_sec: 30.0
  state_same_camera_continuity_min: 0.22

  state_same_chain_min: 0.53
  state_same_chain_support: 2
  state_same_chain_continuity_min: 0.28
  state_same_trajectory_min: 0.35
```

These parameters combine appearance and temporal/spatial continuity.

The important principle is:

```text
A tracker reset does not automatically imply a new person.
```

But:

```text
A physically impossible merge must not be forced just because appearance is similar.
```

---

# 11. Cross-camera configuration

```yaml
identity_v6:
  state_cross_fused_min: 0.50
  state_cross_partial_fused_min: 0.47
  state_cross_strong: 0.80

  state_cross_resnet_min: 0.44
  state_cross_swin_min: 0.44
  state_cross_solider_min: 0.42

  state_cross_required_models: 2
  state_cross_required_views: 1
  state_cross_max_gap_sec: 30.0
```

The resolver can require support from more than one model.

The cross-camera threshold is especially sensitive to:

```text
camera domain
crop quality
viewpoint
checkpoint
preprocessing
feature dimension
```

Calibrate after meaningful model or preprocessing changes.

---

# 12. Attribute thresholds

```yaml
identity_v6:
  attribute_color_min: 0.84
  attribute_pattern_min: 0.64
  attribute_detail_min: 0.72
```

These are supporting evidence.

They should not be interpreted as:

```text
color match = same person
pattern match = same person
```

Instead:

```text
credible ReID match
    +
supporting attribute agreement
    →
stronger identity evidence
```

---

# 13. Track fragment configuration

```yaml
tracking:
  fragment_gap_sec: 2.0
```

This controls the track-fragment handling layer.

The exact acceptable range depends on camera frame rate, detector behavior, and scene dynamics.

---

# 14. Severe-overlap guard

```yaml
overlap_guard:
  enabled: true
  iou_min: 0.80
  intersection_min: 0.85
  recovery_samples: 4
```

The guard intentionally targets severe physical overlap rather than every close interaction.

Interpretation:

```text
iou_min
    → minimum pairwise overlap used to classify severe overlap

intersection_min
    → minimum intersection relative to the smaller/contained region

recovery_samples
    → amount of clean post-overlap evidence requested
```

The correct behavior is:

```text
severe overlap
    ↓
protect identity state
    ↓
wait for clean evidence
    ↓
re-identify
```

---

# 15. Post-overlap recovery

```yaml
recovery_guard:
  enabled: true
  required_models: 2

  fused_min: 0.56

  resnet_min: 0.52
  swin_min: 0.52
  solider_min: 0.50

  relaxed_fused_min: 0.50
  relaxed_resnet_min: 0.48
  relaxed_swin_min: 0.48
  relaxed_solider_min: 0.46

  fail_limit: 4
```

And:

```yaml
post_overlap_interval_frames: 1
trajectory_history_frames: 30
```

Recovery is denser than normal sampling because the system is trying to answer a specific question:

> Is the person after the overlap the same person who entered the overlap?

---

# 16. Input and output paths

```yaml
input:
  videos: []
  output_dir: rebuild_outputs_state_invariant
```

Normally, prefer command-line video arguments for reproducibility:

```bash
python rebuild/run.py batch_state_final \
  --config rebuild/config_state_invariant.yaml \
  --videos <camera_1> <camera_2>
```

This avoids editing source-controlled configuration for every test dataset.

---

# 17. Root `config.yaml`

The root stack has a different configuration family.

Important sections include:

```text
source
detector
tracker
reid
store
identity
live
geometry
```

It is consumed by:

```text
main.py
```

and is not the same configuration contract as:

```text
rebuild/config_state_invariant.yaml
```

### Root ReID representation

The current root configuration is built around the FastReID/torchreid abstraction and can select different backends.

The repository's comments document FastReID SBS ResNet101-IBN as the currently configured backbone.

This is a different feature space from the state-final NVIDIA ONNX models.

Therefore:

```text
Do not reuse thresholds without re-calibration.
```

---

# 18. Qdrant deployment details

Root configuration contains:

```yaml
store:
  enabled: true
  path: qdrant_data
  url: http://localhost:6333
```

A Qdrant server can be started using the repository's Compose file:

```bash
docker compose up -d
```

Check:

```text
http://localhost:6333/dashboard
```

The root stack can take a Qdrant URL/API key from environment variables.

Never commit API credentials.

---

# 19. Live source configuration

For live deployment, the state-final rebuild runner accepts:

```text
camera_name=source
```

Example:

```bash
python rebuild/run.py live \
  --config rebuild/config_state_invariant.yaml \
  --sources \
    entrance=<rtsp-url> \
    lobby=<rtsp-url> \
    corridor=<rtsp-url>
```

Keep camera names stable.

The camera name becomes part of:

```text
logs
tracklet identity
trajectory history
cross-camera reasoning
diagnostics
```

Avoid credentials on the command line where possible because command-line arguments may be visible to other processes.

---

# 20. Generalized environment variables

A deployment may use environment variables such as:

```text
QDRANT_URL=<qdrant-url>
QDRANT_API_KEY=<secret>
```

The root modular stack already supports environment-based Qdrant configuration.

For other secrets, follow the same principle:

```text
secret
    → environment / secret manager
    → runtime
```

not:

```text
secret
    → source control
```

---

# 21. Configuration invariants

The following values are coupled.

### ReID checkpoint ↔ preprocessing

Changing the model without preserving the correct preprocessing can produce valid-looking but degraded embeddings.

### ReID feature space ↔ thresholds

Changing the backbone or feature tap changes score distributions.

### Embedding dimension ↔ vector store

Changing vector dimension requires a compatible store collection/state representation.

### Identity state ↔ model identity

Do not mix incompatible historical embeddings into the current state.

### Camera names ↔ topology/geometry

Use stable logical camera identifiers.

---

# 22. Configuration change procedure

For any material ReID change:

```text
1. Record current config
2. Record current checkpoint
3. Capture / select a validation run
4. Change one parameter or component
5. Run the calibration / evaluation tools
6. Inspect identity errors
7. Compare false merges and false splits
8. Validate output video
9. Only then promote the configuration
```

Do not judge a new ReID configuration from the number of identities alone.

