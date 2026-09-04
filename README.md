# Multi-Camera Person Re-Identification

Production-oriented multi-camera **Person Re-Identification (ReID / MTMC)** pipeline with persistent Global IDs (GIDs), same-camera tracker-fragment repair, state-invariant cross-camera matching, multi-model appearance evidence, body attributes, trajectory/geometry support, severe-overlap protection, and final annotated MP4 output.

The maintained execution path on `working_model` is the **Safe055 / State-Invariant V6 family**. The repository also retains older compatibility runners under `rebuild/`, but the current production-style entry point is:

```bash
python rebuild/run.py batch_state_final \
  --config rebuild/config_state_invariant.yaml \
  --videos CAM1.mp4 CAM2.mp4 CAM3.mp4
```

---

## What the system actually does

The active pipeline is not a simple `YOLO -> tracker -> cosine similarity -> ID` system. It combines multiple independent evidence sources and resolves identities after camera-local tracklets have been formed.

```text
Input videos / RTSP streams
        |
        v
YOLO11m person detection
        |
        v
ByteTrack per camera
        |
        v
Tracklets + segment boundaries
        |
        +------------------------------+
        |                              |
        v                              v
Position/trajectory history       Quality-filtered body crop
        |                              |
        |                     +--------+--------+
        |                     |        |        |
        |                  full/light upper/torso/lower
        |                              |
        |                     +--------+--------+
        |                     |        |        |
        |                  NVIDIA ResNet  NVIDIA Swin  SOLIDER
        |                     |        |        |
        +---------------------+--------+--------+
                              |
                              v
                     Multimodel state banks
                              |
                              v
                    V6 local identity proposals
                              |
                              v
                 Same-camera fragment repair
                              |
                              v
               State-invariant MTMC resolver
                              |
              +---------------+----------------+
              |               |                |
              v               v                v
         ReID evidence   Attributes       Temporal / geometry /
         (3 models)      + colours        trajectory evidence
              |               |                |
              +---------------+----------------+
                              |
                              v
                  Camera-consistent components
                              |
                              v
                    Persistent SQLite registry
                              |
                              v
                     Global IDs (G1, G2, ...)
                              |
                              v
                    Annotated final MP4 files
```

### Core design principles

- **Three independent ReID models remain active:** NVIDIA ResNet, NVIDIA Swin Base, and SOLIDER.
- **Deep ReID is the primary identity signal.** Colour, pattern, head/detail attributes, geometry, temporal compatibility, and trajectory are supporting evidence.
- **Attributes cannot create an identity on their own.** They can only reinforce an already-credible multimodel decision.
- **Same-camera and cross-camera reconciliation are separate problems.** Same-camera matching is used for tracker-reset/fragment repair; cross-camera matching builds MTMC identity components.
- **Severe physical overlap is treated specially.** Tracking continues, feature extraction pauses, and clean post-overlap evidence is collected before reassignment is considered.
- **Position is tracked continuously.** Trajectory history is maintained independently from the ReID sampling interval.
- **Persistent global identity memory uses SQLite for the active state-final path.** The active state-final pipeline does not require the old nested `Inference_PersonReid` application or its Qdrant state.

---

## Current maintained configuration

The authoritative configuration is [`rebuild/config_state_invariant.yaml`](rebuild/config_state_invariant.yaml).

### Detection and tracking

```yaml
model: weights/yolo11m.pt
conf: 0.55
iou: 0.60
tracker: bytetrack.yaml
```

YOLO confidence and ReID crop quality are different signals. Detector confidence answers whether the detector believes the object is a person; crop quality determines whether the crop is useful enough to enter the appearance state bank.

Tracklets are segmented when a track disappears for more than the configured fragment gap:

```text
fragment_gap_sec = 2.0
```

A normal ReID observation is collected every:

```text
interval = 2 frames
```

Additional body-part / illumination views are refreshed at:

```text
part_interval = 4 frames
```

The current minimum crop-quality gate is:

```text
min_quality = 0.20
```

---

## Multi-view appearance representation

Accepted body observations are represented using multiple views rather than one fixed crop:

```text
full
light
upper
 torso
lower
```

`light` is the illumination-normalized variant used in addition to the original crop. The other views provide upper-body, torso, and lower-body evidence when the full body is not equally informative.

This matters because the same person can appear standing, sitting, walking, partially occluded, differently illuminated, differently framed, or with only part of the body visible.

The resolver also carries an `attributes` view containing compact appearance descriptors derived from:

```text
upper-body colour
lower-body colour
upper-body pattern
lower-body pattern
head/detail descriptor
eye-region descriptor
visibility indicators
```

These descriptors are supporting evidence only. In the current resolver their influence is deliberately capped; they never replace the three-model ReID decision.

---

## The three ReID models

### 1. NVIDIA ReIdentificationNet ResNet-50

```text
weights/reid/resnet50_market1501_aicity156.onnx
```

The current adapter uses CUDA execution and produces normalized appearance embeddings for the ReID state bank.

### 2. NVIDIA Swin Base

```text
weights/reid/nvidia_swin_base_1024/export_55/swin_base_market1501_aicity156_featuredim1024.onnx
```

This is the independent transformer representation used alongside the NVIDIA ResNet model.

### 3. SOLIDER Swin Base

```text
weights/solider_swin_base_msmt17.onnx
```

SOLIDER is used as a third independent appearance representation. The old `SOLIDER-REID/` training/export source tree has been removed from `working_model`; only the inference ONNX model is required by the maintained runtime.

The active configuration requires model agreement rather than trusting one model in ordinary cross-camera matching:

```text
state_cross_resnet_min   = 0.44
state_cross_swin_min     = 0.44
state_cross_solider_min  = 0.42
state_cross_required_models = 2
```

---

## State-invariant identity resolution

After extraction, the pipeline first creates camera-local V6 identity proposals and then performs global reconciliation.

### Pass 1 — joint detection, tracking, and multimodel evidence

All cameras are processed in a shared session. Each camera maintains its own ByteTrack state while the pipeline records observations, trajectories, tracklet segments, quality-filtered body crops, and multimodel state banks.

### Pass 2 — protected local V6 identity proposals

`GlobalIdentityBodyV6` creates protected camera-local identity proposals from accumulated appearance evidence.

Relevant V6 configuration includes:

```text
match_threshold      = 0.60
match_margin         = 0.035
strong_threshold     = 0.72
support_required     = 2
accumulated_body     = 0.56
accumulated_support  = 3
partial_threshold    = 0.58
partial_support      = 2
gallery              = 64
candidate_gallery    = 24
promote_quality      = 0.68
novelty              = 0.985
```

### Pass 3 — tracker-reset repair + state-invariant MTMC

The resolver uses:

```text
NVIDIA ResNet evidence
NVIDIA Swin evidence
SOLIDER evidence
multi-view support
attribute support
colour support
temporal compatibility
geometry support
same-camera continuity
trajectory consistency
state-transition evidence
persistent gallery evidence
```

The final result is constructed as camera-consistent identity components and then mapped to persistent GIDs.

---

## Same-camera tracker-reset repair

A track can disappear and later reappear with a new local tracklet. The system therefore treats same-camera repair as a first-class identity problem rather than assuming every tracker ID is permanent.

Same-camera matching uses both appearance agreement and continuous trajectory information. The active overlap-aware resolver calculates trajectory compatibility from predicted position and direction; position contributes more strongly than direction.

Current same-camera guards include:

```text
state_same_fused_min              = 0.48
state_same_model_min              = 0.42
state_same_max_gap_sec            = 30.0
state_same_camera_continuity_min  = 0.22
state_same_chain_min              = 0.53
state_same_chain_support          = 2
state_same_chain_continuity_min   = 0.28
state_same_trajectory_min         = 0.35
```

The resolver allows sequential repair chains, such as:

```text
tracklet A -> tracklet B -> tracklet C
```

for one person, while preventing overlapping time intervals from being stitched together.

---

## Severe-overlap protection

Overlap handling is intentionally conservative. Ordinary people walking close to one another should not enter overlap mode.

The active configuration uses:

```yaml
overlap_guard:
  enabled: true
  iou_min: 0.80
  intersection_min: 0.85
  recovery_samples: 4
```

This means overlap handling is intended for **substantial physical containment / near-complete tracker overlap**, not a small crossing.

### During severe overlap

The system:

1. keeps the detector and ByteTrack running;
2. records bounding-box position and trajectory history;
3. records the overlap event and partner track;
4. **does not extract new ReID features while the severe overlap is active**.

The existing clean pre-overlap appearance state is preserved as an anchor.

### After the overlap clears

The system starts a new segment for the affected local track and performs dense multimodel recovery:

```text
post_overlap_interval_frames = 1
recovery_samples              = 4
```

The clean body is checked again with all three independent ReID models. The post-overlap comparison uses the pre-overlap anchor and requires multimodel support before treating the comparison as a match.

Recovery defaults:

```text
required_models = 2
fused_min       = 0.56
ResNet minimum  = 0.52
Swin minimum    = 0.52
SOLIDER minimum = 0.50
```

A weaker relaxed gate is also available for difficult recovery cases, but recovery remains multimodel rather than single-model.

The important rule is:

> **Overlap itself never causes a person to change identity. The identity decision happens after clean post-overlap evidence is available.**

---

## Attributes, clothing, and state changes

The attribute module computes compact descriptors from the body crop for:

```text
upper colour
lower colour
upper pattern
lower pattern
head/detail
eye-region detail
visibility
```

These are deliberately not allowed to dominate ReID.

For example, a shirt colour changing because of lighting or camera rendering should not by itself create a new GID. The resolver uses the deep models first and uses attributes only as bounded reinforcement when the deep evidence already has sufficient model support.

The configured attribute thresholds are:

```text
attribute_color_min   = 0.84
attribute_pattern_min = 0.64
attribute_detail_min  = 0.72
```

The current attribute resolver caps the clothing/detail contribution to a small score reinforcement. Attributes cannot independently create a match.

---

## Geometry and trajectory evidence

Spatial evidence is supportive, not a replacement for appearance evidence.

The system records, per trajectory observation:

```text
frame
timestamp
bounding box
center point
body height
```

Up to the configured history length is retained:

```text
trajectory_history_frames = 30
```

For same-camera repair, trajectory compatibility is based on predicted endpoint position plus movement direction. This helps distinguish plausible tracker-fragment continuity from an unrelated nearby person.

For cross-camera reasoning, temporal and geometry compatibility remain supporting signals. They are not allowed to override strong visual evidence merely because two people are physically close.

---

## Persistent identity memory

The maintained state-final pipeline uses SQLite:

```text
identity_state/reid_state_invariant_v4.sqlite3
```

Configuration:

```text
model_id  = final-state-invariant-v4
bank_size = 96
```

The persistent gallery is used for identity reuse after a global component has been formed:

```text
state_gallery_match_min = 0.62
state_gallery_margin    = 0.02
```

This is the source of persistent GID continuity across separate processing sessions when the same identity state database is intentionally retained.

**Back up this SQLite state before deleting, replacing, or migrating identity data.**

---

## Cross-camera matching

Cross-camera links require agreement from multiple appearance models and supporting context.

Key configuration:

```text
state_cross_fused_min          = 0.50
state_cross_partial_fused_min  = 0.47
state_cross_strong             = 0.80
state_cross_required_models    = 2
state_cross_required_views     = 1
state_cross_max_gap_sec        = 30.0
```

A normal cross-camera candidate is therefore not accepted simply because one cosine similarity is high. The resolver evaluates multimodel support, view support, state transitions, temporal compatibility, geometry, colour/attribute reinforcement, and gallery context.

Strong single-model cases have a separate guarded route, but the normal path still preserves multi-model support requirements.

---

## Model weights

The actual `.pt`, `.pth`, and `.onnx` model binaries are intentionally **gitignored**. They are deployment artifacts and are not stored in GitHub.

Current local files required by the state-final inference path:

```text
weights/yolo11m.pt
weights/reid/resnet50_market1501_aicity156.onnx
weights/reid/nvidia_swin_base_1024/export_55/swin_base_market1501_aicity156_featuredim1024.onnx
weights/solider_swin_base_msmt17.onnx
```

There is also:

```text
weights/swin_base_msmt17.pth
```

which is a checkpoint/training artifact and is not referenced by the maintained state-final inference configuration.

Do not move the active inference weight files without updating every corresponding configuration and helper path.

---

## Installation

The dependency pins are maintained in [`requirements.txt`](requirements.txt).

Important runtime versions include:

```text
ultralytics             8.4.89
torch                   2.12.1
torchvision              0.27.1
torchreid                0.2.5
onnxruntime-gpu         1.28.0
numpy                    2.2.6
opencv-python            5.0.0.93
PyYAML                   6.0.3
lap                      0.5.13
qdrant-client            1.18.0
```

Create and activate an environment, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The maintained ReID adapters require CUDA-capable ONNX Runtime. Do **not** replace `onnxruntime-gpu` with plain `onnxruntime` in the final environment.

For NVIDIA inference hosts, verify the GPU before running:

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Torch:', torch.__version__)"
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

The current development environment has been used with an NVIDIA RTX 6000-class GPU and CUDA-enabled PyTorch / ONNX Runtime.

---

## Run the maintained batch pipeline

For three recorded cameras:

```bash
python rebuild/run.py batch_state_final \
  --config rebuild/config_state_invariant.yaml \
  --videos \
    cam_213.mp4 \
    cam_224.mp4 \
    cam_219.mp4
```

The input names can be arbitrary; the pipeline derives camera names from the video paths and keeps the cameras in one shared processing session.

Default output directory from the maintained configuration:

```text
rebuild_outputs_state_invariant/
```

The run produces the final annotated video outputs plus diagnostic state/metadata generated by the pipeline.

### Decoder fallback

If OpenCV cannot decode an otherwise valid input, the active joint pipeline attempts an FFmpeg H.264 working copy and retries decoding. This is useful for awkward MP4/H.264 source files and keeps the detector/tracker stage from depending on one fragile decoder path.

---

## Run live RTSP

The maintained live entry point records all cameras to MP4 first and then runs the same Safe055/V6 joint reconciliation over those finalized recordings.

Command shape:

```bash
python rebuild/run.py live \
  --config rebuild/config_state_invariant.yaml \
  --sources \
    cam_213=rtsp://USER:PASSWORD@HOST/ch13/0 \
    cam_224=rtsp://USER:PASSWORD@HOST/ch24/0
```

You can add more cameras in the same `CAMERA=SOURCE` format.

The live flow is:

```text
RTSP camera streams
       |
       v
parallel MP4 capture
       |
       v
finalized camera recordings
       |
       v
exact Safe055/V6 joint video reconciliation
       |
       v
annotated MP4 outputs
```

Live sessions are written under:

```text
recordings/live_state_<timestamp>/
```

The live pipeline requires both `ffmpeg` and `ffprobe` to be available in `PATH`.

`--show` is accepted for compatibility but the maintained capture/reconciliation path is designed to be headless-safe; no GUI is required for the core processing flow.

---

## Outputs and diagnostics

A maintained run can produce several useful artifacts:

```text
rebuild_outputs_state_invariant/
    <camera output MP4 files>
    state_invariant_debug.json
    cache / tracklet / detection metadata
```

For each processed tracklet, diagnostic records can include:

```text
camera
frame
timestamp
track_id
segment
tracklet_key
bbox
detection_score
overlap state / partners
overlap boundary
recovery state
post-overlap ReID scores
trajectory presence
```

These records are intended for debugging identity decisions and verifying that the resolver is behaving as designed.

---

## Interpreting confidence correctly

There are multiple different numbers in this project. They must not be treated as interchangeable.

### Detector confidence

```text
detection_score
```

This is the YOLO person-detection confidence.

### Crop quality

```text
min_quality = 0.20
```

This determines whether an observation is good enough to feed into the appearance bank.

### ReID similarity / fused evidence

These are resolver-level identity signals combining one or more ReID models and supporting evidence.

Therefore:

```text
YOLO confidence != crop quality != ReID similarity != final match confidence
```

---

## Repository layout

The maintained repository is organized around the state-final rebuild/runtime:

```text
.
├── rebuild/
│   ├── run.py
│   ├── config_state_invariant.yaml
│   ├── batch_state_invariant_*.py
│   ├── multimodel_state_invariant_*.py
│   ├── identity_body_v6.py
│   └── supporting V2/V3/V4/V5/V6 compatibility runners
│
├── src/
│   ├── detector.py
│   └── reid/
│       ├── nvidia_reid.py
│       ├── nvidia_swin.py
│       └── solider_reid.py
│
├── identity_state/
│   └── reid_state_invariant_v4.sqlite3
│
├── weights/
│   └── local model artifacts (gitignored)
│
├── recordings/
│   └── live MP4 capture sessions
│
└── README.md
```

The following legacy development material has been removed from `working_model` because it is not required by the maintained Safe055/V6 runtime:

```text
SOLIDER-REID/
docs/
experiments/reports/
tests/
tools/
trackers/
```

The nested `Inference_PersonReid/` directory is a legacy Qdrant/application artifact and is **not part of the active state-final dependency path**. It should not be treated as a second implementation of the current ReID runtime.

---

## Important distinction: active state-final vs legacy compatibility code

`rebuild/run.py` still exposes several older runners for compatibility:

```text
batch
batch_final
batch_state_final   <-- maintained path
batch_local_global
batch_v5
batch_v4
batch_v3
batch_v2
live                <-- maintained Safe055/V6 live path
live_v2
```

For current work, debugging, benchmarks, and client demonstrations, use:

```text
batch_state_final
live
```

Do not switch to an older compatibility runner simply because its command name is shorter or its code appears simpler. The maintained state-final pipeline is where the overlap protection, state-invariant matching, multimodel resolver, trajectory support, and persistent SQLite identity layer are wired together.

---

## Verification checklist

Before declaring a deployment ready, verify:

```bash
# Python import / entry point
PYTHONPATH="$PWD/src:$PWD" python -c "import rebuild.run; print('rebuild.run import: OK')"

# CUDA visibility
python -c "import torch; print(torch.cuda.is_available())"

# ONNX providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Required local model files
ls -lh \
  weights/yolo11m.pt \
  weights/reid/resnet50_market1501_aicity156.onnx \
  weights/reid/nvidia_swin_base_1024/export_55/swin_base_market1501_aicity156_featuredim1024.onnx \
  weights/solider_swin_base_msmt17.onnx
```

A healthy maintained environment should show CUDA availability and a CUDA execution provider in ONNX Runtime.

For a real video run, verify that:

```text
all requested cameras open
people are detected
ByteTrack creates track IDs
tracklets accumulate observations
three ReID models produce embeddings
severe overlap pauses feature extraction
post-overlap recovery performs dense checks
same-camera fragments are repaired when evidence supports it
cross-camera links use multimodel evidence
final GIDs are rendered to MP4
```

---

## Troubleshooting

### `Cannot open video/RTSP source`

Confirm the path/URL is valid and readable. For recorded video, try a known-good MP4. The joint pipeline has an FFmpeg H.264 fallback for decode failures, but it cannot repair an invalid or inaccessible source.

### CUDA is unavailable

Check:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

The active NVIDIA ReID ONNX adapters expect CUDA execution. Do not silently switch the final pipeline to a CPU-only ONNX Runtime package.

### A model file is missing

Remember that `.pt`, `.pth`, and `.onnx` files are intentionally ignored by Git. A fresh checkout therefore needs the model artifacts provisioned locally.

### Identities split after tracker reset

Inspect the same-camera repair evidence and the tracklet gap. The resolver requires both appearance agreement and spatial/trajectory continuity; it is intentionally conservative around overlapping intervals.

### Identity changes around a severe overlap

Check the overlap diagnostics. During severe overlap, feature extraction is paused. After exit, four dense recovery samples are collected by default and compared to the clean pre-overlap anchor using the three ReID models.

### Colour appears to dominate a match

That is not the intended design. Attribute evidence is capped and only reinforces already-supported multimodel matches. A colour-only match should not be enough to create a global identity link.

### Persistent identities look wrong after changing models or preprocessing

Do not mix identity-state databases across incompatible model/config versions. The active registry stores a model identifier and persistent gallery state. Rebuild or migrate the identity database deliberately when changing the identity representation.

---

## Reproducibility and engineering rules

The project is designed around a simple rule:

```text
REPRODUCE FIRST
VALIDATE SECOND
OPTIMIZE THIRD
```

Do not replace the maintained resolver with a generic single-model cosine matcher. Do not remove one of the independent ReID models to reduce compute without validating the effect. Do not raise matching thresholds just because a larger number looks safer. Do not let colour or geometry override multimodel appearance evidence.

When changing the identity system, validate the change on the same camera footage and compare:

```text
identity consistency
cross-camera matching
time/trajectory consistency
tracker-fragment repair
false-merge protection
overlap recovery
GPU runtime
final MP4 quality
```

A system that merely runs successfully is not automatically a correct ReID system.

---

## License and model provenance

Project source is maintained in this repository. The detector and ReID model files are external model artifacts and remain outside Git because of size/deployment constraints. Consult the respective upstream model licenses and terms before redistribution.

---

## Current project status

`working_model` is the cleaned working branch for the maintained multi-camera Person ReID implementation.

The active target is:

```text
YOLO11m
  -> ByteTrack
  -> state-aware tracklets
  -> quality-filtered multi-view crops
  -> NVIDIA ResNet + NVIDIA Swin + SOLIDER
  -> protected V6 local identity proposals
  -> same-camera tracker-reset repair
  -> severe-overlap pause + dense recovery
  -> state-invariant cross-camera reconciliation
  -> attributes + colour + trajectory + geometry as supporting evidence
  -> persistent SQLite gallery
  -> final GIDs
  -> annotated MP4 outputs
```

This README is the self-contained operational reference for the current `working_model` branch.