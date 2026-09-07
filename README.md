# Multi-Camera Person Re-Identification

A production-oriented multi-camera Person Re-Identification (ReID / MTMC) pipeline for detecting people, maintaining camera-local tracks, extracting appearance features, resolving identity across time and cameras, handling difficult overlap cases, and producing persistent Global IDs.

This documentation is written against the **`main` branch** of:

`Rushabh-Savla/Inference_Person_REID`

The repository contains the current state-invariant rebuild pipeline together with earlier V2–V6 implementations and a more modular root `main.py` stack. The documentation below focuses on the maintained state-final architecture while documenting the repository's broader structure where it affects operation and maintenance.

> **Important:** A `track_id` is not a Global ID. Track IDs are camera-local and temporary. ReID is the process that decides whether two observations belong to the same real-world person.

---

## 1. What the system solves

The system addresses the full multi-camera identity problem:

```text
Video / RTSP
    ↓
Person Detection
    ↓
Camera-local Tracking
    ↓
Tracklets
    ↓
Quality-aware Person Crops
    ↓
Multi-model ReID Embeddings
    ↓
Appearance / State Memory
    ↓
Same-camera Fragment Repair
    ↓
Cross-camera Reconciliation
    ↓
Persistent Identity State
    ↓
Global IDs
    ↓
Annotated Output
```

The important conceptual separation is:

```text
Detection
    ≠
Tracking
    ≠
Tracklet
    ≠
ReID Embedding
    ≠
Identity Match
    ≠
Global ID
```

Each layer solves a different problem and should be debugged independently.

---

# 2. Current maintained architecture

The current repository README describes the maintained production-style path as the **state-invariant V6 / Safe055** pipeline.

Its public runner is:

```bash
python rebuild/run.py batch_state_final \
  --config rebuild/config_state_invariant.yaml \
  --videos <video_1> <video_2> ...
```

For live sources:

```bash
python rebuild/run.py live \
  --config rebuild/config_state_invariant.yaml \
  --sources \
    <camera_name>=<source_1> \
    <camera_name>=<source_2> \
    ...
```

`rebuild/run.py` dispatches `batch_state_final` to:

```text
BatchPipelineStateInvariantOverlapReid
```

and `live` to the state-aware live runner.

Older V2–V6 runners remain in the repository for comparison, compatibility, and experimentation. They are not the same pipeline and should not be mixed with the state-final configuration.

---

# 3. System architecture

```text
                    ┌────────────────────────┐
                    │ Video Files / RTSP     │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ YOLO Person Detection   │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ ByteTrack              │
                    │ Camera-local Tracking   │
                    └───────────┬────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
              Tracklets              Position / Trajectory
                    │                       History
                    ▼
             Quality Filtering
                    │
                    ▼
         ┌──────────────────────────┐
         │ Body Representation      │
         │                          │
         │ Full / Light / Upper     │
         │ Torso / Lower            │
         └────────────┬─────────────┘
                      │
          ┌───────────┼──────────────┐
          │           │              │
          ▼           ▼              ▼
     NVIDIA         NVIDIA        SOLIDER
     ResNet-50      Swin Base     Swin Base
          │           │              │
          └───────────┼──────────────┘
                      ▼
             Appearance State Banks
                      │
                      ▼
            Local Identity Proposal
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
 Same-camera Repair          Severe Overlap Guard
        │                           │
        │                      Pause / Protect
        │                           │
        └─────────────┬─────────────┘
                      ▼
             Post-overlap Recovery
                      │
                      ▼
             Cross-camera Resolver
                      │
            ┌─────────┼──────────┐
            ▼         ▼          ▼
           ReID   Attributes   Geometry /
                                Trajectory
                      │
                      ▼
             Persistent Identity
                    State
                      │
                      ▼
                  Global ID
                      │
                      ▼
               Annotated MP4
```

---

# 4. Repository layout

The main repository contains several layers:

```text
.
├── rebuild/
│   ├── run.py
│   ├── config_state_invariant.yaml
│   ├── batch_state_invariant_*.py
│   ├── multimodel_state_invariant_*.py
│   ├── identity_body_v6.py
│   ├── live_*.py
│   ├── test_*.py
│   └── diagnostic / evaluation utilities
│
├── src/
│   ├── detector.py
│   ├── video_source.py
│   ├── drawing.py
│   ├── database/
│   ├── geometry/
│   ├── identity/
│   ├── live/
│   └── reid/
│
├── calibration/
│   ├── frames
│   └── association labels / tracklet pairs
│
├── sheets/
│   └── diagnostic contact sheets
│
├── config.yaml
├── requirements.txt
├── docker-compose.yml
├── deploy.sh
└── README.md
```

### Main responsibilities

| Area | Purpose |
|---|---|
| `rebuild/` | Current state-invariant V6 batch/live implementations |
| `src/detector.py` | Detection, tracking, crop primitives, optional pose ensemble |
| `src/reid/` | ReID backends, models, batching, feature extraction |
| `src/identity/` | Identity decision, verification, reranking, reconciliation, decision logging |
| `src/geometry/` | Calibration, floor coordinates, trajectory/reachability reasoning |
| `src/live/` | Live capture, batching, inference, identity, rendering, writing |
| `src/database/` | Persistent vector/identity storage for the root modular stack |
| `calibration/` | Reference material used to calibrate cross-camera relationships |
| `sheets/` | Visual diagnostics for tracklets and difficult cases |
| `rebuild/test_*.py` | Regression and behavior tests |

---

# 5. ReID models

The maintained state-final pipeline uses three appearance models:

```text
NVIDIA ResNet-50
NVIDIA Swin Base
SOLIDER Swin Base
```

The expected state-final files are:

```text
weights/yolo11m.pt

weights/reid/resnet50_market1501_aicity156.onnx

weights/reid/nvidia_swin_base_1024/export_55/
    swin_base_market1501_aicity156_featuredim1024.onnx

weights/solider_swin_base_msmt17.onnx
```

The repository's current README provides a project weight bundle here:

```text
https://drive.google.com/drive/folders/1jCESnjj5g2WsJRLTqaMuMi9vg2OghYdP?usp=sharing
```

Download the files from the project source and place them under the expected paths before running the state-final pipeline.

### Why three models?

A single embedding model can fail on:

- viewpoint changes;
- lighting changes;
- partial visibility;
- pose changes;
- clothing similarity;
- camera-specific appearance shifts.

The state-final resolver therefore requires model agreement rather than trusting one cosine score.

Configured cross-camera minimum model support:

```text
state_cross_required_models = 2
```

This is a confidence gate, not a guarantee of identity correctness.

---

# 6. Detection and tracking

The state-final configuration uses:

```yaml
detector:
  model: weights/yolo11m.pt
  conf: 0.55
  iou: 0.60
  tracker: bytetrack.yaml
```

The conceptual flow is:

```text
YOLO11m
    ↓
Person detections
    ↓
ByteTrack
    ↓
camera-local track IDs
```

Detection answers:

> Is there a person here?

Tracking answers:

> Is this the same track as in the previous frames?

Neither answers:

> Is this person the same person seen in another camera?

That is the job of ReID and identity reconciliation.

---

# 7. Multi-view body representation

The state-final rebuild path does not depend on a single crop representation.

It uses:

```text
full
light
upper
torso
lower
```

The goal is to preserve identity evidence when one region becomes unreliable.

For example:

```text
standing person
    → full + upper + torso + lower

person partly hidden
    → upper + torso

seated person
    → upper + torso

lower body occluded
    → full + upper + torso

illumination variation
    → light variant + original
```

The system is still fundamentally an appearance-based ReID system. These views are complementary evidence; they are not independent identities.

---

# 8. Identity resolution

The state-final identity layer is deliberately evidence-based.

Important configured values include:

```text
match_threshold       = 0.60
match_margin          = 0.035
strong_threshold      = 0.72

support_required      = 2

accumulated_body      = 0.56
accumulated_support   = 3

partial_threshold     = 0.58
partial_support       = 2

gallery               = 64
candidate_gallery     = 24

promote_quality      = 0.68
novelty               = 0.985
```

These values should be treated as **deployment-specific calibration parameters**, not universal ReID thresholds.

A new model, preprocessing recipe, camera domain, or crop policy can change the score distribution.

---

# 9. Same-camera fragment repair

ByteTrack can lose a track and create a new one.

Example:

```text
Track 12
   ↓
lost
   ↓
Track 47
```

That does not automatically mean a new person.

The state-invariant resolver can combine:

```text
appearance
+
camera continuity
+
time gap
+
trajectory
+
position
```

Key settings include:

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

The resolver must remain conservative about impossible associations.

---

# 10. Severe-overlap protection

Hard physical overlap is handled separately because embedding a crop containing multiple people can corrupt the identity representation.

State-final configuration:

```yaml
overlap_guard:
  enabled: true
  iou_min: 0.80
  intersection_min: 0.85
  recovery_samples: 4
```

During severe overlap, the pipeline keeps tracking state and records the event but protects clean appearance evidence.

Conceptually:

```text
Detection      → continue
Tracking       → continue
Position       → continue
Trajectory     → continue
Overlap log    → continue
New identity   → protect / pause
```

The pre-overlap appearance state acts as an anchor.

When overlap clears:

```text
Overlap clears
      ↓
New segment
      ↓
Dense recovery
      ↓
ResNet + Swin + SOLIDER
      ↓
Compare with clean pre-overlap state
      ↓
Accept continuity / reject continuity
```

Recovery settings include:

```text
required_models = 2
fused_min       = 0.56

resnet_min      = 0.52
swin_min        = 0.52
solider_min     = 0.50
```

The design objective is simple:

> Overlap itself must not be allowed to manufacture an identity change.

---

# 11. Attributes and supporting evidence

The rebuild architecture includes supporting identity evidence for:

```text
Upper-body colour
Lower-body colour
Upper-body pattern
Lower-body pattern
Head/detail descriptors
Eye-region descriptors
Visibility
```

Configured thresholds:

```text
attribute_color_min   = 0.84
attribute_pattern_min = 0.64
attribute_detail_min  = 0.72
```

These are support signals. They should not be interpreted as deterministic identity keys.

---

# 12. Trajectory and geometry

Track observations can preserve:

```text
frame
timestamp
bounding box
center
body height
```

with:

```text
trajectory_history_frames = 30
```

Geometry is used to establish spatial plausibility.

Examples:

```text
Person A in Camera A at t1
Person A in Camera B at t2
```

The system can reason about whether the elapsed time and recorded floor positions are physically compatible.

Geometry should be used as a **veto / plausibility signal**, not as a replacement for visual identity evidence.

---

# 13. Persistent identity store

The project uses **Qdrant** as the persistent vector store for identity embeddings and gallery retrieval. It provides durable storage and similarity search over the feature vectors used by the identity pipeline.

Typical responsibilities include:

```text
store identity embeddings
retrieve nearest identity candidates
maintain a searchable identity gallery
persist identity data across runs
```

Identity matching remains governed by the ReID models and resolver thresholds; Qdrant is the storage and retrieval layer, not the ReID model itself.

Treat stored vectors as coupled to their embedding representation. Do not silently reuse an existing collection after changing:

```text
ReID model
checkpoint
preprocessing
embedding dimension
embedding tap
identity configuration
```

A model-space change may require a new collection, migration, or state rebuild.

---

# 14. Installation

## 14.1 Clone

```bash
git clone <repository-url>
cd Inference_Person_REID
git checkout main
```

The canonical codebase is the `main` branch. Do not assume historical V6 branches or local worktrees are identical to `main`.

---

## 14.2 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository currently pins:

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

FastReID is vendored under:

```text
src/reid/vendor/fastreid/
```

Do not add the CPU-only `onnxruntime` package alongside the CUDA package for the NVIDIA ONNX inference path.

---

# 15. GPU validation

Run:

```bash
nvidia-smi
```

Then:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Torch:', torch.__version__)"
```

For the ONNX ReID models:

```bash
python -c "import onnxruntime as ort; print('ORT:', ort.__version__); print('Providers:', ort.get_available_providers())"
```

Expected provider:

```text
CUDAExecutionProvider
```

A successful Python import without a CUDA provider is not sufficient for the intended GPU runtime.

---

# 16. Verify project imports

```bash
PYTHONPATH="$PWD/src:$PWD" \
python -c "import rebuild.run; print('rebuild.run: OK')"
```

Verify weights:

```bash
ls -lh \
  weights/yolo11m.pt \
  weights/reid/resnet50_market1501_aicity156.onnx \
  weights/reid/nvidia_swin_base_1024/export_55/swin_base_market1501_aicity156_featuredim1024.onnx \
  weights/solider_swin_base_msmt17.onnx
```

---

# 17. Batch inference

Recommended state-final command:

```bash
python rebuild/run.py batch_state_final \
  --config rebuild/config_state_invariant.yaml \
  --videos \
    <camera_1_video> \
    <camera_2_video> \
    <camera_3_video>
```

Process all related cameras in the same run when cross-camera identity reconciliation is required.

---

# 18. Live inference

```bash
python rebuild/run.py live \
  --config rebuild/config_state_invariant.yaml \
  --sources \
    cam_1=<rtsp-or-stream-source-1> \
    cam_2=<rtsp-or-stream-source-2> \
    cam_3=<rtsp-or-stream-source-3>
```

Use camera names that are stable across configuration, topology, geometry, and logs.

Never commit credentials into:

```text
config files
shell scripts
README files
Git history
```

Use environment variables or another secret-management mechanism appropriate for the deployment.

---

# 19. Outputs

The state-final configuration defaults to:

```text
rebuild_outputs_state_invariant/
```

Typical output artifacts include:

```text
annotated videos
debug / diagnostic JSON
track / recovery metadata
```

The exact output set depends on the runner and configuration.

---

# 20. Root modular stack

The repository also includes a root-level architecture entered through:

```bash
python main.py ...
```

That stack contains:

```text
src/live/
src/identity/
src/database/
src/reid/
src/geometry/
```

and supports a Qdrant-backed identity workflow.

It is useful for the modular/live architecture and historical experiments, but it is **not identical** to the state-final `rebuild/` path.

The root configuration currently uses a different ReID representation and a the project uses Qdrant as the vector persistence layer. Keep model-specific thresholds and embedding spaces consistent.

---

# 21. Production deployment notes

The repository includes:

```text
docker-compose.yml
deploy.sh
```

`docker-compose.yml` provides the Qdrant service used by the project for persistent vector storage.

`deploy.sh` provides an rsync-based code deployment pattern.

Treat these as deployment helpers rather than requirements for every installation.

The actual production environment should define:

```text
GPU / driver
CUDA runtime
Python environment
model artifacts
persistent state location
stream source configuration
logging location
storage policy
```

outside the source repository when appropriate.

---

# 22. Validation checklist

Before trusting identity results, verify the full chain:

```text
[ ] Model files exist
[ ] CUDA is available
[ ] ONNX CUDA provider is available
[ ] YOLO detects expected people
[ ] ByteTrack produces stable camera-local tracks
[ ] ReID crops are valid
[ ] ReID embeddings are produced
[ ] Multiple models are loaded in state-final mode
[ ] Appearance state accumulates
[ ] Same-camera fragments are repairable
[ ] Severe overlap is protected
[ ] Post-overlap recovery works
[ ] Cross-camera matching works
[ ] Persistent state is writable
[ ] Final GIDs are consistent
[ ] Annotated output agrees with logs
```

---

# 23. Engineering rule

Use this order when making changes:

```text
REPRODUCE
   ↓
MEASURE
   ↓
ISOLATE THE FAILURE LAYER
   ↓
CHANGE ONE VARIABLE
   ↓
VALIDATE
   ↓
CALIBRATE
```

Do not change detector, tracker, ReID model, thresholds, and reconciliation logic at the same time. That destroys causal information.

---

# 24. Documentation set

For maintainers:

```text
README.md
CONFIG.md
ARCHITECTURE.md
TROUBLESHOOTING.md
```

Use:

- `README.md` for installation and operational overview.
- `CONFIG.md` for configuration and deployment.
- `ARCHITECTURE.md` for processing and identity logic.
- `TROUBLESHOOTING.md` for diagnosing failures.
