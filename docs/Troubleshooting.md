# Person ReID Troubleshooting Guide

This guide is written for the **main branch** and is intentionally generalized. It does not assume a particular workstation, camera IP, username, directory, or deployment host.

Use the guide in the order shown. Jumping directly to ReID thresholds when an earlier stage is broken wastes time and often makes the system worse.

---

# 1. First: identify the runtime

The repository contains multiple runtime generations.

## Recommended state-final rebuild

```bash
python rebuild/run.py batch_state_final \
  --config rebuild/config_state_invariant.yaml \
  --videos <camera_1> <camera_2>
```

or:

```bash
python rebuild/run.py live \
  --config rebuild/config_state_invariant.yaml \
  --sources \
    cam_1=<source_1> \
    cam_2=<source_2>
```

## Root modular stack

```bash
python main.py ...
```

These paths are not identical.

Before debugging anything, confirm:

```text
which runner?
which config?
which ReID model?
which state store?
which output directory?
```

If these are unclear, stop debugging results until they are known.

---

# 2. Problem: repository imports fail

Run:

```bash
PYTHONPATH="$PWD/src:$PWD" \
python -c "import rebuild.run; print('rebuild.run: OK')"
```

If this fails, inspect:

```bash
python --version
python -m pip show torch
python -m pip show ultralytics
python -m pip show onnxruntime-gpu
```

Also confirm the virtual environment is active:

```bash
which python
which pip
```

A common error is installing packages into one Python environment and running the project with another.

---

# 3. Problem: CUDA is unavailable

Run:

```bash
nvidia-smi
```

Then:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Then:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected:

```text
CUDAExecutionProvider
```

If PyTorch sees CUDA but ONNX Runtime does not, the problem is not the ReID model itself.

Likely causes:

```text
wrong ONNX Runtime package
CUDA library mismatch
environment contamination
driver/runtime incompatibility
```

The repository intentionally uses:

```text
onnxruntime-gpu
```

and not plain:

```text
onnxruntime
```

Do not “fix” this by blindly installing the CPU package.

---

# 4. Problem: model file not found

Check:

```bash
ls -lh \
  weights/yolo11m.pt \
  weights/reid/resnet50_market1501_aicity156.onnx \
  weights/reid/nvidia_swin_base_1024/export_55/swin_base_market1501_aicity156_featuredim1024.onnx \
  weights/solider_swin_base_msmt17.onnx
```

The repository's project weight bundle is documented at:

```text
https://drive.google.com/drive/folders/1jCESnjj5g2WsJRLTqaMuMi9vg2OghYdP?usp=sharing
```

Make sure the directory structure exactly matches the YAML.

Do not solve a path problem by randomly editing multiple configurations. First determine which runner you are executing.

---

# 5. Problem: detector finds too few people

This is a detector problem until proven otherwise.

Check:

```text
detector model
confidence threshold
input resolution
camera viewpoint
person size
occlusion
lighting
```

A detector cannot recover a person it never produced as a detection.

The state-final detector is configured around:

```yaml
model: weights/yolo11m.pt
conf: 0.55
iou: 0.60
```

Do not lower ReID thresholds to compensate for missing detections.

Recommended diagnostic:

```text
1. Save raw detector boxes
2. Count detections by camera
3. Inspect missed-person frames
4. Only then evaluate tracker / ReID
```

---

# 6. Problem: duplicate person boxes

Duplicate detections can come from:

```text
detector NMS behavior
crowded overlap
tracker association
optional pose ensemble
```

Inspect whether the duplicates already exist before ReID.

If duplicates are visible directly after YOLO:

```text
ReID is not the first suspect.
```

The root detector also contains an optional pose ensemble designed for cases where one tracker box may contain multiple people.

Be careful when enabling secondary detection/pose models in live mode because they add inference cost.

---

# 7. Problem: tracker IDs change while the person stays visible

This is a tracking problem.

Do not immediately modify:

```text
state_cross_fused_min
state_same_fused_min
```

First inspect:

```text
bbox continuity
frame drops
occlusions
scene density
ByteTrack configuration
```

For a healthy track:

```text
Person appears
    ↓
Track ID starts
    ↓
ID remains stable
    ↓
ID ends when the person leaves
```

If instead:

```text
1 → 7 → 13 → 25
```

while the person remains visible, ReID has less reliable evidence because each fragment becomes shorter.

---

# 8. Problem: one track contains two different people

This is a serious tracker failure.

Example:

```text
track 12
    ↓
person A
    ↓
person B
```

Now the appearance bank contains mixed identity evidence.

Symptoms:

```text
highly unstable cross-camera matching
same person fails to match elsewhere
different people receive similar GIDs
```

The correct debugging action is:

```text
inspect tracklet contact sheets
```

rather than immediately lowering identity thresholds.

Fix the association layer first.

---

# 9. Problem: ReID embeddings are not produced

Run the ONNX provider check:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Then check:

```text
crop exists
crop is non-empty
crop quality passes
model weights load
batch size is valid
```

Remember:

```text
YOLO confidence
```

is not:

```text
crop quality
```

A high-confidence detector box can still create a bad ReID crop.

---

# 10. Problem: embeddings exist but similarity scores are strange

Do not compare scores from different feature spaces.

For example:

```text
Model A, post-processing X
```

and:

```text
Model B, post-processing Y
```

can have very different cosine distributions.

The root repository explicitly documents this issue for its OSNet/FastReID experimentation.

A model change can invalidate:

```text
same-camera thresholds
cross-camera thresholds
gallery thresholds
margin thresholds
verification thresholds
```

Recalibrate.

Do not assume:

```text
old threshold + new model = same meaning
```

---

# 11. Problem: same person gets different IDs in the same camera

Investigate in this order:

```text
1. Did ByteTrack fragment?
2. Is the fragment actually one person?
3. Is the time gap reasonable?
4. Are enough good embeddings present?
5. Is trajectory continuity valid?
6. Does same-camera appearance clear the configured bar?
7. Is a competing identity scoring higher?
```

State-final same-camera settings include:

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

Do not raise or lower all of them together.

---

# 12. Problem: same person matches in one camera but not another

This is a cross-camera issue.

Check:

```text
viewpoint difference
camera exposure
person crop quality
body visibility
time gap
model support
cross-camera threshold
```

The state-final pipeline expects support from:

```text
at least 2 of 3 models
```

Cross-camera settings include:

```text
state_cross_fused_min           = 0.50
state_cross_partial_fused_min   = 0.47
state_cross_strong              = 0.80

state_cross_resnet_min          = 0.44
state_cross_swin_min            = 0.44
state_cross_solider_min         = 0.42

state_cross_required_models     = 2
state_cross_required_views      = 1
state_cross_max_gap_sec         = 30.0
```

A failed match is not automatically evidence that the threshold is too high. The tracklet may simply not contain enough reliable appearance information.

---

# 13. Problem: different people get the same GID

This is a false merge.

Treat it as a high-priority identity error.

Inspect:

```text
which camera pairs?
which tracklets?
which model scores?
which body views?
which time gap?
which attributes?
which geometry?
```

Common causes:

```text
similar clothing
mixed tracklet
bad crop
insufficient model diversity
threshold too low
identity state contamination
```

Do not “fix” a false merge by making thresholds globally extreme.

A threshold that prevents one merge may also create large numbers of false splits.

---

# 14. Problem: IDs change after two people overlap

This is exactly the case the state-final overlap guard is designed for.

Inspect:

```text
overlap start
overlap interval
pre-overlap anchor
overlap clearance
post-overlap samples
model agreement
recovery result
```

State-final settings:

```yaml
overlap_guard:
  enabled: true
  iou_min: 0.80
  intersection_min: 0.85
  recovery_samples: 4
```

Recovery:

```text
required_models = 2
fused_min       = 0.56
resnet_min      = 0.52
swin_min        = 0.52
solider_min     = 0.50
```

The expected behavior is not:

```text
overlap → new GID
```

It is:

```text
overlap
    ↓
protect identity
    ↓
clear overlap
    ↓
recover using clean evidence
    ↓
decide identity
```

---

# 15. Problem: post-overlap recovery keeps failing

Check:

```text
post-overlap crop quality
person visibility
track continuity
number of clean samples
model provider
anchor existence
```

If the first frames after overlap are still partially blocked, the recovery sample may be weak.

The configuration deliberately uses dense sampling:

```yaml
post_overlap_interval_frames: 1
```

This means the recovery phase is expensive by design.

Do not disable it simply because it increases inference cost before measuring whether the recovered identity quality is worth that cost.

---

# 16. Problem: identity becomes worse after changing the ReID model

This is expected enough to be a first-class warning.

Changing:

```text
model
checkpoint
preprocessing
feature tap
embedding dimension
```

changes the feature space.

Therefore:

```text
old scores are not directly comparable
old thresholds are not automatically valid
old identity state may be incompatible
```

Recommended procedure:

```text
1. Start a fresh validation run
2. Measure same/different score distributions
3. Re-run threshold sweeps
4. Compare false merges
5. Compare false splits
6. Re-render with the selected thresholds
7. Only then replace production state
```

---

# 17. Problem: output video looks correct but IDs are wrong

The renderer may be correct while the identity resolver is wrong.

Inspect:

```text
debug JSON
decision logs
tracklet IDs
global identity assignments
```

For the root modular path, identity state and reconciliation are explicitly separate from rendering.

A correct box overlay does not prove a correct identity mapping.

---

# 18. Problem: IDs in logs are correct but output video is wrong

This is a rendering/re-rendering issue.

Check:

```text
camera name
track_id → global_id mapping
run_id
source video
frame indexing
output video order
FPS
```

The root modular architecture performs a final render after reconciliation so the output can use the final Global IDs.

Do not judge reconciliation from an intermediate live overlay when an offline-final render is expected.

---

# 19. Problem: live pipeline becomes very slow

Measure before changing models.

Inspect:

```text
GPU utilization
CPU utilization
queue depth
frame-drop rate
ReID batch size
number of active cameras
decode load
pose ensemble
```

The live modular architecture deliberately uses staged queues and batch scheduling.

If the system is overloaded:

```text
capture drops frames
    ↓
tracking sees fewer observations
    ↓
track fragmentation can increase
    ↓
ReID evidence becomes weaker
```

A “faster ReID loop” that causes more tracking fragmentation is not actually an improvement.

---

# 20. Problem: GPU memory spikes

Likely causes:

```text
large ReID batches
too many concurrent cameras
secondary models
large detector input
multiple simultaneous model copies
```

Lower batch size first:

```yaml
swin_batch: 16
solider_batch: 16
```

or the applicable `max_batch` setting.

For a memory issue, change one parameter and re-measure.

Do not randomly disable model components without recording what changed.

---

# 21. Problem: RTSP source hangs or does not stop cleanly

The root configuration contains explicit RTSP timeout support.

Look for:

```yaml
source:
  rtsp:
    transport: tcp
    open_timeout_ms: 5000
    read_timeout_ms: 5000
```

The purpose is to prevent a blocked network read from holding shutdown indefinitely.

Check:

```text
source reachability
RTSP credentials
transport
network packet loss
FFmpeg availability
```

Do not treat a network timeout as a ReID failure.

---

# 22. Problem: live source opens but frames are missing

Check:

```bash
ffprobe <stream>
```

and test the stream independently of the ReID pipeline.

Then inspect:

```text
decode backend
frame rate
resolution
codec
network stability
```

If the frame stream is already broken before YOLO:

```text
changing ReID logic cannot help
```

---

# 23. Problem: Qdrant connection fails

This applies to the root `main.py` modular stack.

Start Qdrant:

```bash
docker compose up -d
```

Check:

```text
http://localhost:6333/dashboard
```

Then verify:

```text
QDRANT_URL
QDRANT_API_KEY
collection availability
vector dimension
```

The root stack can degrade to a non-reconciled/live-only behavior when the store is unavailable, depending on runtime configuration. That should be treated as an explicit degraded mode, not as a successful cross-camera result.

---

# 24. Problem: vector dimension mismatch

This usually means the model and persistent vector state do not belong together.

Check:

```text
current embedding dimension
existing collection dimension
current model/checkpoint
store configuration
```

Do not force incompatible dimensions together.

A dimension mismatch is often a useful warning that the identity state needs to be rebuilt.

---

# 25. Problem: old identities keep appearing after a model change

Likely cause:

```text
old persistent state is still being loaded
```

For a new embedding representation:

```text
new model
    ↓
new feature space
    ↓
new identity state
```

Use a new state path or an explicit migration.

Do not delete production state casually. Back it up first.

---

# 26. Problem: one camera is much worse than the others

Do not immediately use a global threshold.

Cameras differ in:

```text
distance
mounting height
field of view
lighting
compression
viewpoint
person size
occlusion
```

Inspect per-camera:

```text
detection recall
median crop size
track purity
quality rejection
embedding scores
same-camera score distribution
cross-camera score distribution
```

The root configuration already supports per-camera detector overrides.

Use camera-specific settings only when measurement proves they are needed.

---

# 27. Problem: seated / partially visible people lose identity

This is usually an appearance representation problem.

Check:

```text
upper/torso views
crop quality
post-overlap recovery
visibility
number of valid observations
```

Do not assume full-body embeddings should remain equally reliable for:

```text
standing
sitting
partially hidden
```

The multi-view design exists specifically to reduce this dependence.

---

# 28. Problem: identities fail after camera transition

Check all of:

```text
departure camera tracklet quality
arrival camera tracklet quality
time gap
cross-camera model support
camera naming
camera topology
geometry availability
```

A transition failure can be caused by:

```text
weak arrival track
```

rather than:

```text
weak global identity threshold
```

The arrival camera must actually produce enough valid observations to give the resolver something to match.

---

# 29. Problem: geometry blocks correct matches

Geometry is a veto-style signal, not a replacement for appearance.

Check:

```text
calibration version
camera mapping
floor coordinate quality
recorded timestamps
camera clock offsets
camera topology
```

An invalid calibration can make a physically plausible transition look impossible.

Do not simply disable geometry permanently.

First verify the calibration.

---

# 30. Problem: reconciliation produces unexpected merges

Inspect:

```text
pair score
same-camera exclusion
reciprocal-best setting
co-visibility
geometry reachability
cluster transitivity
minimum tracklet observations
```

The root `src/identity/reconcile.py` architecture is deliberately conservative because merging identities is destructive.

A merge should be explainable from the logged evidence.

---

# 31. Problem: reconciliation produces too many splits

Possible causes:

```text
threshold too high
weak prototype
too few tracklet observations
bad crop quality
different viewpoints
missing cross-camera model support
incorrect camera timing
```

Do not simply lower the global threshold.

Measure:

```text
true-match score distribution
false-match score distribution
```

Then calibrate.

---

# 32. Problem: state seems corrupted

Back up first:

```bash
cp <identity-state> <identity-state>.backup
```

Then determine:

```text
what model generated the state?
what preprocessing generated it?
what resolver version generated it?
what configuration generated it?
```

Only then decide whether rebuilding is necessary.

---

# 33. Problem: tests pass but real footage fails

Passing a smoke test proves only that the code path executes under that test.

It does not prove:

```text
camera robustness
track purity
cross-camera accuracy
overlap recovery
```

Use real representative validation footage.

A proper validation set should contain:

```text
standing
walking
sitting
partial occlusion
full occlusion
camera transitions
similar clothing
different lighting
tracker resets
severe overlap
```

---

# 34. Debugging workflow

Use this exact order:

```text
SOURCE
  ↓
DETECTION
  ↓
TRACKING
  ↓
TRACKLET PURITY
  ↓
CROP QUALITY
  ↓
EMBEDDINGS
  ↓
LOCAL IDENTITY
  ↓
SAME-CAMERA REPAIR
  ↓
OVERLAP RECOVERY
  ↓
CROSS-CAMERA MATCH
  ↓
PERSISTENCE
  ↓
RENDERING
```

At every step, ask:

```text
What evidence proves this stage is working?
```

Do not assume a downstream symptom identifies an upstream cause.

---

# 35. What not to do

Avoid these common mistakes:

```text
Changing multiple thresholds at once
Mixing different model embeddings
Reusing old state after a feature-space change
Calling a tracker ID a global ID
Lowering ReID thresholds to compensate for missed detections
Using identity count as the only accuracy metric
Judging a live provisional overlay as the final reconciled result
Disabling overlap protection without measuring its failure cases
Committing camera credentials to Git
```

---

# 36. Minimal triage checklist

When a GID is wrong, collect:

```text
camera name
frame number / timestamp
track ID
tracklet duration
crop image
ReID model scores
model support count
current identity candidate
runner-up candidate
same-camera evidence
cross-camera evidence
trajectory evidence
geometry evidence
overlap status
recovery status
persistent-state version
```

With that information, the problem usually becomes classifiable as:

```text
detection
tracking
crop
ReID
matching
reconciliation
state
rendering
```

That is the correct starting point for a fix.
