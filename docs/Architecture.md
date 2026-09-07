# Person ReID Architecture and Logic

This document explains the internal architecture and identity logic of the **main-branch multi-camera Person ReID system**.

The focus is the maintained **State-Invariant V6 / Safe055 rebuild path**, with the root modular architecture described where it provides important context.

---

# 1. Core design principle

The repository is deliberately layered.

```text
Detection
    ↓
Tracking
    ↓
Tracklet Formation
    ↓
Crop / Quality Processing
    ↓
ReID Feature Extraction
    ↓
Local Identity Proposal
    ↓
State Memory
    ↓
Same-camera Reconciliation
    ↓
Cross-camera Reconciliation
    ↓
Persistent Identity
    ↓
Rendering
```

No single component should be expected to solve all of these tasks.

The biggest conceptual mistake in multi-camera ReID systems is treating:

```text
detector confidence
tracker ID
embedding similarity
global identity
```

as if they were interchangeable.

They are not.

---

# 2. Stage 1 — Input

The pipeline supports:

```text
recorded video files
RTSP / stream sources
```

For multiple cameras, each source receives a stable logical camera name.

The camera name is important because identity reasoning is not purely visual.

It is also conditioned by:

```text
where the track was observed
when it was observed
what other observations co-existed
whether movement is physically plausible
```

---

# 3. Stage 2 — Frame acquisition

## Batch path

Recorded frames are read from each video source.

## Live path

The modular live pipeline contains:

```text
Decode backend
    ↓
Capture thread
    ↓
Newest-frame slot / queue
    ↓
Batch scheduler
    ↓
Inference
```

The live architecture separates capture from model inference so slow GPU/CPU work does not directly block stream acquisition.

The design also allows controlled frame shedding under load rather than allowing unbounded queues.

---

# 4. Stage 3 — Person detection

The state-final path uses YOLO11m.

Conceptually:

```text
frame
  ↓
YOLO
  ↓
candidate boxes
  ↓
person-class filtering
  ↓
confidence/NMS filtering
  ↓
person detections
```

Each detection contains at least:

```text
bbox
class_id
confidence
track_id
```

The repository's `Detection` data structure also supports:

```text
global_id
reid_id
embedding
crop_quality
reid_score
keypoints
```

The important point is that these are progressively filled by later stages.

---

# 5. Stage 4 — ByteTrack

Tracking solves temporal continuity inside one camera.

```text
frame t
  person A → track 1
  person B → track 2

frame t+1
  person A → track 1
  person B → track 2
```

The tracker uses motion/box association rather than the full multi-camera identity problem.

Therefore:

```text
track 1 in camera A
```

does not mean:

```text
Global Person 1
```

It only means:

```text
camera A's local tracker currently labels this trajectory as 1
```

A track can:

```text
start
continue
fragment
switch
disappear
restart
```

That is why the identity system must exist above the tracker.

---

# 6. Tracklets

A tracklet is a temporally connected set of observations associated with a local track.

A useful mental model:

```text
track_id + time span + boxes + appearance observations
```

The tracklet becomes the unit from which the identity resolver can reason.

A good tracklet contains:

```text
one real person
multiple observations
reasonable spatial continuity
multiple appearance views
```

A bad tracklet can contain:

```text
a person + another person after a tracker switch
too few observations
heavy occlusion
very poor crops
```

This distinction is critical because identity reconciliation assumes its evidence is meaningful.

---

# 7. Stage 5 — Crop extraction and quality

For each valid tracked person:

```text
frame + bbox
      ↓
safe person crop
      ↓
quality evaluation
      ↓
accepted / rejected
```

Quality features include concepts such as:

```text
width
height
area
aspect ratio
blur
brightness
box-area ratio
occlusion
keypoint visibility
```

The state-final rebuild also performs additional views:

```text
full
light
upper
torso
lower
```

Why quality matters:

```text
bad crop
   ↓
bad embedding
   ↓
bad identity score
   ↓
bad gallery state
   ↓
wrong future matches
```

This is not an isolated frame problem. A corrupted observation can contaminate later decisions.

---

# 8. Stage 6 — ReID feature extraction

The state-final pipeline runs three models:

```text
NVIDIA ResNet-50
NVIDIA Swin Base
SOLIDER Swin Base
```

Each crop generates model-specific features.

The feature extractor's contract is conceptually:

```text
person crop
    ↓
model preprocessing
    ↓
model forward pass
    ↓
raw feature vector
    ↓
L2 normalization
    ↓
ReID embedding
```

An embedding says:

> This is the appearance representation of this crop.

It does **not** say:

> This crop belongs to Global ID 17.

Identity assignment is a separate stateful problem.

---

# 9. Multi-model reasoning

Why not trust one model?

Because the models encode different learned representations.

A useful decision abstraction is:

```text
ResNet similarity
        +
Swin similarity
        +
SOLIDER similarity
        ↓
model support / fused evidence
        ↓
identity proposal
```

The state-final cross-camera resolver requires support from multiple models.

Configured minimum:

```text
2 of 3 models
```

This reduces the risk of one model producing an unusually high similarity for the wrong person.

It does not eliminate false matches.

---

# 10. Multi-view reasoning

A single full-body crop can be unreliable because:

```text
legs disappear
upper body is blocked
person sits
person rotates
camera crops differently
lighting changes
```

Therefore the system can maintain multiple body views:

```text
FULL
LIGHT
UPPER
TORSO
LOWER
```

The logic is evidence aggregation, not majority voting between “different persons.”

For example:

```text
full      → strong
upper     → strong
torso     → medium
lower     → missing
```

can still be a good identity sample.

---

# 11. Local identity state

The V6 state layer maintains appearance state for each active identity/track representation.

It retains multiple observations rather than one permanent vector.

Conceptually:

```text
new observation
      ↓
quality gate
      ↓
embedding
      ↓
compare to candidate identities
      ↓
evidence accumulation
      ↓
local identity proposal
      ↓
state update
```

This is more robust than:

```text
one crop
   ↓
one nearest neighbour
   ↓
permanent identity
```

---

# 12. Same-camera fragment repair

Suppose:

```text
camera_1
  Track 12
     ↓
  lost
     ↓
  Track 31
```

The system asks:

```text
Could Track 31 plausibly be the same person as Track 12?
```

Evidence can include:

```text
appearance similarity
model support
temporal gap
camera continuity
trajectory continuity
position
```

The resolver should favor:

```text
same person + tracker reset
```

when evidence is strong enough, while avoiding:

```text
different people accidentally merged
```

The second error is usually more damaging because it corrupts one identity with two people's observations.

---

# 13. Global cross-camera identity

Cross-camera matching is harder than same-camera matching.

The problem is:

```text
Camera A
  person P

Camera B
  person P
```

The crops may differ because of:

```text
viewpoint
scale
illumination
background
pose
partial visibility
camera color response
time gap
```

The cross-camera resolver therefore combines:

```text
multi-model appearance
+
multi-view evidence
+
temporal compatibility
+
attributes
+
trajectory
+
geometry / topology when available
```

The goal is not simply:

```text
highest cosine wins
```

but:

```text
highest plausible identity
subject to identity and physical constraints
```

---

# 14. Severe overlap state machine

Severe overlap is treated as a state transition.

```text
NORMAL
   │
   │ severe overlap detected
   ▼
OVERLAPPED
   │
   │ tracking / position continues
   │ appearance state protected
   ▼
OVERLAP CLEARS
   │
   ▼
RECOVERY
   │
   │ dense clean multi-model observations
   ▼
IDENTITY CHECK
   │
   ├── match
   │     ↓
   │  restore same identity
   │
   └── fail
         ↓
      continue guarded recovery /
      create or assign a different identity
      according to resolver policy
```

The point is not merely “ignore overlapping frames.”

The system preserves temporal and spatial continuity while preventing contaminated visual evidence from immediately changing identity.

---

# 15. Pre-overlap identity anchor

Before severe overlap, the pipeline can preserve clean full-body appearance references.

Conceptually:

```text
clean observations
   ↓
anchor bank
   ↓
severe overlap
   ↓
appearance protected
   ↓
clean observations after overlap
   ↓
compare against anchor
```

The anchor is valuable because it represents the person immediately before the ambiguous event.

This is better than comparing only against a generic global gallery when the specific question is:

> Did the same tracked person emerge from this overlap?

---

# 16. Post-overlap recovery

The state-final rebuild forces dense feature extraction after the overlap.

Recovery evidence uses:

```text
ResNet
Swin
SOLIDER
```

and requires multiple-model support.

Typical settings:

```text
required_models = 2
fused_min       = 0.56
resnet_min      = 0.52
swin_min        = 0.52
solider_min     = 0.50
```

The recovery logic can inspect multiple clean samples.

The underlying idea is:

```text
one frame after overlap
    ≠
sufficient evidence

several clean frames after overlap
    →
more trustworthy state reconstruction
```

---

# 17. Trajectory logic

The state-final pipeline stores per-observation position history.

Conceptually:

```text
frame
timestamp
bbox
center
height
```

The center can be derived from:

```text
x_center = (x1 + x2) / 2
y_center = (y1 + y2) / 2
```

The history is retained for a bounded number of observations.

The system can then reason about:

```text
continuity
direction
movement consistency
fragment linkage
post-overlap behavior
```

Tracking position every observation is separate from extracting ReID features. A frame can contribute trajectory information without contributing a fresh embedding.

That separation is important for throughput.

---

# 18. Geometry and reachability

The geometry subsystem contains:

```text
calibration
floor mapping
reachability
recording
```

The intended architecture is:

```text
live run
   ↓
camera observation
   ↓
calibrated position
   ↓
record position with observation
```

Later reconciliation consumes the recorded positions.

A good architectural rule is:

```text
compute / record geometry once
        ↓
reuse recorded geometry
```

rather than recomputing different geometry during later reconciliation.

This keeps a reconciliation deterministic for a given run.

---

# 19. Identity decision philosophy

The project is intentionally conservative.

Why?

Because two major errors exist:

### False split

One real person receives:

```text
GID 7
GID 18
```

This is bad, but often repairable with later reconciliation.

### False merge

Two real people receive:

```text
GID 7
```

This corrupts both histories.

The architecture therefore biases uncertain cases toward:

```text
new identity / unresolved
```

rather than forcing a merge on weak evidence.

That is not the same as blindly using a high threshold. The system uses multiple gates and contextual constraints.

---

# 20. Persistent identity store

The project uses **Qdrant** for persistent identity-vector storage and similarity retrieval. Qdrant stores the searchable embedding gallery used to retrieve candidate identities across observations and runs.

Conceptually:

```text
ReID embedding
      ↓
Qdrant vector collection
      ↓
nearest candidate identities
      ↓
identity resolver / validation
      ↓
Global ID
```

Qdrant does not generate the ReID embeddings and does not by itself make the final identity decision. It provides persistent vector storage and similarity search for the identity layer.

The stored collection is coupled to its embedding space. For a model change:

```text
new model
   ↓
different feature space
   ↓
existing vectors may no longer be directly comparable
   ↓
use a new collection / migrate deliberately
```


# 21. Root modular identity architecture

The root `src/identity/` system provides a more general identity service abstraction.

Important components include:

```text
src/identity/service.py
src/identity/reconcile.py
src/identity/reranking.py
src/identity/verifier.py
src/identity/decision_log.py
```

The root `IdentityService` follows a:

```text
candidate
   ↓
decide
   ↓
commit
```

model.

A new observation is matched against stored candidates, a decision is made, and the observation is committed to the persistent gallery.

The service also supports an optional online/live evidence accumulation path.

This root architecture is more modular than the direct rebuild runner, but it is not the same runtime configuration.

---

# 22. Re-ranking and verification in the root stack

The root identity layer includes optional mechanisms for:

```text
camera-aware re-ranking
verification
decision logging
```

The principle is:

```text
raw retrieval
    ↓
candidate re-ranking
    ↓
verification
    ↓
identity decision
```

This is useful because nearest-neighbour retrieval and final identity acceptance are different problems.

A nearest candidate is not necessarily a valid identity match.

---

# 23. Offline reconciliation

The root architecture also contains a dedicated:

```text
src/identity/reconcile.py
```

stage.

The rationale is important.

A live camera worker may decide:

```text
Camera A → GID 10
Camera B → GID 14
```

before either camera has enough information about the other.

Once the complete run is available, offline reconciliation can inspect the full gallery and discover:

```text
GID 10 == GID 14
```

and merge the histories subject to safety constraints.

This is especially valuable in multi-camera batch processing because the complete run has more information than any single camera worker had at decision time.

---

# 24. Reconciliation safety

The root reconciliation design includes several important protections.

## Appearance gate

A candidate pair must clear an appearance threshold.

## Physical exclusion

Two identities that are simultaneously present in the same camera should not be merged merely because they look similar.

## Transitive safety

Cluster unions must not allow a chain of pairwise matches to merge physically incompatible tracks.

## Reciprocal best match

A candidate pair can be required to be mutual best matches rather than allowing one identity to absorb many look-alikes.

## Geometric reachability

A proposed identity chain should remain physically plausible in the available time.

This leads to a safer mental model:

```text
appearance says:
    "these could look like the same person"

constraints say:
    "this merge is physically and temporally possible"

identity resolver says:
    "accept or reject"
```

---

# 25. Live modular pipeline

The root live pipeline is explicitly staged:

```text
DecodeBackend
    ↓
CaptureThread
    ↓
NewestSlot
    ↓
BatchScheduler
    ↓
InferenceStage
    ↓
IdentityStage
    ↓
RenderStage
    ↓
WriterStage
```

The design separates model inference from capture and rendering.

Startup is organized around:

```text
capability report
    ↓
model warm-up
    ↓
consumer stage startup
    ↓
capture startup
```

Shutdown is designed to drain and finalize all stages so output videos are not left incomplete.

---

# 26. Performance logic

The system must balance:

```text
detection cost
tracking cost
ReID cost
camera count
queue pressure
GPU memory
video decode cost
```

A common mistake is to treat “more model inference” as automatically better.

It is not.

Excessive ReID inference can create:

```text
GPU saturation
CPU starvation
queue growth
frame drops
unstable tracking
more fragmented tracklets
```

which can ultimately reduce identity accuracy.

Therefore sampling and batching are part of identity quality, not only performance optimization.

---

# 27. Why sampling is used

A person's appearance changes very little between adjacent frames.

Therefore:

```text
frame t
frame t+1
frame t+2
```

often provide nearly redundant evidence.

The embedder can instead sample:

```text
frame t
frame t+k
frame t+2k
```

and keep the latest valid feature between fresh extractions.

Warmup and recovery are deliberately denser because those stages need fresh evidence faster than normal steady-state tracking.

---

# 28. Diagnostic interpretation

When GIDs are wrong, inspect in this order:

```text
1. Detection
2. Tracking
3. Tracklet purity
4. Crop quality
5. ReID embeddings
6. Local identity proposal
7. Same-camera reconciliation
8. Cross-camera reconciliation
9. Persistent state
10. Rendering
```

This prevents the common mistake of “fixing” a detector or tracker failure with an identity threshold.

---

# 29. Example failure propagation

Suppose YOLO misses a person for several frames:

```text
missing detection
    ↓
track interruption
    ↓
new track ID
    ↓
short tracklet
    ↓
few ReID samples
    ↓
weak identity evidence
    ↓
cross-camera split
```

Changing a ReID threshold does not solve the first failure.

Likewise:

```text
tracker switch
    ↓
one track contains two people
    ↓
mixed appearance prototype
    ↓
wrong cross-camera matches
```

Again, the root problem is tracking, not the final threshold.

---

# 30. The correct way to improve the system

Use measurable experiments.

For example:

```text
baseline
   ↓
one change
   ↓
same validation run
   ↓
measure:
    detector recall
    track purity
    false merges
    false splits
    cross-camera recovery
    overlap recovery
    runtime
    GPU use
   ↓
accept / reject change
```

Avoid changing:

```text
YOLO
+ ByteTrack
+ ReID model
+ threshold
+ geometry
```

all at once.

That creates an experiment with no useful causal interpretation.

---

# 31. Final architecture summary

The maintained state-final design can be reduced to:

```text
DETECT
  ↓
TRACK
  ↓
CLEAN
  ↓
EMBED
  ↓
ACCUMULATE
  ↓
PROTECT
  ↓
RECOVER
  ↓
RECONCILE
  ↓
PERSIST
  ↓
RENDER
```

The most important architectural rule remains:

```text
A tracker ID is not an identity.
An embedding is not an identity.
A similarity score is not an identity.

Identity is a stateful decision made from appearance evidence
plus temporal, camera, and physical constraints.
```
