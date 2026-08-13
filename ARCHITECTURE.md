# Architecture

Multi-camera person re-identification (ReID). Given one source per camera — a
**live RTSP stream** (`--mode live`) or a **recorded video file** — the system
detects and tracks people, embeds their appearance, and assigns a **reid id**
that is stable for the same real person **across cameras** and **across
re-appearances** within a camera. The pipeline is fully self-contained: it
builds and owns its own Qdrant gallery — there is no separate registration step
or external service. Output: annotated videos (boxes + reid-id labels, with
`global_id` kept for compatibility) and a console run-summary. No crop images
are written to disk — the embedder makes its own in-memory crop.

There are two entry paths, and **both settle cross-camera identity with the same
offline reconcile** (`identity/reconcile.py`):

- **File-batch** (`main.py`, [section 2](#2-data-flow)) — a worker thread per video file; live sticky
  assignment during the pass, then offline reconcile + re-render.
- **Live streaming** (`src/live/`, [section 8](#8-live-streaming-pipeline--srclive)) — a real-time, load-shedding, per-camera
  threaded pipeline for RTSP. It never records the raw feed; it persists
  observations + the processed frames during the run and, on stop, runs the
  same offline reconcile to produce the corrected `output_<cam>.mp4`.

> **How cross-references are written here.** "**section N**" means the numbered
> section of **this** file (1–8 below); a reference to another document names it
> first, e.g. "[README.md → section 6](README.md#6-run-the-pipeline)". All of
> them are clickable links. (The old bare section-sign shorthand, which never
> said which document it meant, has been replaced everywhere.)

---

## 1. Design principle: separation of concerns

Three layers, kept strictly independent (see `src/identity/DESIGN.md`):

| Layer | Module | Knows about | Does NOT know about |
|---|---|---|---|
| **Appearance** | `reid/extractor.py` | one crop → an L2-normalised vector (2048-d under the current backend) | cameras, time, identity |
| **Storage** | `database/store.py` | vectors + payloads, nearest-neighbour search | how vectors are produced, camera topology |
| **Identity** | `identity/service.py`, `identity/reconcile.py` | appearance + **where** + **when** | how the model computes a vector |

Why: the model sees one crop with no context; the store only knows vector
distance. Only the identity layer combines appearance with *where* and *when*
a person was seen, so only it is allowed to decide who someone is. That
boundary keeps the code maintainable.

ADR-002 adds two more stages **inside** the identity layer's decision, between
"find candidates" and "decide":

| Stage | Module | Purpose |
|---|---|---|
| **Re-ranking** | `identity/reranking.py` | camera-aware k-reciprocal + Jaccard re-ranking of candidates, on top of raw cosine |
| **Verification** | `identity/verifier.py` | scores each candidate's P(same identity) from multiple signals instead of a single rigid threshold |

---

## 2. Data flow

```
                        config.yaml
                            │
                            ▼
        ┌──────────────  main.py  ──────────────┐
        │   one worker THREAD per camera video   │
        └────────────────────────────────────────┘
                            │
   per frame, per camera:   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. VideoSource          video file      → frame              │
   │ 2. (optional) resize    frame            → frame (downscaled) │
   │ 3. PersonDetector       frame            → [Detection + track_id]  (YOLO11n + ByteTrack)
   │ 4. TrackEmbedder        crop             → det.embedding (2048-d) [shared model lock]
   │      └─ quality gate + occlusion gate + throttle/cache          │
   │ 5. IdentityService      embedding+where+when → det.reid_id     [identity lock]
   │      └─ candidate (Qdrant search) → re-rank (Upgrade 1) →       │
   │         verify (Upgrade 2) → decide → COMMIT to the gallery     │
   │ 6. drawing → live display window; box geometry captured for the │
   │      final render (the video is NOT written in this pass)       │
   └─────────────────────────────────────────────────────────────┘
                            │  (all camera threads join)
                            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 7. reconcile_tracklets  OFFLINE, whole-gallery view:           │
   │      rebuild identities from tracklets, merge across cameras   │
   │ 8. render_final_videos  re-draw with FINAL reid ids →           │
   │      output_<cam>.mp4  (same person = same REID in every video) │
   │ 9. print_run_summary   → console only (no files written)      │
   └─────────────────────────────────────────────────────────────┘
```

---

## 3. Components

### VideoSource — `src/video_source.py`
Yields decoded frames from a video file, one at a time.

### PersonDetector — `src/detector.py`
YOLO11n (`yolo11n.pt`, confidence ≥ 0.4, COCO class 0 = person) with ByteTrack.
Returns `Detection(x1,y1,x2,y2, confidence, class_id, track_id, ...)`. `track_id`
is **per-camera** and stable frame-to-frame; it may be `None` until ByteTrack
confirms a box. `crop_person()` is the shared "box → safe crop" primitive.

### ReIDExtractor — `src/reid/extractor.py`
The configured backend (`reid.model`, today **FastReID SBS ResNet101-IBN /
MSMT17**), loaded once and shared. `src/reid/backends.py` owns the architecture,
checkpoint load, **preprocessing recipe** and feature tap; the extractor keeps the
backend-invariant contract (batching, `max_batch` chunking, the shared-model forward
lock, L2-normalisation, empty-crop rejection). Pipeline per crop: BGR→RGB → resize
to the backend's size (**384×128** for FastReID, 256×128 for OSNet) → `/255` →
ImageNet mean/std → forward in `eval()` mode (2048-d for FastReID, 512-d
feature) → **L2-normalize**. Unit vectors ⇒ cosine similarity == dot product.
Preprocessing is pinned to match training or embeddings silently degrade.

### TrackEmbedder — `src/reid/service.py`
One per camera (cache keyed by per-camera `track_id`). `process()`:
- **Throttle + cache**: re-embed a track at most every `reid.interval` (10)
  frames; reuse the cached vector otherwise. Evict tracks after `ttl` (300)
  unseen frames.
- **Quality gate** (`_crop_quality`): reject on size / box-area-ratio / aspect /
  blur (Laplacian variance) / brightness.
- **Occlusion gate** (`_occlusion_ratio`): reject if another person's box covers
  more than `max_occlusion_ratio` (0.5) of this box — keeps multi-body crops out
  of the gallery.
- Accepted crops go through **one batched forward pass**; rejected crops keep the
  track's last good vector.

### PersonVectorStore — `src/database/store.py`
Thin Qdrant wrapper. This is the ONLY gallery the pipeline uses — the identity
layer both writes to it and searches it.
- `add`/`add_many(embedding, payload)` → one point per raw observation.
- `search(embedding, k)` → k nearest by cosine.
- `set_global_id(points, gid)` / `clear_global_id(points)` → payload rewrites
  used by offline reconciliation.
- Backend precedence `client > url > path`; config uses `store.url` for a
  shared Docker/Cloud server, `store.path` for a local single-process folder.

### IdentityService — `src/identity/service.py`
The one component allowed to decide WHO someone is.
- Sticky per track: decide once on a track's first observation, cache
  `(camera, track_id) → reid_id`, reuse thereafter. `global_id` is still
  written in parallel for compatibility with older payload readers.
- `assign()` → `_match_or_mint()`:
  1. **Candidate retrieval**: `store.search()` for the top-`top_k` raw hits,
     filtered to already-identified, non-same-camera-overlapping candidates.
  2. **Re-ranking** (`identity.rerank.enabled`, Upgrade 1): if enabled,
     `CameraAwareReranker` re-scores candidates using real k-reciprocal
     neighbour sets + Jaccard overlap over identity prototypes, with a wider
     neighbourhood window for cross-camera pairs (see [section 5,
     ADR-002](#5-adr-002-why-re-ranking--verification-not-just-a-better-threshold)).
  3. **Verification** (`identity.verification.enabled`, Upgrade 2): if enabled,
     `Verifier` scores each candidate's `P(same identity)` from cosine, the
     re-ranked score, observation count, recency, and crop quality — replacing
     the plain `score >= threshold and gap >= min_score_gap` check. If
     disabled, falls back to that original plain check.
  4. Accept the best candidate only if its score/probability clears the
     configured threshold **and** beats the runner-up by the configured
     margin; otherwise mint a new identity. The same-camera-overlap exclusion
     is a **hard** pre-filter in both paths — never something a soft signal
     can override (two people present in the same camera at the same time can
     never be merged).
- `_commit()` stores the observation under the chosen id, updates its
  prototype bank, span cache, observation count, and last-seen record.
- Warm start rebuilds all of the above from the store on startup, so a
  persistent gallery survives process restarts.

### CameraAwareReranker — `src/identity/reranking.py` (ADR-002 Upgrade 1)
Real camera-aware k-reciprocal + Jaccard re-ranking (CA-Jaccard, CVPR'24) —
**not** `cosine × camera_weight`. Operates over per-identity prototypes (the
mean-pooled, re-normalized bank vectors `IdentityService` already keeps in
memory), since Qdrant only holds raw per-observation points.
- Builds a cached pairwise-cosine neighbour graph over all known identity
  prototypes (cheap on CPU — tens of identities, not millions); rebuilt only
  when the gallery signature changes.
- For a query embedding: finds its reciprocal neighbour set among known
  identities, widening the effective neighbourhood window
  (`cross_camera_k1_boost`) for cross-camera comparisons, since cross-camera
  cosine similarity runs systematically lower.
- Blends `lambda * cosine + (1 - lambda) * (1 - jaccard_distance)`.
- **Small-gallery guard**: with fewer known identities than `k1 + 1`,
  reciprocal sets degenerate to trivial/empty sets and would fabricate a
  "perfect" Jaccard overlap regardless of match quality — below that floor,
  it falls back to plain cosine instead.

### Verifier — `src/identity/verifier.py` (ADR-002 Upgrade 2)
A HEURISTIC placeholder for the "verification layer" — a fixed-weight logistic
combination of signals, not (yet) a trained model, since there is no labeled
same/different-person dataset for this deployment. The interface —
`Verifier.score(features) -> float` — is the same shape a trained
classifier would expose (`model.predict_proba(...)`), and every decision's
full feature vector + resulting probability is appended to
`identity.verification.log_path` (`logs/verification_decisions.jsonl`), so a
real MLP/GBT can be trained on accumulated deployment data later with **no
change to the calling code**.
- Features: cosine score, re-ranked score, `log1p(observation_count)`, frames
  since last same-camera sighting, and a crop-quality scalar.
- Weights are calibrated so cosine ≈ 0.63 sits near the decision boundary --
  measured on this project's footage with the `osnet_x1_0_msmt17` checkpoint.
  After swapping to `osnet_ain_x1_0` (see [section 6, "Known
  limitations"](#6-known-limitations-model-not-plumbing)), the decision log
  (`logs/verification_decisions.jsonl`) showed accepted matches averaging
  cosine ~0.79 (min 0.53), rejected candidates averaging ~0.45 (p90 0.59) --
  a wider, cleaner gap. Raising `accept_threshold`/`min_prob_gap` (0.55→0.62,
  0.05→0.08) to exploit that margin was tried but measurably hurt overall
  accuracy (97% -> 89%, more false splits) and was reverted; left at 0.55/0.05.
  **Re-measure and re-anchor these weights whenever the ReID checkpoint or
  camera domain changes** — they are specific to this model's score distribution, not a
  universal constant.

### Legacy reconcile_tracklets — `src/identity/reconcile.py`
Runs once after all cameras finish, with the full gallery visible. Rebuilds
identities from scratch (does **not** trust live `reid_id`s); the **tracklet**
(one camera's view of one track) is the unit of evidence.
1. **Gather** points by `(camera, track_id)` → vectors, point ids, frame span,
   gids.
2. **Suppress** tracklets with `< min_tracklet_observations` (3) via
   `clear_global_id` — 1–2-frame detector blips don't become people.
3. **Prototype** per tracklet (mean of normalized vectors).
4. **Union-find** with hard conflict guards, all statements about where one body
   can be rather than about how alike two crops look: same-camera,
   time-**overlapping** tracklets can never merge (provably two people);
   cross-camera pairs overlapping in time in cameras that cannot both see one
   person (`covisibility`); and pairs whose **recorded floor positions** would
   require impossible speed (`GEOMETRIC_UNREACHABLE`, see
   [the geometry section](#geometry--srcgeometry)). All fail open.
5. **Phase 1 — same-camera defrag**: merge same-camera, time-disjoint pairs with
   cosine ≥ `same_camera_threshold` (0.90).
6. **Phase 2 — cross-camera, iterated to convergence**: on cluster prototypes,
   merge different-camera pairs with cosine ≥ `threshold`
   (`identity.reconcile.threshold`, 0.63) **and** (if `require_reciprocal_best`)
   mutual nearest neighbour — stops one tracklet absorbing multiple
   look-alikes. This is not a single pass: after any merges, cluster
   prototypes change, so scores and reciprocal-best partners are recomputed
   from scratch and another round runs, until a round merges nothing. This
   matters for **chains** of mutually-similar fragments — e.g. A's own best
   match is B, but B's own best match is C, not A — which a single pass would
   leave stuck (A↔B never merges because B "prefers" C). Iterating lets the
   chain fully consolidate (A merges with B in round 1; the merged A+B
   cluster's new prototype then reciprocally matches C in round 2) while every
   round still enforces the same mutual-best-match safety rule — merging never
   gets easier than that rule allows, it just gets more chances to apply as
   clusters grow. Each merge strictly reduces the number of clusters by one, so
   this always terminates. This runs **independently of the verifier** above
   (its own separate threshold, not the verifier's `accept_threshold`) —
   recalibrate both together if you change the ReID checkpoint.
7. **Survivor**: each final cluster keeps the smallest existing reid ID
   (deterministic); all its points rewritten via `set_global_id` so both
   `reid_id` and `global_id` stay aligned.

### geometry — `src/geometry/`
Answers one question: *could one person have been in both of these places?* It is a
**check** on identity, not a tracker — nothing here does 3D tracking, and nothing
here can create or merge an identity. It can only refuse a merge.

    src/geometry/calibration.py   the floor-frame record + the metric-scale guard
    src/geometry/floor.py         bbox -> point on a shared floor (owns the homography)
    src/geometry/reachability.py  two recorded points -> possible / impossible
    src/geometry/recorder.py      the LIVE run's writer -- the only place a position
                                  is ever computed
    tools/fit_floor_frame.py      fits the floor frame from people's own foot points
    tools/backfill_geometry.py    applies a fresh calibration to a run captured
                                  BEFORE it existed, so one capture suffices

One formula: `required_speed = distance / elapsed`, vetoed above a speed ceiling.
At `elapsed ≈ 0` that is "one body cannot be in two places"; at `elapsed > 0` it is
"could not have got there in time". `src/live/topology.py`'s hand-set min-transit
veto is **superseded** by it and stays disabled — it asserted 2–3 s minimums between
adjacent cameras and pruned the true match; measured positions cannot make that
mistake, because overlapping cameras give a distance near zero.

**Three invariants.** Each is enforced, not merely documented — see
`tests/live/test_geometry_*.py` and `ADR-003D`.

1. **The live run records geometry; offline reconcile only consumes it.** Positions
   are computed once, at capture, and stored in the observation payload under
   `floor`. `reconcile.py` never loads a calibration, applies a homography, or
   derives a position from a box: it may import `geometry.reachability` (pure
   arithmetic) and nothing else under `geometry/`. Why — the live feed is never
   recorded, so an unwritten position is gone; and if reconcile derived positions,
   re-fitting a calibration would silently change a finished run's identities.
2. **Units are floor units, not metres.** The frame is fitted from imagery alone,
   which fixes the plane only up to scale. `is_metric` is False and the
   metre-facing API raises until a trusted metric reference is recorded (verified
   floor plan/CAD, verified architectural dimension, or an independently measured
   reference distance). Nothing is lost: the speed ceiling is measured from the same
   footage in the same unit, so the unknown scale cancels. Metres are needed only to
   relate **separate floor groups**, which is why `cam_206` and `cam_213` — which
   overlap nothing — have no geometry at all.
3. **It fails open, always.** Uncalibrated camera, missing box, mismatched image
   size, cameras in different floor groups, no timestamp, an unmeasured speed
   ceiling → *unavailable*, treated as no opinion. Every error budget biases toward
   permitting the merge, because refusing one is unrecoverable while missing one
   leaves a false merge that already happens today.

Ships disabled (`geometry.enabled`, `geometry.reconcile.enabled` both false).
File-batch mode records nothing — `IdentityService._commit` stores neither `bbox`
nor `ts` — so use `--mode live` to run geometry over recorded footage.

### Two clocks — `src/live/frame.py`
`ts` is stamped once at frame read and answers *"when did this frame reach the
pipeline"* — the right question for scheduler freshness, writer pacing and the live
engine's TTLs. For a **live camera** it doubles as "when did this happen", because
the read follows the event by milliseconds and all cameras share one machine clock.

For a **recorded file it does not.** Frames decode as fast as the disk allows (125+
fps measured), so `ts` tracks decode progress; two files read in parallel get
timestamps whose difference reflects *thread scheduling*. Anything cross-camera and
time-sensitive — the co-presence veto, all of geometry's co-temporal pairing — would
then rest on invented simultaneity that looks entirely plausible.

So a file source also carries `source_ts = offset + frame_index / source_fps`, and
**`event_ts()`** is what every "when did this happen" consumer reads: the stored
payload's `ts`, geometry, and the render sidecar. The machinery deliberately stays on
`ts` — notably the live engine, whose TTL bookkeeping is paired with
`sweep(time.time())` and would evict every identity on the first pass if fed media
time. Its ids are provisional, so that asymmetry costs nothing.
`live.capture.file_time_offsets` lines up recordings that were not started together.
Both halves are pinned by `tests/live/test_media_time.py`.

### render_final_videos — `main.py`
Second render pass, after reconciliation. The live pass only captured per-frame
box geometry; this maps each `(camera, track_id)` to its FINAL reid ID
(`build_gid_map`, read back from the store) and re-draws the source frames so the
same person carries the **same REID and colour in every camera's `output_<cam>.mp4`**.

### print_run_summary — `main.py`
Scrolls the store for this `run_id`; counts observations per camera and distinct
reid IDs; flags IDs seen in >1 camera as cross-camera. **Console output only —
writes no files.**

---

## 4. Concurrency

- One **worker thread per camera video**; each owns its detector (independent
  ByteTrack state), crop saver, and TrackEmbedder.
- **Shared, lock-guarded**: the ReID model (`model` lock — one forward pass at a
  time) and the identity service (`identity` lock — serializes `assign()`,
  since it both searches and writes the shared gallery).
- OpenCV windows aren't thread-safe: workers publish frames to a shared buffer;
  the main thread displays (or just joins, headless).

---

## 5. ADR-002: why re-ranking + verification, not just a better threshold

Testing surfaced the failure mode that motivated this: with the original
OSNet/Market1501 checkpoint on this footage, **different people scored closer
than the same person** (a genuine same-person cross-camera match at cosine
~0.72-0.76 vs. different-but-similar-looking people reaching ~0.83). The two
distributions overlapped, so **no single cosine threshold was simultaneously
correct** — raising it fixed false merges but created false splits, and vice
versa. Switching to an OSNet/MSMT17 checkpoint substantially widened that gap
on this footage, and a later swap to OSNet-AIN (`osnet_ain_x1_0`, same family,
added Adaptive Instance Normalization for domain generalization) widened it
further still (see [section 6](#6-known-limitations-model-not-plumbing)) — but the underlying risk (some future camera/domain
reintroducing overlap) is exactly what re-ranking and verification exist to
guard against, not something a single "right" checkpoint permanently solves.

Design principles this drives:
- **False merges are worse than false splits.** A merge fuses two people's
  histories and corrupts both; a split just gives one person a spare ID,
  which reconciliation can still fix later. When uncertain, mint a new
  identity rather than guess.
- **Appearance generates candidates; it doesn't decide identity alone.**
  Re-ranking (Upgrade 1) and verification (Upgrade 2) exist to combine
  appearance with additional structure (neighbourhood overlap, observation
  history, crop quality) before a decision is made — not to replace the
  embedding model. This constraint held through the FastReID switch, which was a
  bigger change than ADR-002 anticipated — a **different architecture and a
  different width** (OSNet 512-d at 256×128 → FastReID R101-IBN 2048-d at 384×128)
  — and still required **zero** change to identity logic, because `backends.py`
  owns the architecture and preprocessing while `extractor.py` owns the invariant
  contract. What would breach the constraint is a model that stops producing one
  L2-normalised vector per crop.
- The hard same-camera-overlap physical exclusion is never overridden by a
  soft signal — a physical impossibility is a different kind of fact than a
  similarity score, and stays a hard veto in both the legacy and new decision
  paths.

**Deferred** (documented, not implemented): prototype confidence/variance with
adaptive per-identity thresholds; a MetaBIN training pass to reduce the
underlying domain gap. These need accumulated deployment data this project
doesn't have yet — see `identity/verifier.py`'s decision log as the mechanism
that starts collecting it.

**Built, then disabled**: the camera transition graph rejecting
physically-impossible transitions exists in the live path (`live/topology.py`,
`live.topology.enabled: false` — see [section 8.2](#82-key-live-configuration-configyaml--live)).
Measured transit times were entered and A100-tested; it pruned the *true*
cross-camera match (cross-camera links 5 → 1, `topology_pruned=508`) because
these cameras' views are adjacent or overlapping, so the minimum transit time
is effectively zero. Transit-time vetoes need cameras with a real gap between
fields of view.

---

## 6. Known limitations (model, not plumbing)

The pipeline is correct; the **embedding model is the ceiling** on this domain.

- **History**: the original OSNet/Market1501 checkpoint was out-of-domain for
  this CCTV footage. Measured directly on this project's own videos: genuine
  same-person matches across two different video sessions only reached
  **cosine 0.65–0.76** (even after denoising via prototype averaging), while
  *different* people reached **0.70–0.83** — the distributions **overlapped**,
  so no fixed threshold could perfectly separate them.
- **Previous default (OSNet/MSMT17)**: re-measured on the same footage, this
  checkpoint separated the two cases better than Market1501 — different
  people's prototype cosine topped out **~0.48–0.57**, genuine same-person
  cross-video matches reached **~0.70–0.80** — but on this project's actual
  CCTV footage the two clusters still ran close enough (~0.72 vs. ~0.55) that
  a domain-generalization backbone was worth trying.
- **Previous default (OSNet-AIN, `osnet_ain_x1_0`)** — superseded on 2026-07-31
  by FastReID R101-IBN, below. Kept because **every threshold in `config.yaml` was
  derived in this feature space** and none has been re-anchored since: same OSNet
  family and 512-d embedding, with Adaptive Instance Normalization added to
  generalize to unseen camera domains — a near-drop-in swap (same torchreid
  API, same 256×128 preprocessing). Measured from the decision log on this
  project's 3-video run: accepted (same-person) matches average cosine
  **~0.79** (min 0.53), rejected candidates average **~0.45** (p90 0.59) — a
  wider, cleaner gap than MSMT17's. Despite that margin, raising
  `identity.threshold`/`identity.reconcile.threshold` (to 0.68) and the
  verifier's `accept_threshold`/`min_prob_gap` (to 0.62/0.08) to exploit it
  was tried and reverted -- it measurably hurt end-to-end accuracy (97% ->
  89%) via more false splits, so all four stay at their MSMT17-era values
  (0.63 / 0.55 / 0.05) for now (sections [3](#3-components) and
  [7](#7-key-configuration-configyaml)). The occlusion crop-quality gate
  (`max_occlusion_ratio`) was also tried tighter (0.35) to fight a rare
  occlusion-triggered false merge and reverted for the same reason -- see
  [section 7](#7-key-configuration-configyaml).
- **This is footage-and-checkpoint-specific, not solved in general.** A
  different deployment (different cameras, lighting, clothing diversity)
  could easily reintroduce distribution overlap even with a new checkpoint. Re-run
  `tests/calibration/measure_score_separation.py` (or inspect
  `logs/verification_decisions.jsonl` directly) whenever you point this at
  new footage, and re-anchor the verifier's weights and thresholds if the gap
  changes. If overlap persists, escalate to a foundation backbone (CLIP-ReID /
  SOLIDER) — that's a bigger change (different embedding dimension, Qdrant
  schema rebuild, new preprocessing) done as a separate step.
  That's why re-ranking and verification exist as a safety net on top of raw
  cosine, and why the verifier logs decisions instead of claiming to be a
  finished, trained classifier.
- **Current default (2026-07-31): FastReID SBS ResNet101-IBN, MSMT17.**
  `reid.model: fastreid_sbs_R101_ibn`, from the vendored definition in
  `src/reid/vendor/fastreid/` (FastReID has no `setup.py`, so it cannot be
  pip-installed; see that directory's `PROVENANCE.md`). **2048-d, 384×128 input,
  post-bnneck features, GeM pooling with a learned exponent (p=2.138).** Upstream
  MSMT17 numbers: Rank@1 84.8% / mAP 62.8%. There is no feature tap to choose —
  FastReID's `EmbeddingHead` returns the post-bnneck feature unconditionally at
  eval, so the backend requires `reid.tap: n/a` and raises otherwise. That means
  the full sphere is available by construction: measured on random crops, **60.5%
  of dimensions are negative and 0% exactly zero**, against OSNet post-ReLU's 0%
  negative / 21% zero — the property the #39 tap experiment was chasing.
  **Every threshold in `config.yaml` predates this switch and is therefore void**,
  and the Qdrant collection must be rebuilt (512-d → 2048-d). Neither has been
  done yet; see the migration note below.
- **Swapping the backbone is a config edit, not a refactor.** `reid.model` in
  `config.yaml` selects a backend registered in `src/reid/backends.py`
  (`fastreid_sbs_R101_ibn` | `fastreid_sbs_R50_ibn` | `osnet_ain_x1_0` |
  `osnet_ibn_x1_0` | `osnet_x1_0`). A backend owns the
  architecture, checkpoint load, preprocessing recipe and feature tap;
  `reid/extractor.py` keeps the backend-invariant contract (batching, the shared
  model's forward lock, L2-normalization, empty-crop rejection) so a backbone
  trial cannot silently break the invariants
  `tests/calibration/verify_embedding_contract.py` asserts. The Qdrant
  collection is sized from `extractor.embedding_dim` — measured from the loaded
  model — so a different-width backbone needs a collection rebuild but cannot
  quietly create a mis-sized one. The calibration harness reads the same config
  keys (`CALIB_REID_MODEL` / `CALIB_REID_WEIGHTS` override), so an A/B is two
  runs of `measure_score_separation.py` with no code edit.
  **A backbone change voids every threshold in `config.yaml`**, exactly like the
  feature tap: it is a different feature space, not better numbers in the same
  one. Derive thresholds *once*, after the backbone is chosen — not before.
- **If overlap reappears, the fix is still model-side**, roughly cheapest
  first:
  1. Try other available pretrained checkpoints (already done once here).
  2. Fine-tune the backbone on crops from your own cameras with a metric-learning
     loss (triplet / ArcFace) — directly targets a specific deployment's domain
     gap. **Currently shelved** (ADR-003D §1): there is no labelled pair set, and
     the on-disk crop path was *removed*, not merely disabled — `crop_saver.py`
     wrote nothing and was deleted, so this needs a real crop writer first, not a
     config flag.
  3. Synthetic pretraining (RandPerson) and/or MetaBIN (see [section
     5](#5-adr-002-why-re-ranking--verification-not-just-a-better-threshold)) — the
     "textbook" fix, but there is currently no training pipeline in this repo
     to build on (an earlier `tools/`/`notebooks/` scaffold for this was
     removed) — this would mean building one from scratch, with real GPU
     compute and experimentation time.

---

## 7. Key configuration (`config.yaml`)

| Key | Value | Meaning |
|---|---|---|
| `detector.confidence_threshold` | 0.4 | drop weak detections |
| `tracker.config` | bytetrack.yaml | tracker |
| `reid.model` | fastreid_sbs_R101_ibn | which backend in `reid/backends.py` runs. **Note `backends.DEFAULT_BACKEND` is `osnet_ain_x1_0`** — that is only the fallback when `reid.model` is absent, never what ships |
| `reid.weights` | msmt_sbs_R101-ibn.pth | ReID checkpoint — recalibrate everything below if this changes ([section 6](#6-known-limitations-model-not-plumbing)) |
| `reid.interval` / `ttl` | 10 / 300 | re-embed cadence / cache eviction |
| `reid.quality.max_occlusion_ratio` | 0.5 | reject multi-body crops |
| `store.enabled` | true | the pipeline's own gallery — always on for the default flow |
| `store.url` / `store.path` | http://localhost:6333 / qdrant_data | shared server vs. local single-process folder |
| `identity.enabled` | true | assigns and merges reid ids |
| `identity.threshold` / `min_score_gap` | 0.63 / 0.03 | plain-path match acceptance (used when verification is disabled) |
| `identity.top_k` / `bank_size` | 50 / 20 | candidates considered / prototype bank size |
| `identity.rerank.enabled` | true | ADR-002 Upgrade 1 |
| `identity.rerank.k1` / `cross_camera_k1_boost` | 8 / 4 | reciprocal-neighbourhood size / cross-camera widening |
| `identity.rerank.lambda_` | 0.7 | cosine vs. Jaccard blend weight |
| `identity.verification.enabled` | true | ADR-002 Upgrade 2 |
| `identity.verification.accept_threshold` / `min_prob_gap` | 0.55 / 0.05 | P(same) acceptance / runner-up margin |
| `identity.verification.log_path` | logs/verification_decisions.jsonl | decision log for future model training |
| `identity.reconcile.threshold` | 0.63 | cross-camera merge bar — separate from the verifier, recalibrate together |
| `identity.reconcile.same_camera_threshold` | 0.90 | same-camera defrag merge |
| `identity.reconcile.min_tracklet_observations` | 3 | below this = detector noise |
| `identity.reconcile.require_reciprocal_best` | true | mutual-NN guard on cross-camera merges |

---

## 8. Live streaming pipeline — `src/live/`

The real-time path (`--mode live`, or `auto` when any source is a stream URL).
It targets a GPU server for N live RTSP cameras and is **headless** — its
deliverable is `output_<cam>.mp4`. It never records the raw feed. Stage
boundaries and queue/drop policies are fixed; the discipline is **bound latency,
shed load**: queues are small and drop (and count the drop) rather than let lag
grow, so overload is a bounded, logged loss instead of ever-growing latency.

**Per-camera + shared stages** (one carrier `Frame` object flows through each):

```
per camera:  DecodeBackend -> CaptureThread -> NewestSlot            \
shared:      BatchScheduler -> inference_queue -> InferenceStage ->   > identity_queue
shared:      IdentityStage -> per-cam render_queue                   /
per camera:  RenderStage -> (writer / capture clip)
```

- **CaptureThread** (`capture.py`) decodes one camera; `NewestSlot` keeps only
  the freshest frame (drop-stale) so a slow consumer never sees a backlog.
- **BatchScheduler** (`scheduler.py`) gathers a freshness-bounded batch across
  cameras; **InferenceStage** (`inference.py`) runs per-camera detector +
  ByteTrack (one tracker per camera) and the shared, lock-serialised ReID
  extractor.
- **IdentityStage** (`identity_stage.py`) is a single serial thread wrapping the
  thread-free **IdentityEngine** (`identity_engine.py`): an in-memory active-set
  with the proven policies (evidence gate, same-camera-overlap hard veto,
  same-camera cold-reactivation, cross-camera reciprocal-best, mint-when-uncertain)
  plus a fail-open **topology** veto (`topology.py`). A `CameraFairQueue`
  (`priority.py`) interleaves cameras so none starves under a burst.
- **RenderStage** / **WriterStage** (`render.py`, `writer.py`) draw and encode;
  the writer paces to wall-clock in the non-reconcile mode.

### 8.1 Offline reconciliation on stop (`live.reconcile`, default on)

The online engine decides each track causally and can't see a person who is in
two cameras at the same instant — so its live ids are provisional. To deliver
correct ids without recording the raw stream, the live path mirrors the
file-batch flow using the frames it already processes:

1. **Persist observations** — `IdentityStage` writes every *fresh* embedding
   (already throttled to every `reid.interval` frames, optionally subsampled by
   `live.reconcile.sample_stride`) to the Qdrant gallery, with metadata
   `camera / track_id / frame / run_id / ts / bbox / confidence / crop_quality`
   and **no live id** (reconcile rebuilds identity from scratch).
2. **Capture processed frames** — `RenderStage` runs in *capture mode*: it writes
   each clean processed frame to a transient `._live_src_<cam>.mp4` and records
   that frame's box geometry, both from the same `Frame` so they stay
   index-aligned. This is the processed frames only — not the raw feed.
3. **On stop** (Ctrl-C / all sources ended / `max_duration`), after every stage
   has joined, `pipeline.py::_finalize_offline()` runs the file path's
   `reconcile_tracklets` → `render_final_videos` → `print_run_summary`
   **unchanged**, at the `identity.reconcile.*` thresholds — so a live run and a
   file run reconcile identically — then deletes the temp clips (kept with
   `live.reconcile.keep_frames`).

If the store is unreachable or `live.reconcile.enabled` is false, the live path
falls back to writing the immediate (online-id) `output_<cam>.mp4`.

### 8.2 Key live configuration (`config.yaml` → `live:`)

| Key | Meaning |
|---|---|
| `run.mode` | `live` / `auto` (live for stream URLs) / `batch` |
| `run.device` | `auto` (GPU if present) / `cuda:N` / `cpu` |
| `run.max_duration_sec` | 0 = until stopped (Ctrl-C / sources end) |
| `capture.*` | reconnect + frame-staleness bounds |
| `inference.max_batch_size` / `max_workers` | batch size (1 on CPU) / per-camera concurrency |
| `inference.pose_ensemble` | live-only: skip the 2nd (pose) model for throughput |
| `identity.min_evidence_obs` / `*_threshold` / `accept_margin` | online-engine gates (on-screen ids) |
| `reconcile.enabled` | run the offline reconcile + re-render on stop |
| `reconcile.sample_stride` | store every Nth fresh embedding per track |
| `reconcile.keep_frames` | keep the transient processed-frame clips |
| `topology.enabled` / `edges` | fail-open cross-camera transit-time veto (off by default) |

> The cross-camera **merge** thresholds come from `identity.reconcile.*` (shared
> with the file path), not from `live.identity.*` (which only tunes the
> provisional on-screen ids). Recalibrate `identity.reconcile.*` when the ReID
> checkpoint or camera domain changes ([section
> 6](#6-known-limitations-model-not-plumbing)).
