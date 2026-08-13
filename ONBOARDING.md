# Onboarding — Multi-Camera Person Re-Identification

**Last updated:** 2026-08-03 · **Branch:** `research` (the working branch; `main` lags)

Everything a new person — or a fresh agent session — needs before touching this
project. Read this first, then follow the reading order in section 10.

If you are an agent: **`CLAUDE.md` is the short version and it is tracked in git.**
This file is not. Read both; where they overlap, they agree by construction.

---

## 1. What the system does

Detect people across four CCTV cameras, embed their appearance, and assign a
**reid id** that stays the same for one real person **across cameras** and **across
re-appearances** (someone leaves a room and comes back).

Two input modes, one identity result:

| Mode | Command | Where it runs |
|---|---|---|
| **Live RTSP** | `python main.py --mode live` | GPU server, real time, never records the raw feed |
| **Video files** | `python main.py` (default) | CPU or GPU, recorded footage |

Both modes settle final identity with the **same offline reconcile**, which runs
after `Ctrl-C` (live) or end-of-file (batch), and re-render annotated
`output_<cam>.mp4` where one person carries one id and colour in every camera.

The pipeline owns its own Qdrant gallery. There is no separate registration step
and no external identity service.

**Pipeline:** `capture → detect (YOLO11m) → track (ByteTrack) → crop → embed
(FastReID) → Qdrant (+ recorded floor position) → live identity engine (provisional
ids) → offline reconcile (final ids, geometric reachability veto) → render`

> **State in one line (2026-08-03):** the FastReID wiring is correct and verifiable
> (`tools/preflight.py --load-model`). The **thresholds are void** because the
> feature space changed (§3.4) — the run that seemed to quantify how badly had an
> unknown headcount, so treat the *size* of that problem as unmeasured, but the bars
> genuinely do not belong to the space that is running. The deployment is an
> **office where people sit at desks**, which constrains geometry in ways §6.6
> spells out.

---

## 2. Hardware and infrastructure — read this before running anything

### 2.1 Two machines

| | Dev box | **A6000 server** |
|---|---|---|
| What | CPU-only WSL2, **no CUDA** | Everything with a camera or a GPU |
| Path | `/home/seif/Projects/Inference` | `~/seifer_work/Inference_PersonReid` |
| Used for | editing, unit tests, synthetic tests, doc work | **all model runs, all captures, all calibration** |

Synced by git:

```bash
# dev box                     # A6000
git push origin research      git pull origin research
```

`deploy.sh` rsyncs instead, and is **the only way to move gitignored files** —
model weights, and the ADR set. Qdrant runs on the server in Docker
(`docker compose up -d`, REST 6333, gRPC 6334 already mapped).

### 2.2 Why this split matters constantly

The ReID model is **FastReID ResNet101-IBN at 384×128**, which is roughly
**0.76 s/crop on CPU**. On the dev box a 12-second clip processes ~4 frames before
load-shedding drops the rest. **Any conclusion about throughput, identity quality,
or thresholds drawn on the dev box is wrong.** Use it for correctness only.

The geometry work is the exception that proves the rule: its **maths** is fully
unit-tested on the dev box (`tests/live/test_geometry_*.py`, 91 checks, seconds, no
GPU), while its **calibration** and every judgement about whether it helps require
the server.

### 2.3 Weights are not in git

`.gitignore` covers `*.pt` and `*.pth`. One exception is grandfathered in:
`src/reid/weights/osnet_ain_x1_0.pth` (53 MB) was committed before the rule and
stays tracked so fresh clones still get it.

The current checkpoint must be fetched per machine:

```bash
curl -L -o src/reid/weights/msmt_sbs_R101-ibn.pth \
  https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R101-ibn.pth
```

It is **537 MB**, over GitHub's 100 MB per-file hard limit — a push carrying it is
*rejected*, not merely slow. `yolo11m.pt` is also gitignored and auto-downloads on
first run (or `scp` it if the server has no outbound network).

### 2.4 Cameras

Four cameras, named by the last octet of their IP: **cam_206, cam_213, cam_219,
cam_224**.

- They **overlap**. cam_219 and cam_224 **share a room**. This is load-bearing
  three times over: it is why the transit-time topology veto failed, why
  cross-camera co-occurrence cannot prove two tracks are different people, and why
  cam_219+cam_224 are the **only** pair a floor frame can be fitted for (section 6).
- cam_206 and cam_213 overlap nothing, so **no geometry is derivable for them from
  imagery at all.** Not a code gap — a fact about where the cameras are.
- **The deployment is an OFFICE. People sit at desks and work.** This is
  load-bearing and was only established on 2026-08-03, after most of the design
  above. Consequences: most people are stationary for long stretches; a seated
  person's bounding box bottom is a **chair or desk edge, not their feet**; and
  tracks are long and stable but re-appearances are common (people get up, walk
  about, come back). See §6.6 for exactly what this does to geometry, and §5.7 for
  what it does not.
- **cam_206 has been missing from every run since `20260730_093723`** — dropped
  from a launch command by a shell error and never re-added. It is *also* the
  camera with the detection-recall complaint ("5 people present, 1 detected") and
  the only one whose per-camera bar has never been measured under yolo11m. **Put
  it back in the next capture.**

**Cameras are not currently configured in `config.yaml`.** `source.env_urls` is
empty and `source.videos` points at `register_file.avi`, so `--mode live` reads a
*file*. To use the cameras, create an untracked `.env`:

```bash
CAM_206=rtsp://user:pass@192.168.0.206:554/...
CAM_213=rtsp://user:pass@192.168.0.213:554/...
CAM_219=rtsp://user:pass@192.168.0.219:554/...
CAM_224=rtsp://user:pass@192.168.0.224:554/...
```

then set `source.env_urls: [CAM_206, CAM_213, CAM_219, CAM_224]`. Credentials go in
`.env`, never on the command line — a URL in `argv` is visible to every user via
`ps` and lands in shell history.

> The real URLs and path suffixes are hardware-specific and are **not recorded in
> this repo**. Ask the operator.

---

## 3. Current state (as of 2026-08-03)

### 3.1 The ReID backbone was switched, and is now accepted as-is

**From** torchreid OSNet-AIN (`osnet_ain_x1_0`, 512-d, 256×128)
**To** **FastReID SBS ResNet101-IBN, MSMT17** (`fastreid_sbs_R101_ibn`, 2048-d,
384×128).

Confirmed running on the server: `fastreid_sbs_R101_ibn (2048-d, 384x128,
post-bnneck, GeM p=2.138) on cuda:0`.

**There is no fine-tune planned.** That is a decision, not an omission: the
accuracy has to come from **geometry plus offline reconcile** instead. It is why
ADR-003B (label mining → verifier training → backbone fine-tune) is shelved.

What landed with the switch:

- `src/reid/backends.py` — a `ReIDBackend` interface. A backend owns the
  architecture, checkpoint load, **preprocessing recipe** and feature tap;
  `src/reid/extractor.py` keeps the backend-invariant contract (batching,
  `max_batch` chunking, the shared-model forward lock, L2-normalisation,
  empty-crop rejection). Swapping backbones is a `reid.model` config edit.
- `src/reid/vendor/fastreid/` — 5 torch-only files copied from upstream. FastReID
  has **no `setup.py`** and cannot be pip-installed. See its `PROVENANCE.md`.
- **No new pip dependencies.**

Two facts that surprise people, both read off upstream source:

1. **Input is 384×128, not 256×128.** `Base-SBS.yml` overrides `INPUT.SIZE_TEST`.
   Feeding 256×128 runs fine and quietly costs accuracy.
2. **There is no feature tap to choose.** `EmbeddingHead.forward` returns the
   post-bnneck feature unconditionally at eval; `NECK_FEAT` affects training only.
   The backend **raises** on any tap, so `reid.tap: n/a` is required.

### 3.2 Geometry landed as a reachability veto — built, tested, shipped OFF

New this change. Full rationale in **`ADR-003D`**; the rules that must survive are
duplicated in `CLAUDE.md` §1–2 and the `geometry:` block of `config.yaml`.

The one formula: `required_speed = distance / elapsed`, vetoed above a speed
ceiling. At `elapsed ≈ 0` that is *one body cannot be in two places*; at
`elapsed > 0` it is *could not have got there in time*.

```
src/geometry/calibration.py   the floor-frame record + the metric-scale guard
src/geometry/floor.py         bbox -> point on a shared floor (owns the homography)
src/geometry/reachability.py  two recorded points -> possible / impossible
src/geometry/recorder.py      the LIVE run's writer -- the only place a position
                              is ever computed
tools/fit_floor_frame.py      fits the floor frame from people's own foot points
```

**Three invariants, all enforced by tests, not just documented:**

1. **The live run records geometry. Offline reconcile only consumes it.** Positions
   are computed once, at capture, and stored in the observation payload under
   `floor`. Reconcile never loads a calibration, applies a homography, or derives a
   position from a box — it may import `geometry.reachability` (pure arithmetic) and
   nothing else under `geometry/`.
   *Why, since reconcile already has the `bbox`:* the live feed is never recorded,
   so an unwritten position is gone; and re-fitting a calibration is cheap and
   therefore likely, so a reconcile that derived positions would silently return
   **different identities** for a finished run, with nothing saying why.
2. **Units are floor units, not metres** — see 3.3.
3. **It fails open, always.** Uncalibrated camera, missing box, mismatched image
   size, different floor groups, no timestamp, unmeasured ceiling → *unavailable*,
   treated as no opinion.

Both switches default off: `geometry.enabled` (recording) and
`geometry.reconcile.enabled` (the veto). Turning recording on is additive and safe;
only the veto changes identities.

`src/live/topology.py`'s hand-set min-transit veto is **superseded** and stays
disabled. It failed because it asserted 2–3 s minimums between adjacent cameras and
pruned the true match. Measured positions cannot make that mistake — overlapping
cameras give a distance near zero, so the check is silent exactly where the guess
over-fired.

### 3.3 There is no metric reference, so nothing claims metres

> Metric geometry cannot be established from monocular cameras alone. A single
> trusted metric reference must be provided before any metre-based threshold
> (0.5 m, 3.0 m, walking speed in m/s) is considered valid.

The requirement is **one trustworthy source of metric scale** — *not* a tape
measure, and *not* a survey. In order of preference: verified floor plans or CAD;
known architectural dimensions (corridor/door width, column spacing, tile pitch)
verified on site; one or more independently measured reference distances.

**None exists for this deployment.** Consequences:

- `CalibrationRecord.is_metric` is False; the metre-facing API raises
  `MetricScaleUnavailable`, whose message says exactly what to supply.
- Every metre-denominated number in **ADR-003, 003A, 003B and 003C is void as
  written** — the 0.5 m / 3.0 m mining bands above all.
- **Nothing is lost for the veto.** The speed ceiling is measured from the run's own
  same-track motion, in the same unit as the distances, so the unknown scale
  cancels exactly. Ceiling = p99.9 of observed speed × `safety_factor`.
- Metres are needed for exactly one thing: relating **separate floor groups**, i.e.
  bringing cam_206 and cam_213 in. `group_distances` is where a real-world distance
  would go, and it ships empty.

### 3.4 Thresholds are void — and this is now the project's biggest problem

**This is the bottleneck. Not geometry, not the backbone.** Every bar in
`config.yaml` was derived in OSNet-AIN's 512-d post-ReLU space and none has been
re-anchored to FastReID's 2048-d post-bnneck space.

> ⚠️ **CORRECTION, 2026-08-03.** An earlier version of this section claimed run
> `20260803_121136` "turned two people into 21 identities" and called that ~10×
> over-fragmentation. **That was wrong.** Two people were visible in *both* cameras;
> there were **more people around**, and the true headcount was never established.
> Every count-based conclusion below is therefore **unproven**, and the corrected
> reading is in §3.4a. The threshold suspicion is still reasonable — it is no longer
> *measured*.

Run `20260803_121136` (70 s, cam_219 + cam_224, unknown headcount) produced:

| observed | what it does NOT prove |
|---|---|
| 29 tracklets → **21 identities** | nothing, without the headcount. 21 ids is bad for 3 people and roughly right for 12 |
| `ABSOLUTE_THRESHOLD` failed **76 of 96** decisions | a refusal is **correct** when the pair is two different people. With several people present, most pairs *should* fail |
| `BELOW_ABSOLUTE_THRESHOLD` excluded **834** candidates | same — with N people, most of the N² candidate pairs are genuinely different people |
| best **rejected** same-camera score **0.671** vs a **0.90** bar | only evidence if that pair was truly one person. Unknown |
| cross-camera merges at 0.937, 0.706, 0.670 | that three merges happened, not that they were right |

**The lesson is §5.2's, and it caught us again:** a cluster count cannot tell you
whether a cluster is one person or three. It also cannot tell you whether 21 is too
many, unless you know how many people walked past.

Independent evidence that thresholds *are* mis-set — headcount-free, so it survives:

- at `same_camera_threshold: 0.90` only **50%** of same-person fragments merge; 100%
  merge at 0.75 (measured on labelled fragments, not inferred from a count);
- `compare_backbones.py` measured FastReID's **different-person p95 at 0.40** versus
  OSNet's **0.59**. The shipped bars were positioned for OSNet's stranger ceiling, so
  there is headroom in the new space. *(Margins are not comparable across feature
  spaces for judging which model is better; this number is being used only to place a
  bar inside FastReID's own space.)*

### 3.4b Why nothing had been recalibrated: the measurement script was broken

Found 2026-08-03, and it is the direct answer to "have you recalibrated for
FastReID?" — **no, and one of the two tools needed to do it could not run at all.**

`tests/calibration/measure_score_separation.py` — which `tests/calibration/README.md`
calls *"the measurement that sets every identity threshold in the system"* — died on
line 71 under FastReID:

```
AttributeError: 'Sequential' object has no attribute 'global_avgpool'
```

It hardcoded torchreid OSNet internals (`model.global_avgpool`, `model.featuremaps`,
`model.fc[0..2]`) because its original subject was comparing OSNet's post-ReLU tap
against post-BN. FastReID has no such block and **no selectable tap**, so the
surgery raised before anything was measured.

So from the backbone switch until 2026-08-03 there was **no way to derive a threshold
for the backbone actually running.** That is why the bars are still OSNet's. Fixed by
discovering the tap axis from the backend instead of assuming it: OSNet still gets
both taps, anything else gets one — `production` — and every other section of the
script is backend-agnostic and runs unchanged.

**The value of this script is that it needs NO operator labels.** Its
different-person pairs are *proven* by co-occurrence in a single frame (one body
cannot be two simultaneous detections), so it sidesteps the ground-truth problem in
§3.4a entirely — for the same-camera axis.

Its limits, from its own closing section:

- **same-camera only.** Cross-camera same-person scores run lower, so
  `cross_camera_threshold` cannot be calibrated from a single-camera clip.
- `other MAX` is an extreme-value statistic that **grows with sample size** — prefer
  p95.
- small samples are hypotheses. Read the `SAMPLE:` footnote it prints.

### 3.4a What is actually needed before any threshold is changed

**Ground truth for one run.** Nothing in §8.4's sweep means anything without it — the
sweep ranks settings by identity count, and a count is only judgeable against a known
answer. Two ways, cheapest first:

1. **Watch `output_cam_219.mp4` / `output_cam_224.mp4`** from a run and answer the two
   §5.3 questions: does any one person carry more than one id? does any one id cover
   more than one person? Record verdicts with `review_links.py <run_id> --label` so
   they accumulate — there are still only **11** labels in the project.
2. **Capture a run with a known cast.** Agree the headcount before recording, keep
   others out of frame, and write it down. That single number turns every future
   sweep on that run into a real measurement.

Until then the honest statement is: **the thresholds are void because the feature
space changed, not because a count proved them wrong.**

Second, unrelated: **the Qdrant collection must be rebuilt** on any machine still
holding 512-d vectors. `python tools/preflight.py` reports it and `--fix-store`
rebuilds it.

### 3.5 Known trap while the collection rebuild is outstanding

`IdentityStage` **swallows** the store's dimension `ValueError`. Running against a
stale 512-d collection therefore **silently drops observations** rather than
failing loudly. Still unfixed. It is the most likely way a first run after the
switch goes wrong.

### 3.6 Dead code removed in this change

All git-tracked, so recoverable; none was on the product path.

- `tests/calibration/click_covisible_points.py` — the interactive point-clicking UI.
  Superseded: the floor frame is now fitted from people's own foot points.
- `tests/calibration/measure_covisible_geometry.py` — its pixel-space consumer. Its
  one good idea, the **label-free triangle test**, is reimplemented inside
  `tools/fit_floor_frame.py` as the fit's validation gate.
- `src/crop_saver.py` + its `main.py` wiring + the `crops:` config block — wrote
  nothing to disk, yet was constructed and called every frame with the result
  discarded.
- Five demo/validation scripts that read `crops/<cam>/id_XXXX` folders nothing
  writes any more, so none could run: `src/reid/{demo_detect_reid,demo_track_reid,
  test_reid}.py`, `src/identity/demo_identity.py`,
  `src/database/demo_store_search.py`.
- `src/identity/diagnose.py` — superseded by `explain_merge_failure.py`, which
  reconstructs the actual refusal at production settings.
- `tests/calibration/measure_pose_ensemble.py` — measured a batch path that no
  longer runs.

> Reviving ADR-003B's fine-tune later means re-adding a real on-disk crop writer.
> Do not resurrect `CropSaver` — it never wrote anything.

### 3.7 Last runs on the server (2026-07-31)

| run_id | obs | Source | Result |
|---|---|---|---|
| `20260731_134913` | 485 | `register_file.avi` (**not** cameras) | 10 tracklets → 9 identities; one same-camera merge at cosine 0.931; cross-camera people 0 |
| `20260731_135020` | 61 | same file via `--mode live` | ended at EOF on its own; never printed a `run_id` |

**Whether those 9 identities are correct is unknown.** Nobody has watched
`output_register_file.mp4`. Do not infer quality from the cluster count — section 5.

---

## 3.8 SESSION LOG — everything that changed on 2026-08-03/04

Written for chat migration: if you are a fresh session, this is what the previous one
did and what it concluded. Nothing here is a plan; it all landed.

### Code added

| path | what |
|---|---|
| `src/geometry/calibration.py` | floor-frame record + the metric-scale guard (`MetricScaleUnavailable`) |
| `src/geometry/floor.py` | bbox → point on a shared floor; owns the homography and every refusal |
| `src/geometry/reachability.py` | `required_speed = distance / elapsed` vs a measured ceiling. Imports nothing else from `geometry/` — that is what lets reconcile use it |
| `src/geometry/recorder.py` | the LIVE run's writer, the only place a position is ever computed |
| `src/quiet.py` | `suppressed_native_stderr()` — silences the codec probe's C++ stderr |
| `tools/fit_floor_frame.py` | fits the floor frame from people's own foot points on a finished run |
| `tools/backfill_geometry.py` | applies a fresh calibration to a run captured before it existed, so ONE capture suffices |
| `tools/preflight.py` | verifies the configured stack; exits non-zero on silent-garbage conditions |

### Code changed

- **`src/identity/reconcile.py`** — consumes recorded geometry as a hard
  `GEOMETRIC_UNREACHABLE` veto. Added `resolve_geometry_policy`,
  `build_speed_envelope`, `closest_pairings`, `reachability_verdict`; `_gather_tracklets`
  now collects the `floor` rows; `conflict_reason` checks geometry first and returns
  the *actual* blocking gate (it used to hardcode the same-camera one in Phase 1).
- **Media time** (`src/live/frame.py`, `capture.py`, `decode_backend.py`,
  `identity_stage.py`, `render.py`) — `Frame.source_ts` / `event_ts()`. A file's `ts`
  tracks decode speed, so two files read in parallel got timestamps differing by
  *thread scheduling*; every cross-camera temporal rule would have been built on
  invented simultaneity. Also `live.capture.file_time_offsets`.
- **The annotations sidecar now records `frame_ts`** per frame. Its absence is why the
  wiped runs' surviving clips cannot be replayed for geometry.
- **`tests/calibration/measure_score_separation.py`** — see §3.4b. It could not run on
  FastReID at all.
- **`_common.py`** — `--geometry` / `--no-geometry` / `--geometry-safety` on all four
  reconcile-replaying tools; **`explain_merge_failure.py`** taught the new gate, or it
  would report "no rule refused" about a geometrically vetoed pair.

### Code removed (all git-tracked, recoverable)

`click_covisible_points.py`, `measure_covisible_geometry.py`, `crop_saver.py` + its
`main.py` wiring + the `crops:` config block, five demo scripts that read `crops/`
folders nothing writes, `diagnose.py`, `measure_pose_ensemble.py`.

### Tests: 19 files, all passing

New: `test_geometry_reachability.py` (30 checks), `test_geometry_floor_frame.py` (37),
`test_geometry_not_recomputed.py` (24 — the AST invariant), `test_media_time.py` (17).

### Two bugs the tests caught in the new code

1. Same-timestamp pairs returned `UNAVAILABLE` — silently discarding the *strongest*
   signal the check has.
2. `position_error_units` must be **≥** the real cross-camera disagreement for one
   person, or the same-instant rule vetoes every true co-visible match. The fit now
   derives it from the held-out reprojection p95, which *is* that disagreement.

### What was retracted

The claim "run `20260803_121136` turned two people into 21 identities". The headcount
was never established — two were co-visible, others were around. See §3.4. Every
count-based conclusion from that run is unproven; what survives is label-free.

### What is NOT done

- **No threshold value has been changed.** The measurement is recorded beside
  `identity.reconcile.same_camera_threshold` in `config.yaml`; the value is still 0.90.
- Geometry has **never run on real data** — no calibration record exists. Both switches
  ship `false`.
- The floor-frame fit has **failed twice** on degenerate footage (§6.3).
- `cross_camera_threshold` is un-recalibrated and cannot be done from a single-camera
  clip.
- **Commits are local only.** No git credentials in the previous session's environment;
  `./deploy.sh` (from the dev box) or `git push` is a human action.

### The one number that matters most

At `same_camera_threshold: 0.90`, **50% of genuine same-person fragments do not
merge**, and any bar in **(0.434, 0.810)** is perfect on the sample. The bar sits above
that window. Evidence and the recommended 0.80 are in `config.yaml`.

---

## 4. The document map

| Document | What it is | Read when |
|---|---|---|
| **`CLAUDE.md`** | the rules that outrank instinct, in one page. **Tracked in git** | first, always |
| `README.md` | user-facing: what it does, how to run it | first |
| **`ARCHITECTURE.md`** | full data flow, every component (incl. `src/geometry/`), concurrency, **§6 known limitations** | before changing any code |
| **`REMEDIATION_PLAN.md`** | the working plan. **Part A** = what NOT to retry (with evidence). **Part J** = field results. **Part H** = every measurement and its caveats. **§0.75** = machine + corpus | before proposing any change |
| **`ADR-003D`** | **the geometry decision of record.** Reachability veto, floor units, the three invariants, the gates | before touching `src/geometry/` or the veto |
| `ADR-003`, `003A`, `003B`, `003C` | the earlier geometry design. **Superseded in part** — every metre-based number is void (3.3), 003B is shelved, 003C's scoring shape is dropped. Read for rationale, not for values | background |
| `src/identity/DESIGN.md` | identity subsystem design | working on reconcile/verifier |
| `tests/calibration/README.md` | what each calibration script decides and what its numbers cannot | before trusting any number |
| `NVIDIA_DeepStream_Adoption_Plan.md` | archive of the v1→v4 iterations. Read once; do not maintain | never, in practice |

**The ADR set and the DeepStream plan are gitignored**, so they are *not* on the
A6000 and *not* in a fresh clone. That is why the load-bearing rules are duplicated
into `CLAUDE.md`, `ARCHITECTURE.md` and `config.yaml` comments, all of which travel.

**Reading order for a fresh session:** `CLAUDE.md` → `REMEDIATION_PLAN.md` §0 →
**Part A** → **Part J** → `ADR-003D` → `ARCHITECTURE.md` §§3, 5, 6.

---

## 5. Hard-won lessons — the expensive ones

These are not style preferences. Each cost real time.

### 5.1 Threshold tuning is not the lever

**Five** changes have been reverted for *hurting* accuracy: `identity.threshold` →
0.68, verifier `accept_threshold` → 0.62/0.08, `max_occlusion_ratio` → 0.35, and the
topology min-transit veto.

Settled by two live runs plus an offline sweep: cam_224 at 0.80 **fused several
people into one reid**; at 0.90 **one person shattered into many**. Both directions
wrong means the number is not the variable. The cause is that reconcile compares
prototype *means*, which score one person's front-vs-back fragments **below** two
strangers in similar clothing (0.640 vs 0.800 on the counterexample). Fix the
scoring or add a physical guard; tune bars only afterwards.

The reachability veto is that physical guard. It is deliberately **not** a threshold
you tune toward accuracy — its one knob, `safety_factor`, should be **raised when in
doubt, never lowered** (5.4).

### 5.2 A cluster count cannot tell you whether a cluster is one person or three

Every number quoted for a threshold change on 2026-07-30 was **equally consistent
with the good and the bad outcome**, and two settings chosen that way both made the
videos worse. Rank candidates with the sweep, then **render and watch** before
shipping.

### 5.3 Ground truth is the operator watching video

Symptoms are reported by watching, not by metrics. Verbatim, because this is what
"working" has to mean:

- *"reid 2 correct by his front, becomes reid 7 by his back"* (cam_213)
- *"reid 3 becomes reid 7 when the person moves at a bad angle"* (cam_224)
- *"reid 1 becomes reid 6 when he leaves the cam and comes back the other side of
  the room"* (cam_219, same camera)
- *"multiple other people in the room are also called reid 6"* — a false merge
- *"5 people present but only 1 detected at first"* (cam_206)

Labels belong in `calibration/link_labels.jsonl` (via `review_links.py --label`),
**not in a conversation**, so they accumulate across runs with provenance. There are
currently **11** usable labels (8 same, 3 different, one inferred rather than
stated) — far too few to finely rank two good models.

### 5.4 False merges are worse than false splits — except for vetoes

A merge fuses two people's histories and corrupts both. A split just gives one
person a spare id, which reconciliation can still fix. When uncertain, mint a new
identity.

**A veto inverts this,** and getting it backwards is how the topology veto destroyed
a run (`linked` 5 → 1, `topology_pruned=508`). Refusing a merge is *unrecoverable* —
reconcile cannot un-split a person. So every geometric error budget biases toward
permitting the merge: distance is **shrunk** by the position error radii, elapsed
time is **grown** by the clock budget, the ceiling is **raised** by `safety_factor`.
A sloppier calibration therefore yields **fewer** vetoes, never wrong ones.

### 5.5 Methodological rules for any measurement

1. **A "different person" pair requires co-occurrence in the same frame — within
   one camera.** One person cannot be two simultaneous detections. **Across
   cameras this proves nothing here**, because the cameras overlap. (This same fact
   is what makes the calibration's triangle validation label-free — 6.3.)
2. **Bank queries must be held out of the bank**, or a query matches itself at
   ~1.0 and inflates the same-person side.
3. **`other MAX` grows with sample size.** Prefer p95. On the same clip, 48 vs 90
   frames moved raw MAX from 0.819 to 0.936. The speed ceiling uses p99.9 for
   exactly this reason.

### 5.6 Shell traps that have already caught someone

- **Placeholders get pasted literally.** `<run_id>` and `<chosen>` have both been
  run verbatim; `<` is a shell redirect. Write commands that derive their values.
- **`python main.py --reset` is not a wipe** — it clears the store *and then runs
  the whole pipeline*.
- **`output_cam_*.mp4` existing does not mean the last run succeeded.** They are
  only overwritten by a completed render, so stale files look like success. Check
  mtime against the `run_id`.
- **`run_id=` is only printed when reconcile is enabled**, so a run that dies early
  leaves `RUN` empty and every downstream command fails confusingly.

### 5.7 Sitting is not a problem for ReID generally — only for foot-point geometry

Worth stating because §6.6 reads alarming. In an office, seated people are the
*easy* case for the appearance model: stable pose, stable lighting, consistent
illumination, and long uninterrupted tracks. The hard cases are transitions and
re-appearances, which is exactly what the offline reconcile exists for.

Two things already handle the common office failure with no geometry involved:

- two people visible **simultaneously in one camera** can never be merged — the
  same-camera temporal-overlap veto is a hard rule and is on today;
- reconcile rebuilds identity from scratch over the whole run, so a person who sits,
  leaves and returns can be re-joined after the fact in a way the live engine
  structurally cannot.

What sitting specifically breaks is the assumption that **the bottom of a bounding box
is a point on the floor**. That assumption belongs to `src/geometry/` alone. Nothing
else in the pipeline makes it.

---

## 6. The geometry workstream — how to actually use it

**Status: implemented, shipped disabled, never yet run on real footage.**

### 6.1 What it can and cannot cover

| Cameras | Geometry | Consequence |
|---|---|---|
| cam_219 + cam_224 (same room) | **yes** — one floor group | full reachability check |
| cam_213, cam_206 | **none derivable** | appearance-only, always fail-open |

A floor frame is fitted from people's foot points seen from **both** cameras, so it
needs overlap. Cross-group pairs report *unavailable*, **not** "far apart" — a naive
implementation would read the incomparable coordinates, find them 500 units apart,
and veto every cross-room merge in the deployment.
`tests/live/test_geometry_not_recomputed.py` §6 pins that it does not.

### 6.2 Calibrate once, from a finished run, with the cameras off

```bash
python tools/fit_floor_frame.py <run_id>              # writes calibration/floor_frame.json
python tools/fit_floor_frame.py <run_id> --dry-run    # report only
python tools/backfill_geometry.py <run_id>            # give THAT run positions too
```

**ONE capture, not two.** A floor frame is fitted *from* a run, so the first run on
any deployment is necessarily captured before a calibration exists — which would
seem to need a second capture to record positions with. It does not:
`backfill_geometry.py` applies the fresh calibration to the `bbox` + `ts` already in
that run's payloads, so the run it was fitted from ends up carrying positions too.
No frames, no detection, no camera time.

Backfilling the same run the fit came from is mildly self-serving — the fit saw some
of those feet — so it is for checking the plumbing and **watching whether the veto's
decisions are right**, not for claiming generalisation. The first honest
generalisation test is the next capture, which costs nothing extra because
`geometry.enabled: true` records positions live by then.

`bbox` and `ts` have been in every observation payload since the live reconcile
landed, so **any finished multi-camera run can be calibrated retroactively.** No
clicking, no floor plan, no camera time, no per-run calibration: the record is keyed
by camera name and pixel size, so every later run on those cameras reuses it. Re-fit
only when a camera is physically moved or `source.resize_width` changes.

Record a metric reference later, if one ever exists:

```bash
python tools/fit_floor_frame.py <run_id> \
  --metric-reference "floor_plan:corridor width, cam_219 doorframe to far wall:3.42:2.10"
```

### 6.3 WHAT A CALIBRATION RECORDING MUST CONTAIN — and the first attempt failed

**Read this before recording anything.** The first real attempt
(run `20260803_121136`) failed, and it failed on requirements that look like
nitpicks and are not:

```
1 confident match: cam_219:10 <-> cam_224:2, overlap 6.3s -> 16 foot pairs
RANSAC: 6/16 inliers (38%)          [needs 55%]
held-out error: median 203.7 px     [~3 people wide in cam_219]
```

Diagnosis: **16 points along one person's 6-second walk are nearly collinear.** A
homography maps a *plane* to a plane and cannot be recovered from a line.
`cv2.findHomography` still returns a matrix — it is simply meaningless away from that
line, which is what 38% inliers and a 203 px error look like.

A second attempt at `--min-cosine 0.65` got *worse* (8% inliers) because it admitted
a **stationary** person: 106 of 122 correspondences came from one tracklet that never
moved. Those 106 are mutually consistent (they are the same point), so RANSAC scored
them as a model and rejected everything else — while producing a flattering 9.7 px
median that described nothing but the blob.

**The requirements, in order of how often they are the reason it fails:**

| requirement | why |
|---|---|
| **8–12 DISTINCT floor locations** | this, not the correspondence count, determines the plane. The tool now reports distinct locations and refuses below 8 |
| **Stand still 3–4 s at each spot** | observations are sampled every ~0.5 s, so a pause plants several points at one location |
| **Spread over the room's extent** | corners of the overlap, the middle, near each camera. A loop or a straight line is degenerate however long it is |
| **NOBODY SEATED** | the foot point is the box bottom, so a chair or desk edge encodes a plane that does not exist — and those points are self-consistent enough for RANSAC to *prefer* them |
| **Feet visible** | behind a desk is the same failure as seated |
| **≥1 moment with two people in one camera while a third track runs in the other** | required by the validation (§6.3); one person alone fits a homography and cannot verify it |

**One person is enough for the fit, and is actually ideal** — no matching ambiguity.
More people matter only for the validation. The mistake is thinking this needs a
crowd; it needs *coverage*.

The tool now reports arrangement before it reports any residual:

```
arrangement of the 19 point(s) in the target frame:
   span 900 x 660 px  (35% x 46% of the frame)
   collinearity 0.733   (0 = all on one line, 1 = spread evenly in both directions)
```

Below 0.15 it refuses and says the fix is footage, not a flag.

### 6.4 The fit is graded by people it is *proven* must be apart

Fitting from appearance matches to constrain appearance is circular. The **grading**
is not: two tracklets co-occurring in ONE camera are provably two people, so a
tracklet in cam_219 must land near **at most one** of them. The tool refuses to write
a record that fails ≥20% of those triangles, or that has fewer than 3, or whose
RANSAC inlier fraction is under 55%.

Read the triangle pass rate and the held-out RMS before trusting anything
downstream — the held-out p95 becomes `position_error_units`, which is what protects
every true co-visible match from the same-instant rule.

### 6.5 Then, in order

```bash
# 1. RECORD (additive, safe). Set geometry.enabled: true, then capture. For a run
#    captured BEFORE the calibration existed, use tools/backfill_geometry.py (6.2)
#    -- but note positions can never be recovered for a run that stored no bbox/ts.
python main.py --mode live

# 2. Check the run summary named a high positioned fraction PER CAMERA.
#    A camera at 0% means its image_size does not match the frames.

# 3. Try the veto offline, on that finished run -- no cameras, no re-capture.
python tests/calibration/sweep_reconcile_thresholds.py <run_id> --geometry
python tests/calibration/rerender_from_clips.py <run_id> --geometry   # WATCH IT

# 4. More permissive if it over-fires. Never less.
python tests/calibration/rerender_from_clips.py <run_id> --geometry --geometry-safety 3.0
```

Gates, from `ADR-003D` §7: **G0** calibration validates · **G1** recording works
per camera · **G2** judged on video, not on a count · **G3** the cross-camera
identity count must not drop — that is the exact number the topology veto destroyed.

If G2/G3 fail: raise `safety_factor`, or set `geometry.reconcile.enabled: false` and
**keep recording**. Recording cannot hurt; only the veto changes identities.

### 6.6 SITTING: what it does to geometry at runtime

The deployment is an office (§2.4), so this is not an edge case. A seated person's
recorded position is **wrong** — displaced by however far the chair is from their
feet, projected onto the floor, which at a shallow camera angle can be metres.

But wrong is not the same as useless, because the error is **stable and tied to
where they are sitting**:

| what is being compared | position error | veto behaviour |
|---|---|---|
| two colleagues at **different desks** | both wrong, but *differently* | **correctly refuses** the merge — this is the win |
| same person, same desk, two tracklets | both wrong *identically* | distance ≈ 0 → correctly allows |
| **same person, seated then standing** | displaced by the seat height | ⚠️ **risk of a false veto** |
| people walking about | correct | works as designed |

Row 1 is the operator's actual complaint (*"multiple other people in the room are
also called reid 6"*), and it still works: **the map does not have to be truthful to
be discriminative.** Two people at two desks reliably land in two different wrong
places.

Row 3 is the genuine risk. Someone stands up, their apparent position jumps, and over
a couple of seconds that reads as impossible speed → false veto → an unrecoverable
split. Two honest caveats:

- **The median-over-pairings rule does NOT protect against this.** It defends against
  *occasional* bad points. Someone seated for a whole tracklet has every pairing
  wrong in the same direction, so the median is wrong too.
- **It partly self-corrects, by accident.** The speed ceiling is measured from your
  own footage *including* those stand-up jumps, so they inflate p99.9, the envelope
  widens, and fewer vetoes fire. It fails in the safe direction — but it also makes
  the check weaker.

**Nothing currently detects "this person is not standing."** `clipped` catches frame
edges only. The proposed fix (not built) is a per-camera regression of box height
against foot position — a standing person's height is a predictable function of where
their feet are, so a much shorter box means seated, crouching or occluded. That would
let seated observations fail open *on purpose* rather than by luck, and would exclude
them from calibration automatically.

**Honest verdict for an office.** The veto is **less powerful here than in a
corridor**, and that is worth knowing before switching it on. Its power comes from
"close in time, far apart" — but most office false merges are *time-separated*
(a colleague at the same desk an hour later), where `distance / elapsed` is tiny and
geometry is silent. What it does still catch, and nothing else does, is two people
seen by cam_219 and cam_224 **at the same moment** standing apart that appearance
scores 0.9 — covisibility deliberately never vetoes that pair, because one person
legitimately can be in both. Real, but narrow.

### 6.7 Running on recorded footage — and the timestamp trap

`--mode live` on files. The file-batch path records **no** geometry —
`IdentityService._commit` stores neither `bbox` nor `ts`, so there is nothing to
position. The live pipeline records both for files exactly as it does for RTSP.

**But `frame.ts` alone is not usable for recorded footage.** It is stamped at frame
*read*, and files decode as fast as the disk allows (125+ fps measured here), so it
tracks decode progress, not events. Two files read in parallel get timestamps whose
difference reflects **thread scheduling** — and every cross-camera temporal rule
(the co-presence veto, all of geometry's co-temporal pairing) would be built on
invented simultaneity that looks entirely plausible.

So since 2026-08-03 a file source also carries **media time**,
`source_ts = offset + frame_index / fps`, and `Frame.event_ts()` is what every
"when did this happen" consumer reads — including the stored payload's `ts`. The
pipeline's own machinery (scheduler freshness, writer pacing, the live engine's
TTLs) deliberately stays on the wall clock; `tests/live/test_media_time.py` pins
both halves. Capture prints which clock is in force per camera:

```
[capture:cam_219] recorded source: MEDIA time active (22.913 fps, offset +0.000s).
```

If it instead reports the rate as **UNKNOWN**, that camera has no media time and
nothing cross-camera from the run can be trusted — re-encode the file with a valid
fps.

**Files must overlap in real time.** Media time makes two recordings *comparable*;
it cannot create overlap that was never filmed. Concurrent recordings started
together need no configuration (0 offset = "all files begin at t=0"). If one started
late, set `live.capture.file_time_offsets: {cam_224: 1.35}`. Verify rather than
assume: a wrong offset shows up as a held-out reprojection error in
`fit_floor_frame.py` that no calibration can reduce, because you are pairing one
person's feet with where they were a second earlier.

---

## 7. The corpus — what already exists

> ## ⚠️ THE QDRANT COLLECTION WAS FOUND EMPTY ON 2026-08-03
>
> **Every run in the table below is gone from the store.** The store only *warns* on
> a dimension mismatch and never wipes, so the cause is something that emptied it
> deliberately — most likely `python main.py --reset`, which clears the store **and
> then runs the pipeline**, so it looks like an ordinary run.
>
> Consequences, in order of how much they cost:
>
> - **The 11 operator link labels in `calibration/link_labels.jsonl` are orphaned.**
>   They name `(run_id, camera:track_id)` pairs, so `review_links.py --score` has
>   nothing to grade against. That was the project's only ground truth.
> - **A floor frame cannot be fitted** until a run with cam_219 + cam_224 exists
>   again (§6.2).
> - `sweep_reconcile_thresholds.py` / `rerender_from_clips.py` / `explain_merge_failure.py`
>   all read stored observations, so none of them can run either.
>
> **Partially recoverable, if the clips survived on disk.** `keep_frames: true` kept
> `._live_src_<cam>.mp4` + `.annotations.json` per camera, and the sidecar preserves
> every box **and its `track_id`** — so re-embedding the clips would rebuild the same
> tracklets and make the labels usable again. Check with
> `ls -la ._live_src_*` on the server.
>
> **But NOT for geometry.** Sidecars written before 2026-08-03 carry no per-frame
> wall clock — only a single `measured_fps` — and that gives each camera its own
> timeline with an *unknown offset* relative to the others. Co-temporal pairing is
> the foundation of both the floor-frame fit and the veto, so geometry cannot be
> replayed from an old clip. `render.py` now persists `frame_ts` per frame precisely
> so this cannot happen again.

Runs that **were** on the server's Qdrant, kept here because the clips may still
exist and because they document what the corpus contained:

| run_id | Observations | Cameras | Notes |
|---|---|---|---|
| `20260730_093723` | 4209 | 4 | the Part J analysis. No clips (predates the sidecar) |
| `20260730_111232` | 5048 | 213/219/224 | **best corpus.** Clips + sidecars kept. Operator confirmed reid 5, 6 and 10 each held more than one person |
| `20260730_120551` | 1482 | 213/219/224 | ~2 min. Clips + sidecars kept |
| `20260731_060425` | — | 4 | the run all 11 operator labels came from |

**The one run that DOES exist (captured 2026-08-03, after the wipe):**

| run_id | Observations | Cameras | Notes |
|---|---|---|---|
| `20260803_121136` | 1480 (cam_219 604, cam_224 876) | 219 + 224 | 70 s. Two people co-visible in both cameras, **plus others around — true headcount UNKNOWN**. Clips + sidecars kept. 29 tracklets → 21 identities, which cannot be judged without that headcount (§3.4). **Too degenerate to fit a floor frame** (§6.3). Usable for the sweep only once someone watches the video and labels it (§3.4a) |

**These hold OSNet 512-d vectors.** They cannot be swept for FastReID appearance
thresholds — the sweep reads embeddings out of Qdrant.

**They held OSNet 512-d vectors, and they are gone regardless** — see the notice
above. Had they survived, `tools/fit_floor_frame.py` would have worked on them
despite the backbone switch, because it uses *relative* appearance similarity to
pick correspondences and never compares a cosine to an absolute bar.

### 7.1 Why the frozen clips are valuable

`src/live/render.py::_capture` writes the **clean** frame (no boxes drawn) to
`._live_src_<cam>.mp4`, plus an `.annotations.json` sidecar holding every box and
`track_id`. Clip + sidecar + stored embeddings are a **complete record**: any
reconcile setting — or any *backbone* — can be replayed offline with **zero camera
time**.

Geometry adds a second sidecar, `logs/geometry_<run_id>.jsonl`, which keeps the
**image-space** foot point. That makes re-deriving positions under a better
calibration a matrix multiply over a text file rather than a re-detection. It is
**analysis input only** — feeding it back into reconcile would break invariant 1.

### 7.2 Test footage in the repo

| Clip | Frames | H.265 reference errors |
|---|---|---|
| `register_file.avi` | 1428 | **0** — use this for anything appearance-related |
| `test_v2.avi` | 1573 | 207 (13%) |
| `test_file.avi` | 682 | 294 (43%) |

Measuring feature quality on a corrupted clip measures the corruption.

### 7.3 Known limits of the sample

Six people, one camera, one clip, 14 proven-distinct pairs. **No cross-camera data
in the repo footage at all**, so nothing on the dev box can fit a floor frame or
exercise the veto on real footage. A prior 3-camera run reconciled 35 tracklets into
7 identities — 5× fragmentation — and **none of the available clips reproduce that**.

---

## 8. Running things

### 8.1 Is this machine configured correctly? Run this first

```bash
python tools/preflight.py                 # seconds, loads no models
python tools/preflight.py --load-model    # also builds the net and MEASURES it
python tools/preflight.py --fix-store     # DESTRUCTIVE: rebuild a wrong-width collection
```

Verifies the whole configured stack and exits non-zero if a run would **silently**
produce garbage: `reid.model` vs `reid.weights` vs the measured embedding width, the
Qdrant collection's width and metric, whether an RTSP run will write a watchable
video with reconciled ids, the RTSP transport/timeouts, and which `device` key each
path actually reads (`reid.device` is file-batch only; the live path reads
`live.run.device`).

The check that matters most is the **collection width**. A collection left at the
previous backbone's 512-d accepts nothing at 2048-d, the store only *warns*, and
`IdentityStage` swallows the error — so the run persists nothing and reconcile has
nothing to reconcile, with no failure anywhere. That is the most likely way a first
run after the backbone switch goes wrong (§3.5).

### 8.2 Correctness checks (fast, dev box)

```bash
python tests/run_all.py                                   # 19 test files
python tests/calibration/verify_embedding_contract.py     # the one real regression test
```

`verify_embedding_contract.py` is the only asserting script — run it after **any**
change under `src/reid/`. Everything it covers is a *silent* failure mode: a broken
BGR→RGB swap, a transposed resize, a batch/crop misalignment. No error, no crash,
just worse matching.

The three geometry test files are pure logic — no GPU, no footage, no Qdrant, no
calibration file — so the maths is fully verifiable here:

```bash
python tests/live/test_geometry_reachability.py     # the envelope + the metric guard
python tests/live/test_geometry_floor_frame.py      # bbox -> floor, and every refusal
python tests/live/test_geometry_not_recomputed.py   # the invariant, via the AST
```

### 8.3 Comparing backbones

```bash
# single camera, from a video
python tests/calibration/compare_backbones.py register_file.avi 60 6

# multi-camera, from a frozen run's clips — the one that matters
python tests/calibration/compare_backbones.py --clips <run-dir> --device cuda
```

Uses **threshold-free** metrics (ROC AUC, R@1) because a cosine bar means nothing
across feature spaces. It **refuses to name a winner** when the footage saturates —
which `register_file.avi` does: FastReID R101 and OSNet-AIN both score prototype
AUC 1.0000 and R@1 48/48 there.

### 8.4 Threshold calibration (server, needs a run)

```bash
RUN=<a real run id>                     # never paste a placeholder
python tests/calibration/sweep_reconcile_thresholds.py "$RUN" --cross 0.30,0.40,0.50,0.63,0.70
python tests/calibration/review_links.py "$RUN" --score --cross 0.45
python tests/calibration/rerender_from_clips.py "$RUN" --cross 0.45      # WATCH IT
python tests/calibration/review_links.py "$RUN" --label                  # record verdicts
python tests/calibration/analyze_decision_log.py logs/reconcile_decisions_$RUN.jsonl
```

Steps after the capture need **no camera time** — they replay frozen clips. Every
one of these tools prints the full setting line first; if it disagrees with the run
you are explaining, nothing below it means anything.

`cross_camera_threshold` **cannot** be calibrated from a single-camera run.

### 8.5 Wipe the store without running the pipeline

```bash
python -c "import sys; sys.path.insert(0,'src'); from database.store import PersonVectorStore; s=PersonVectorStore(url='http://localhost:6333'); s.reset(); print('points:', s.count())"
```

---

## 9. If you are picking this up cold — do this

**The single highest-value thing available is re-anchoring the thresholds (§3.4),
and it needs no camera time — but it needs GROUND TRUTH first (§3.4a).** A sweep
ranks settings by identity count, and a count is unjudgeable without knowing how many
people were present. Watch a run's output and label it before tuning anything.

1. Read `CLAUDE.md`, then `REMEDIATION_PLAN.md` §0 → Part A → Part J, then §3.4 and
   §6.6 here.
2. On the dev box: `python tools/preflight.py --load-model` then
   `python tests/run_all.py` — 19 files, seconds. Confirms the FastReID wiring, the
   Qdrant width and the geometry maths with no GPU.
3. On the A6000: sync the code (`./deploy.sh` **from the dev box**, or
   `git pull origin research`), fetch the checkpoint (§2.3), start Qdrant, then
   `python tools/preflight.py --load-model` again there.
4. **Sweep the thresholds on `20260803_121136`.** You know its ground truth: two
   people. Aim for 2–4 identities without fusing them:
   ```bash
   python tests/calibration/sweep_reconcile_thresholds.py 20260803_121136 \
     --cross 0.45,0.55,0.63 \
     --same "cam_219=0.90,cam_224=0.80 ; cam_219=0.75,cam_224=0.75 ; cam_219=0.65,cam_224=0.65"
   ```
5. **Re-render the best candidate and WATCH it** (§5.2 — a cluster count cannot tell
   you whether a cluster is one person or three):
   ```bash
   python tests/calibration/rerender_from_clips.py 20260803_121136 --cross <best> --same "..."
   ```
   Record the verdict with `review_links.py --label`. Two people is a small sample;
   do not treat a bar chosen from it as final.
6. Only then, if you still want geometry: record a **deliberate calibration walk**
   to §6.3's requirements — 8–12 distinct spots, standing, nobody seated — then
   `fit_floor_frame.py` → `backfill_geometry.py` → re-render with `--geometry`.
7. Put **cam_206** back in the next capture (§2.4); it has been missing since
   2026-07-30 and its recall problem has never been measured under yolo11m.

---

## 10. Open questions

- **Does the floor frame fit on this footage?** **First attempt failed** (§6.3):
  degenerate correspondences, 38% then 8% RANSAC inliers. Not yet answered, because
  no recording has met §6.3's requirements — the failures so far are about *coverage
  of the floor*, not about the maths. Still open, and cheap to retry once a
  deliberate calibration walk exists.
- **Is the veto worth having in an OFFICE at all?** §6.6 argues it is real but
  narrow: it catches co-temporal look-alikes across cam_219/cam_224, and stays silent
  on the time-separated merges that probably dominate here. Nobody has measured which
  kind your false merges actually are — the decision log
  (`analyze_decision_log.py`) could answer it from a run you already have.
- **Should the "is this person standing" check be built?** (§6.6) It would make
  seated observations fail open on purpose rather than by luck, and exclude them from
  calibration automatically. Not built.
- **Is the FastReID switch better?** Unproven. The only footage that could test the
  cross-camera-domain-shift argument holds OSNet vectors, so it needs a new capture
  or a re-embedding script (which does not exist).
- **Will a metric reference ever exist?** Until it does, cam_206 and cam_213 stay
  outside geometry entirely (3.3). A verified floor plan would be the cheapest fix.
- **What are the real RTSP URLs?** Not recorded in the repo.
- **How many distinct people are in `register_file.avi`?** Never established, which
  is why "9 identities" cannot be judged.
- **Should `IdentityStage` stop swallowing store errors?** (3.5) Currently silent
  data loss on a dimension mismatch.
- **Six documented defects remain PRESENT** in
  `tests/calibration/characterize_known_defects.py` — unguarded `_reinforce`, the
  single-bad-exemplar max-score problem, the two-lane leak, `_gid_coactive`
  skipping other cameras, and two pose-ensemble defects. Run that script after each
  phase to see what moved.
