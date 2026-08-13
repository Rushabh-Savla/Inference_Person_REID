# CLAUDE.md — rules that outrank your instincts on this repo

Read this before proposing anything. Every rule below is here because it was
learned the expensive way, and each one contradicts what a reasonable engineer
would otherwise do. Full detail lives in `ARCHITECTURE.md`, `REMEDIATION_PLAN.md`
Part A, and `ADR-003D`.

**For the offline-reconcile work in flight, read `AGENT_BRIEF.md` first** — it holds
the diagnosis of the operator's reported split (§3b below is the short version), the
table of approaches already ruled out, and the task order. `RECONCILE_PATCHES.md`
and `CAPTURE_PROTOCOL.md` are its companions. All three are tracked; they were
hand-pasted into four sessions before anyone noticed they were downloads.

This file is **tracked in git**, so it reaches the A6000 and every fresh session.
The ADR set is not (see §7).

---

## 1. Geometry: recorded live, consumed offline. Never recomputed.

> **The live run records geometry. Offline reconcile only consumes it.**

Each observation's floor position is computed **once, at capture**, by
`src/geometry/recorder.py`, and stored in that observation's Qdrant payload under
`floor`. `src/identity/reconcile.py` **reads** it. Reconcile must never:

- load a calibration record,
- apply a homography,
- derive a position from a bounding box.

Mechanically enforced: reconcile may import `geometry.reachability` (pure
arithmetic) and nothing else under `geometry/`.
`tests/live/test_geometry_not_recomputed.py` asserts it via the AST.

**Why, since reconcile already has the `bbox` and could just do it:**

1. The live RTSP feed is never recorded. A position not written during the run is
   gone — you cannot go back for it.
2. Re-fitting a calibration is cheap (`tools/fit_floor_frame.py` needs no camera
   time), therefore likely. If reconcile derived positions, re-reconciling a
   finished run would silently return **different identities**, with nothing in
   either output saying why. Reconcile is the authority on final ids; an authority
   that changes its mind for invisible reasons is not one.

One run, one geometry, decided once.

## 2. No metre-based threshold without a trusted metric reference.

The floor frame is fitted from camera imagery alone, which fixes the plane only up
to an **arbitrary scale**. Units are **floor units**, not metres.

> Metric geometry cannot be established from monocular cameras alone. A single
> trusted metric reference must be provided before any metre-based threshold
> (0.5 m, 3.0 m, walking speed in m/s) is considered valid.

The requirement is *one trustworthy source of metric scale* — **not** a tape
measure, and **not** a survey. In order of preference:

1. verified floor plans or CAD drawings;
2. known architectural dimensions (corridor width, door width, column spacing,
   floor-tile pitch), verified on site;
3. one or more independently measured reference distances (laser rangefinder, tape,
   measuring wheel) sufficient to establish **and validate** scale.

**As of 2026-08-03 none of these exist for this deployment**, so nothing claims
metres. `CalibrationRecord.is_metric` is False and the metre-facing API raises
`MetricScaleUnavailable`. This costs nothing: the within-group reachability check
compares a floor-unit distance against a floor-unit speed ceiling measured on the
same footage, so the unknown scale **cancels exactly**.

Metres are needed for exactly one thing: relating **separate floor groups**
(`group_distances`), because an operator-supplied real-world distance arrives in
metres. That is why `cam_206` and `cam_213` have no geometry.

## 3. Threshold tuning is not the lever. Measure first.

**Five** tuning changes have been reverted for *hurting* accuracy:
`identity.threshold` → 0.68, verifier `accept_threshold` → 0.62/0.08,
`max_occlusion_ratio` → 0.35, and the `live.topology` min-transit veto.

Settled by two live runs plus an offline sweep: cam_224 at 0.80 **fused several
people into one reid**; at 0.90 **one person shattered into many**. Both directions
wrong means the number is not the variable.

So: do not propose a threshold change as a fix. Fix the scoring, or add a physical
guard, and re-anchor bars only afterwards.

## 3a. The one measurement that exists in FastReID space (2026-08-04).

`identity.reconcile.same_camera_threshold` is **0.90 and too strict**. Measured on
`register_file.avi` with `measure_reconcile_thresholds.py`, prototype vs prototype:

```
same-person fragments merging   0.75 →100%   0.80 →100%   0.85 →75%   0.90 →50%
different people (proven distinct by co-occurrence)   n=6  p95=0.427  MAX=0.434
=> any bar in (0.434, 0.810) is perfect on this sample; 0.90 is ABOVE it
```

The evidence supports **0.80** — near the top of the window, because too low fuses
two people (unrecoverable) while too high splits one (recoverable). `cam_213` and
`cam_224` already override to 0.80, so it would only move `cam_219`/`cam_206`.

**The value has NOT been changed**, because a bar must be watched on video first
(§4). Full evidence sits beside the key in `config.yaml`. `cross_camera_threshold`
cannot be derived from a single-camera clip and remains un-recalibrated.

**A calibration script that cannot run on the shipping backbone is why nothing had
been recalibrated.** `measure_score_separation.py` crashed on FastReID until
2026-08-04 — it reached into torchreid OSNet's `fc` block. If a measurement seems
never to have been taken, check that its script still runs on the current model.

## 3b. The reported split is a SAME-camera bar, not a cross-camera one (2026-08-04).

The operator's standing symptom — one person carrying one reid in `cam_219` and a
different one in `cam_224` + `cam_213` — is **diagnosed**, with numbers. Do not
re-derive it. `explain_merge_failure.py 20260804_064551 1 2`:

```
camera    fragment pair     prototype   max_exemplar   consensus   bar
cam_213   0031 vs 0035      0.630 fail   0.663 fail    0.567 fail  0.80
cam_219   0020 vs 0008      0.574 fail   0.600 fail    0.468 fail  0.90
cam_224   0001 vs 0030      0.907 PASS   0.907 PASS    0.582 fail  0.80
```

cam_219 cannot merge its own front/back fragments at 0.90, so each is absorbed
cross-camera into a *different* cluster. Both clusters then contain cam_219
members, so the pair is judged by `strictest_same_camera_bar` = **0.90** — the
cross-camera `threshold` is **never consulted**. Consequences:

- **`identity.reconcile.threshold` cannot fix it.** Neither can geometry: a veto
  only ever *refuses* a merge, so it cannot create a missing link.
- **Lowering cam_219 to 0.80 is contraindicated**, not merely insufficient —
  measured, it captured `cam_219(8)` into `cam_219(38)` instead of the right person.
- `scoring: consensus` was tried and **reverted the same day**: it moved cam_224's
  pair from 0.907 PASS to 0.582 FAIL, which is the "cam_224 went reid 1, then 2,
  then back to 1" report. Sixth reverted tuning change.
- The published same-camera window ("any bar in (0.434, 0.810) is PERFECT") was
  measured by **splitting one continuous track in half** — the easy case. Real
  re-appearance measures **0.574**, so the same-person lower edge is ~0.57 and the
  usable window is far narrower than that comment claims.

Full detail, the retired-approaches table and the task order: `AGENT_BRIEF.md`.

## 3c. GROUND TRUTH EXISTS NOW (2026-08-05). Use it. Stop using proxies.

Run **`20260805_093512`** (3 cameras, 92 s, 1299 obs, clips + sidecars kept) is the
first run with **operator-confirmed identities read off the video**, and the first
with **CROSS-CAMERA same-person labels** — every cross-camera claim before this was
unmeasurable. 32 labelled pairs in `calibration/tracklet_pairs.jsonl` (8 same-camera,
24 cross-camera), three people:

```
A  reid 3 + 4 + 8   cam_213:19,22   cam_219:7,12,25   cam_224:3,15,24
B  reid 7 + 9       cam_219:6                          cam_224:1
C  reid 6 + 11      cam_219:5                          cam_224:4,10
```

**At the settings that shipped, 0 of 3 were recovered** — A split three ways, B and C
two ways each. Measured with `tests/calibration/sweep_against_labels.py`, which runs
the REAL reconcile over the run's stored vectors and scores every cell on the labels
plus 23 PROVABLE strangers (co-present in one camera >0.5 s):

```
people recovered / provable false merges / identities
                     same-> 0.90  0.80  0.70  0.60  0.55
prototype     cross 0.63     0/3   0/3   0/3   1/3   1/3   <- WAS SHIPPING
prototype     cross 0.45     1/3   1/3   1/3   3/3   3/3
max_exemplar  cross 0.55     1/3   2/3   1/3  3/3.0.7  3/3  <- NOW SHIPPING
```

Four things follow, and each one contradicts something this repo used to say:

1. **The CROSS-camera bar is the binding constraint, not the same-camera bars.** At
   `threshold: 0.63` no same-camera value reaches better than 1 of 3. Every previous
   attempt moved same-camera bars only. Now `threshold: 0.55`.
2. **`max_exemplar` beats `prototype`**, reaching 3/3 with 0.10 more headroom on the
   cross bar. It compares observation PAIRS instead of two means, which is the loss
   §3a's window was really describing. Now `scoring: max_exemplar`.
3. **PROVABLE false merges were ZERO in every cell of both grids**, down to cross 0.40
   / same 0.50. The false-merge fear that drove every conservative choice here does
   not materialise on this footage at any bar. That is *necessary, not sufficient* —
   co-presence can only convict people who share a frame.
4. **The grids are NON-MONOTONIC** (max_exemplar recovers 2/3 at same=0.80 but 1/3 at
   0.70). Merge order interacts, so a plateau is the unit of evidence and a single
   cell is meaningless. Never quote one cell.

**Raising `max_observations_per_side` from 64 to 0 (all observations) changes nothing**
in any mode — median observations per tracklet is 8, so the cap was never binding.
"Compare more embeddings" is already happening; the pooling function was the problem.

**These values are label-validated but NOT YET WATCHED.** The footage exists, so
close the loop with no camera time:
`python tests/calibration/rerender_from_clips.py 20260805_093512`

## 4. A cluster count cannot tell you whether a cluster is one person or three.

Every number quoted for the 2026-07-30 threshold change was **equally consistent
with the good and the bad outcome**, and both settings chosen that way made the
videos worse. Rank candidates with the sweep, then **render and watch**:

```bash
python tests/calibration/rerender_from_clips.py <run_id> --geometry
```

Ground truth is the operator watching video. Labels belong in
`calibration/link_labels.jsonl` via `review_links.py --label`, never in a
conversation — there are only 11 so far.

## 5. False merges are worse than false splits — except for vetoes.

A merge fuses two people's histories and corrupts both. A split just gives one
person a spare id, which reconciliation can still fix. When uncertain, mint.

**A veto inverts this.** Refusing a merge is unrecoverable — reconcile cannot
un-split a person — which is exactly how the topology veto destroyed a run
(`linked` collapsed 5 → 1, `topology_pruned=508`). So every geometric error budget
biases toward *permitting* the merge:

- distance is **shrunk** by both positions' error radii,
- elapsed time is **grown** by the clock-differential budget,
- the speed ceiling is **raised** by a safety factor.

A sloppier calibration therefore produces **fewer** vetoes, never wrong ones. When
in doubt, raise `geometry.reconcile.safety_factor`; never lower it.

## 6. Measurement rules that have already been got wrong once.

1. **A "different person" pair requires co-occurrence in the same frame, within one
   camera.** One body cannot be two simultaneous detections. **Across cameras this
   proves nothing here** — `cam_219` and `cam_224` share a room.
2. **Bank queries must be held out of the bank**, or a query matches itself at
   ~1.0 and inflates the same-person side.
3. **`other MAX` grows with sample size.** Prefer p95. On one clip, 48 vs 90 frames
   moved raw MAX from 0.819 to 0.936.

## 7. Facts about the repo that will otherwise waste your time.

- **The ReID model is FastReID SBS ResNet-101-IBN (MSMT17), 2048-d at 384×128.**
  **R101, not R50** — `fastreid_sbs_R50_ibn` is registered as a selectable option
  and is not used. Two things read as if OSNet were still running and are **not**
  the model in force: `backends.DEFAULT_BACKEND = "osnet_ain_x1_0"` (only the
  fallback when `reid.model` is absent) and `OSNetBackend.__init__`'s
  `arch="osnet_ain_x1_0"` (which OSNet *that class* builds; it cannot select
  FastReID at all). What runs is `config.yaml` → `reid.model` → the `BACKENDS`
  registry, and the run banner prints it. A mismatched checkpoint **raises**.
  OSNet-AIN weights stay in the repo because **every threshold in `config.yaml`
  was derived in its 512-d space** and none has been re-anchored.
- **The dev box is CPU-only WSL2, no CUDA.** FastReID R101-IBN is ~0.76 s/crop
  there. **Any conclusion about throughput, identity quality, or thresholds drawn
  on the dev box is wrong.** Use it for correctness only. The A6000 does all model
  runs, captures and calibration.
- **The ADR set and `NVIDIA_DeepStream_Adoption_Plan.md` are gitignored**, so they
  are **not on the A6000** and not in a fresh clone. Anything that must survive
  belongs here, in `ARCHITECTURE.md`, or in `config.yaml` comments.
- **`python main.py --reset` is not a wipe — it is a DESTRUCTIVE RUN. Never use
  it.** It clears the store *and then runs the whole pipeline*, so it looks like an
  ordinary run and nothing warns you. It has already cost this project its entire
  corpus once (`ONBOARDING.md` §7 attributes the 2026-08-03 wipe to exactly this
  command), and the store currently holds the **only run with usable ground truth**
  — `20260804_064551`, 2238 observations. Every measurement tool reads stored
  observations, so a wipe blocks all of them and no re-capture recovers the run that
  was watched.
- **`20260804_064551`'s CLIPS ARE GONE, and this file used to claim otherwise.**
  It said "clips and sidecars intact"; they were overwritten by `20260804_094039`
  and then `20260804_120409`. `._live_src_<cam>.mp4` carries **no run_id** and every
  run overwrites it, while `keep_frames: true` only stops the *final render* deleting
  it. So that run is **store-only, permanently**: it can never be re-rendered,
  contact-sheeted or re-embedded, and its three operator-confirmed pairs can never be
  extended. **Before every capture, copy the previous run's clips aside:**
  `for f in ._live_src_*; do cp -a "$f" "clips_<run_id>/$f"; done`
- **A stale CLIP is worse than a stale output.** Re-rendering run A while run B's
  clips are on disk draws A's stored ids onto B's pixels, and ByteTrack renumbers
  every run, so the two id spaces barely intersect: measured, `20260804_064551`
  against `20260804_120409`'s clips reported cam_219 **100% UNRESOLVED** and 84% of
  all boxes unidentified. That is a join error that looks exactly like a total
  identity failure, and it cost this project one round of operator video review.
  `rerender_from_clips.py` and `measure_unresolved.py` now REFUSE a sidecar whose
  `run_id` differs; the field was there all along and nothing checked it.
- **Every measurement in this project reads the STORE, so anything that fails
  upstream of the store is invisible to all of them.** The annotations sidecar is the
  only independent source — it holds every drawn box, including those of people who
  never produced a stored observation. Joining sidecar against store is what finally
  measured the "visible person with no id" rate (`measure_unresolved.py`). Any future
  claim about coverage must use it; a store-only statistic cannot see its own gaps.
- **`output_cam_*.mp4` existing does not mean the last run succeeded.** They are
  only overwritten by a completed render, so stale files look like success. Check
  mtime against the `run_id`.
- **Placeholders get pasted literally.** `<run_id>` has been run verbatim; `<` is a
  shell redirect. Write commands that derive their values.
- **Recorded files need MEDIA time, not `frame.ts`.** `ts` is stamped at frame read,
  so on a file it tracks decode speed (125+ fps), not events — two files read in
  parallel get timestamps that differ by thread scheduling. `Frame.event_ts()`
  returns media time (`offset + frame_index / fps`) for files and wall-clock for
  streams; the stored payload's `ts` and all of geometry use it, while scheduler
  freshness, writer pacing and the live engine's TTLs deliberately stay on `ts`.
  Never conflate them — `tests/live/test_media_time.py` pins both halves.
- **File-batch mode records no geometry** — `IdentityService._commit` stores neither
  `bbox` nor `ts`. To run geometry over recorded footage use `--mode live` on the
  files.
- **A veto's error budget must be checked against a SYSTEMATIC error, not just a
  noisy one.** The geometric check takes a median over pairings, which defends
  against one bad foot point but *not* against someone seated for a whole tracklet —
  every pairing is then wrong in the same direction. The deployment is an office
  where people sit, so this matters; nothing currently detects "not standing".
- **Never re-add `tests/live/_synth.py` to `.gitignore`.** It was untracked while
  its importers were tracked, and `python tests/run_all.py` failed on every fresh
  clone.

## 8. Before you run, and before you commit

```bash
python tools/preflight.py --load-model                    # BEFORE any run
python tests/run_all.py                                   # 19 files, CPU, seconds
python tests/calibration/verify_embedding_contract.py     # after ANY src/reid/ change
```

`preflight.py` verifies the configured stack and exits non-zero when a run would
**silently** produce garbage — chiefly a Qdrant collection at the previous backbone's
width, which the store only warns about and `IdentityStage` swallows, so the run
persists nothing and every id stays provisional with no error anywhere. It also
reports which `device` key each path reads, and that the thresholds are stale.

`verify_embedding_contract.py` is the only asserting script, and everything it
covers is a **silent** failure mode: a broken BGR→RGB swap, a transposed resize, a
batch/crop misalignment. No error, no crash, just worse matching.
