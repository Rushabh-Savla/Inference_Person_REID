# Pipeline Remediation Plan

**Status:** **ALL IMPLEMENTABLE ITEMS LANDED.** Three blocked on missing libraries or
vendored code; two finished features deliberately OFF (each re-scales every threshold);
Part C is out of scope by verification. **Not yet validated on footage since the last
batch -- the next action is a RUN.** See §0.
**Created:** 2026-07-30 · **Last updated:** 2026-07-31
**Scope:** detection, tracking, embedding, reconciliation, and the final rendered output

This document is the reference plan for fixing identity instability in the live RTSP →
reconciled-MP4 product. It records **what was measured**, **what was only read**, what we
deliberately decided *not* to change, and the design decisions settled while writing it.

Read Part A before proposing any threshold change. Read Part H before trusting any number.

---

## 0. START HERE — current state and next action

### 0.1 State, in one paragraph

**Everything in this plan that can be implemented, is implemented** (2026-07-30/31).
What remains is three items that are genuinely blocked on missing hardware libraries
or vendored code, a set of live-engine defects the plan already scoped OUT, and two
finished features deliberately left switched OFF because each one re-scales every
threshold in the config. The pipeline has NOT been validated on footage since the
last batch landed: **the next action is a run**, not a code change.

### 0.2 Do this first

```bash
python tests/run_all.py                                # expect 14 files pass
python tests/calibration/verify_embedding_contract.py  # expect PASS
python tests/calibration/characterize_known_defects.py # expect #25/#26 FIXED
```

Then capture a run (three or four cameras, 4-5 minutes, people crossing views):

```bash
python main.py --mode live --videos "rtsp://..." "rtsp://..." "rtsp://..." \
    > run.log 2>&1 &
echo $! > run.pid
# ...then ONCE, from a different prompt:
kill -INT $(pgrep -f "python.*main.py")
```

> **Never launch with `| tee`.** Ctrl-C reaches the whole foreground group, `tee`
> dies first, and every later `print` raises `BrokenPipeError`. Two production runs
> were lost that way and a third to running `tail -f` and `kill` in the same block.
> Redirect, background, signal by pid, follow the log separately.

Wait for `[live] shutdown complete.` -- that line, and only that line, means the ids
were decided and the videos written.

### 0.3 The workflow that makes this cheap now

Every threshold question used to cost a live run: cameras, people walking, five
minutes, and a fresh set of track ids that could not be compared to the last set.
**That loop is gone.** A run now leaves a complete record -- `._live_src_<cam>.mp4`
(clean frames), `._live_src_<cam>.annotations.json` (per-frame box geometry), and
its embeddings in Qdrant -- so reconcile and the render can be replayed offline on
the SAME footage with the SAME track ids:

```bash
# re-cluster at any settings, in seconds, read-only, no cameras and no model
python tests/calibration/sweep_reconcile_thresholds.py <run_id> \
    --cross 0.60,0.70,0.80 --same "cam_213=0.80,cam_224=0.80" --scoring consensus

# and turn any of them into WATCHABLE video for the operator to judge
python tests/calibration/rerender_from_clips.py <run_id> --cross 0.63,0.70

# what the run itself decided
python tests/calibration/analyze_decision_log.py logs/reconcile_decisions_<run_id>.jsonl
```

Requires `live.reconcile.keep_frames: true` (now the default). Replays everything
downstream of the store: thresholds, scoring mode, reciprocal-best,
`min_tracklet_observations`. Does NOT replay what changed the recording -- detector
model, `imgsz`, `reid.interval`, crop quality still need a live run.

### 0.4 The two switches that are OFF, and why

Both are implemented, tested, and one config line each. **Do not enable either
without re-deriving the thresholds in the same commit** -- each is a different
score space, not a better number in the same one.

| Switch | Default | What it changes |
|---|---|---|
| `reid.tap` (#39) | `post_relu` | post-BN keeps the negative half of the feature space. Measured: separation margin improved in EVERY scoring mode at BOTH sample sizes (+0.055 -> +0.086 at 48 frames, +0.108 -> +0.157 at 90), different-person ceiling 0.845 -> 0.782. Voids every threshold in Part H |
| `identity.reconcile.scoring` (#45a) | `prototype` | `consensus` / `max_exemplar` compare observation SETS instead of means. On the front/back counterexample prototype scores one person's two visits **0.640, below two strangers at 0.800**; consensus scores them 1.000 vs 0.800 and still holds a bad crop to 0.880 |

Procedure for either: flip it, sweep for new bars against a captured run, re-render,
watch, then commit the switch AND its bars together.

### 0.5 What is NOT implemented, and why

| # | Item | Why |
|---|---|---|
| 53 | NVDEC hardware decode | `NVDEC_IMPLEMENTED = False` is a stub for a GPU decode library that is not installed. Not blocking: four streams decode on CPU with <3% drops (J.9) |
| 54 | ByteTrack Kalman `dt` | Inside Ultralytics' `multi_predict`. Patching a vendored motion model to fix a defect that only bites under heavy frame loss, when measured loss is ~1%, risks tracking quality for no gain |
| 24 | Frame-level deterministic replay | The reconcile half is covered by 0.3. Frame-exact replay needs the capture decimation bypassed |
| 13-17 | Phase 1 leftovers: label-free correctness counters, per-camera timing, timestamp diagnostics | Pure instrumentation; nothing depends on them |
| 31, 37 | Renderer purity tests, `as-known-at-time` playback | #37 is a new feature, not a defect |
| 42, 44 | `min_tracklet_observations` and `RECIPROCAL_BEST` sweeps | Calibration, and both changed meaning when Phase 1 mutual-best landed. Redo with 0.3 |
| Part C | Live-engine defects (bank poisoning, two-lane leak, co-active expiry) | Verified end-to-end as computed and DISCARDED while `live.reconcile.enabled: true`. `live.identity.*` has no effect on the deliverable |

### 0.6 What has landed

| Commit | What |
|---|---|
| `182c677` | Phase 1 instrumentation: `decision_log.py`, reconcile instrumented additively, `tests/calibration/` |
| `e05c476` | **J.5:** finalization survives a failed `print`. Two runs had been lost to it |
| `8458a62` | **#40** per-camera `same_camera_threshold` (`per_camera` block, `strictest_same_camera_bar`) |
| `8bbd8d6` | **yolo11m** detector, with `compare_detector_models.py` (H.11) |
| `3dd0107` `680ce61` | **Offline threshold sweep** -- re-cluster a finished run in seconds, read-only, both axes |
| `c7cc79c` | **#41** first ever calibration of `cross_camera_threshold` |
| `e51175b` | **Offline re-render** -- `annotations.json` sidecar + `rerender_from_clips.py`; `keep_frames` now true |
| `a7a042c` | **#45a** scoring modes (`prototype` / `max_exemplar` / `consensus`), default unchanged |
| `9f5e7fc` | **#28/#29** RTSP TCP + socket timeouts; `VideoSource.open`/`_reopen` unified |
| `530b446` | Revert to run 2's reconcile settings after the operator rated the tuned ones worse |
| `afbfe9e` | **Phase 1 mutual-best** + **#38** simultaneity veto, both ON · **#45/#46** per-camera fps · **#35** palette · **#55 #58 #59 #64 #66** loud failures |
| `84ed133` | **#25** lone tracklet gets an identity · **#27** stale-roots merge bug · **#32/#33** UNRESOLVED · **#34** fit/margin labels · **#36** clipped labels |
| `cfc6fc9` | **#19-22** store transport/filter/validation · **#30** credentials · **#47** per-second embedding · **#56** batch chunking · **#57** camera recovery · **#61 #62 #63 #65 #67** |
| `9dafa42` | **#39** post-BN tap behind a flag |
| `7015ed4` | **#48** per-frame-period staleness · **#50** narrowed ReID lock · **#51 #52 #60** |

**Test state: 14 files pass.** Notable: `test_reconcile_physical_guards.py` (20),
`test_per_camera_same_camera_bar.py` (36), `test_phase1_decision_log.py` (28),
`test_reconcile_scoring_modes.py` (23), `test_rtsp_options.py` (17),
`test_shutdown_reaches_reconcile.py` (14).

### 0.7 Current shipping configuration

```
detector.model                              yolo11m.pt
reid.tap                                    post_relu      (#39 available, OFF)
reid.interval_sec                           0.4            (#47)
identity.reconcile.scoring                  prototype      (#45a available, OFF)
identity.reconcile.threshold                0.63           cross-camera
identity.reconcile.same_camera_threshold    0.90           global
  per_camera: cam_213 0.80, cam_224 0.80
identity.reconcile.same_camera_reciprocal_best  true       Phase 1 mutual-best
identity.reconcile.covisibility.enabled     true           #38, 6 pairs
source.rtsp                                 tcp, 5000 ms   (#28/#29)
live.reconcile.keep_frames                  true           enables 0.3
```

### 0.75 The machine, and what is already on it

Development happens on a **CPU-only WSL2 box** (no CUDA). Everything with a camera
or a GPU happens on the **A6000 server**, and the two are synced by git:

```bash
# dev box                          # A6000
git push origin research           git pull origin research
```

`deploy.sh` rsyncs instead, and is the only way to move gitignored files (model
weights). `yolo11m.pt` is gitignored, so a fresh clone downloads it on first run --
or `scp` it if the server has no outbound network.

Server project root: `~/seifer_work/Inference_PersonReid`. Qdrant runs there in
Docker (`docker compose up -d`, REST 6333, gRPC 6334 already mapped).

**A corpus already exists on that server. A new agent does NOT need to capture a
run to start working on reconcile:**

| run_id | Observations | Cameras | Notes |
|---|---|---|---|
| `20260730_093723` | 4209 | 4 | The J.6-J.12 analysis. No clips (predates the sidecar) |
| `20260730_111232` | 5048 | cam_213/219/224 | **The best corpus.** Clips + sidecars kept. Operator confirmed reid 5, 6 and 10 each held more than one person |
| `20260730_120551` | 1482 | cam_213/219/224 | ~2 min. Clips + sidecars kept |

So the sweep and the re-render (§0.3) can be run against real data immediately:

```bash
RUN=20260730_111232
python tests/calibration/sweep_reconcile_thresholds.py "$RUN" --cross 0.63,0.70
python tests/calibration/rerender_from_clips.py "$RUN" --cross 0.63
```

**cam_206 has been absent from every run since `093723`** -- it was dropped from the
launch command by a shell error and not re-added. It is the camera with the
detection-recall complaint (5 people present, 1 detected) and the only one whose
per-camera bar has never been measured under yolo11m. Put it back in the next run.

For a NEW run, get the id out of the log rather than reading it off the screen:

```bash
RUN=$(grep -o 'run_id=[0-9_]*' run.log | tail -1 | cut -d= -f2)
```

### 0.8 Reading order for a fresh session

Section 0 (this) -> **Part A** (what NOT to retry, with evidence) -> **Part J**
(field data and operator observations) -> Part G (design decisions) -> Part H
(measurements and their caveats).

Parts C and D exist to stop rediscovery: C is deferred/out-of-scope, D is verified
clean.

### 0.9 The one lesson this project keeps paying for

**A cluster count cannot tell you whether a cluster is one person or three.** Every
number quoted for a threshold change during the 2026-07-30 session was equally
consistent with the good and the bad outcome, and two settings chosen that way both
made the videos worse. Thresholds were then measured to be the wrong lever entirely
(Part A, first row).

So: no change that touches identity ships without a re-rendered video compared
against the previous one. Section 0.3 makes that a local job of seconds. Use it.
---

## Contents

0. [START HERE — current state and next action](#0-start-here--current-state-and-next-action)
1. [Product scope](#1-product-scope)
2. [Camera inventory](#2-camera-inventory)
3. [Evidence conventions](#3-evidence-conventions)
4. [Part A — Deliberately not changing](#part-a--deliberately-not-changing)
5. [Part B — Phases](#part-b--phases)
6. [Part C — Out of scope / deferred](#part-c--out-of-scope--deferred)
7. [Part D — Verified clean](#part-d--verified-clean-do-not-re-investigate)
8. [Part E — Runs required](#part-e--runs-required)
9. [Part F — Acceptance criteria](#part-f--acceptance-criteria)
10. [Part G — Design decisions log](#part-g--design-decisions-log)
11. [Part H — Measurement reference](#part-h--measurement-reference)
12. [Part J — Field results, 2026-07-30](#part-j--field-results-2026-07-30-a6000-4-cameras-live-rtsp)
13. [Part L — Handover notes](#part-l--handover-notes-2026-07-31)
14. [Part I — Historical context](#part-i--historical-context)

---

## 1. Product scope

Live RTSP in, `Ctrl-C`, out come **N reconciled MP4s — one per camera** (`output_<cam>.mp4`).

Success criteria, in priority order:

1. Identities are correct — no flicker, no swapping between people
2. Consistent **within** a camera and **across** cameras
3. Finalization (reconcile + re-render) is not slow

Non-goals: identities persistent across runs (ids are run-scoped); real-time on-screen ids;
batch mode on recorded files. `--mode live --videos foo.mp4` must keep working.

### 1.1 How the per-camera outputs form one set

`build_gid_map` is called **once** across the whole reconciled gallery, returning
`{(camera, track_id): gid}`. Each camera's render looks up its own `(name, track_id)`, so a
person walking cam_213 → cam_219 carries the **same id and colour in both files**. That is
the product claim. Several plan items exist solely to keep those four videos comparable to
each other — see #35, #44, #45.

---

## 2. Camera inventory

| Camera | Resolution | fps | Codec | Notes |
|---|---|---|---|---|
| cam_213 | 1920×1080 | 25 | H.265 | chokepoint |
| cam_224 | 2560×1440 | **15** | H.265 | co-visible with cam_219 |
| cam_206 | 1920×1080 | 25 | H.265 | |
| cam_219 | 2560×1440 | 25 | H.265 | co-visible with cam_224 |

**Co-visibility: only 224↔219.** No other pair can see one person at once, even briefly.
This is a hard physical constraint and Phase 7 exploits it.

Camera names derive from the IP's last octet, so replacing hardware changes names and the
config must follow.

---

## 3. Evidence conventions

Every finding below is tagged:

- **measured** — reproduced by a script that asserts, or quantified with numbers
- **verified** — confirmed by reading the exact code path (and quoted line)
- **read** — found by reading; mechanism is clear but impact is unquantified

Where a measurement has a small sample, the sample size is stated. Do not promote a
**read** finding to a fix without measuring first — that mistake was made four times during
this project's tuning history (Part I).

---

## Part A — Deliberately not changing

These were measured and rejected. Do not revisit without new evidence.

| Change | Why not | Evidence |
|---|---|---|
| NMS `iou: 0.60` | 0.60→0.45 removed 2 boxes of 190 on one clip, **0 of 201** on another. At the old 0.70 default there were 4 high-overlap pairs, so **0.60 already fixed duplicate boxes** | measured |
| `imgsz` 640 → 1280 | Same 5 track ids in every configuration, 2.4× the compute | measured |
| `conf` 0.4 → 0.1 + post-filter | Byte-identical results. ByteTrack's second-association stage is inert (its pool is 0.1–0.25, below the 0.4 pre-filter) but activating it changed nothing | measured |
| `same_camera_threshold` → 0.85 **as a decision** | Evidence was n=6 fragment pairs, one camera, one clip. Stays **0.90** until the Phase 9 sweep | measured, weak sample |
| `cross_camera_threshold` | No cross-camera data exists yet. Untouched until Phase 9 | — |
| Deepening `max_inference_queue` | 6 MB/frame; observed peak depth 900 ≈ 5.6 GB RAM | measured |
| Frame-drop → fragmentation fixes | **CONFIRMED DEAD on production hardware (J.1):** real drop rate is 0.5–6.5%, not the 85–99% measured on CPU-with-file. Phase 11 solves a problem that does not exist at this crowd size | measured, field |
| `TOP2_MARGIN` as a **gate** | Accepted and rejected margin distributions are near-identical on 163 real decisions (median 0.0224 vs 0.0186, p5 0.0017 vs 0.0016). A gate would be near-random. Compute and log it; never enforce it. See J.6 | measured, field |
| **Threshold tuning as the fix for id instability**, of ANY value on ANY bar | **Settled by two live runs, an offline sweep, and the operator watching the videos.** cam_224 at 0.80 fused several people into one reid; at 0.90 one person shattered into many numbers. BOTH directions wrong means the number is not the variable. The cause is #45a: reconcile compared prototype MEANS, which score one person's front-vs-back fragments **below** two strangers in similar clothing (0.640 vs 0.800 on the counterexample) -- so for a real fraction of people NO bar orders them correctly. Fix the SCORING or add a physical guard; tune bars only afterwards | measured, field |
| Choosing a setting from **cluster counts** | A cluster count cannot tell you whether a cluster is one person or three. Every number quoted for a threshold change on 2026-07-30 was equally consistent with the good and the bad outcome, and two settings picked that way both made the videos worse. Rank candidates with the sweep, then **render and watch** (§0.3) before shipping | measured, field |
| **Any** `same_camera_threshold` as the fix for id instability | **Settled 2026-07-30 by two live runs plus an offline sweep.** cam_224 at 0.80 fused several people into one reid; at 0.90 one person shattered into many numbers. Both were observed in the rendered videos. This is J.6's overlapping boundaries reaching the product, and it is #45a (prototype MEANS), not a number. Tune bars *after* choosing a scoring mode, never instead | measured, field |
| A **global** `same_camera_threshold` of any value | The per-subject boundaries overlap across cameras (J.6): p95 of "top different" = 0.816 exceeds p5 of "worst same" = 0.719. No single number works. **Per-camera bars landed 2026-07-30 (#40)** — so do not "fix" this by picking a better global number; tune the per-camera entries | measured, field |
| `yolo11n` → `yolo11m` as a **no-op** | Superseded 2026-07-30. Part A previously implied detector changes buy nothing, on the strength of `imgsz`/`conf` results. Capacity is a different lever and it **does** move fragmentation: yolo11m holds one 150-frame track where yolo11n splits the same person into 64 + 37 (H.11). Shipped as a measurement; the throughput half is still unmeasured | measured |

---

## Part B — Phases

Phases 1–3 change no output and need no new footage. Phases 4–8 each change behaviour and are
measured on the replay harness. Phase 9 is calibration. Phases 10–12 are performance and ops.

### Phase 0 — Baseline run (operator)

Qdrant up. Set `live.reconcile.keep_frames: true` and `live.metrics.log_interval_sec: 10`.
Four cameras, 2–3 minutes, people crossing between views, **one** interrupt:

```bash
# NOT `| tee`. See the warning below -- that pipeline cost two complete runs.
python main.py --mode live --videos rtsp://...213/1/1 rtsp://...224/ch01/0 \
    rtsp://...206/1/1 rtsp://...219/ch01/0 > run1.log 2>&1 &
echo $! > run.pid
# ...let it run, then ONCE:
kill -INT $(cat run.pid)
# watch it finish WITHOUT holding the terminal that owns the process:
tail -f run1.log        # Ctrl-C here is safe; it only stops tail
```

Keep `run1.log` and the four `._live_src_cam_*.mp4`. Then `grep -c hevc run1.log`.

> **Never launch this with `| tee`.** Ctrl-C reaches every process in the foreground
> group, `tee` dies first, and every subsequent `print` in the pipeline raises
> `BrokenPipeError` — which used to abandon reconcile entirely (J.5). That fix has
> landed (`e05c476`), so a broken stdout no longer costs the ids, but redirect-and-signal
> is still the correct habit: it keeps finalization off the terminal's process group, so
> a dropped SSH session or a closed window cannot interrupt it either. **Two production
> runs were lost to `| tee`, and a third to running `tail -f` and `kill` in the same
> command block** — the `tail` held the foreground and the interrupt went to the wrong
> process. Send the signal by pid, then follow the log separately.

Yields: per-camera drop rate, fragmentation ratio, first cross-camera data, replayable
footage, and the H.265 corruption answer.

> **Check free space first** — see #65. Four clips can reach ~21 GB/hour combined, and
> `keep_frames: true` means they are not deleted.

### Phase 1 — State model + instrumentation (no output change)

| # | Item | Evidence |
|---|---|---|
| 1 | State machine first-class: `Tracklet → Candidate → merged Candidate → Resolved \| Expired unresolved` | spec §2 |
| 2 | Provisional handles (`U-07`) in a separate namespace, **internal-only pre-reconciliation**, excluded from every identity tally by construction | spec §2.3 |
| 3 | Reconcile has **zero rejection diagnostics** — 4 `log()` calls, all successes | verified |
| 4 | All gates evaluated independently, **no short-circuit**; `gate_detail` on every record with the applied threshold value | spec §5.1 |
| 5 | Gate order: ranking-independent hard constraints → `ABSOLUTE_THRESHOLD` → ranking → `TOP2_MARGIN` → `RECIPROCAL_BEST` → merge | decision D3 |
| 6 | `TOP2_MARGIN` added, **logged only** (`threshold: null`); both `margin_eligible` and `margin_all_scored`; single `basis` config key | decision D4 |
| 7 | Annotated candidate vector: `excluded_by`, `would_fail_reciprocity`, `pair_similarity_to_best`, cameras, cluster size, observations, `candidates_truncated` | decision D5 |
| 8 | Summary block + test asserting it matches recomputation from the vector | decision D5 |
| 9 | `runner_up_differs` → run-level disagreement rate (bands: <1% drop the distinction, 10–20% meaningful, ~50% hard constraints reshaping the space) | decision D6 |
| 10 | Fragmentation metrics: tracklets per identity (mean/max/distribution); observations per tracklet **per camera** | new |
| 11 | Suppressed-tracklet count and share of total observations | verified silent today |
| 12 | Cross-camera metrics: identities spanning ≥2 cameras; per camera-pair candidates vs merged vs each rejection reason | new |
| 13 | Label-free correctness: two non-overlapping boxes in one frame sharing a positive id; id changes along one `(camera, track_id)` (must be 0) | new |
| 14 | Timing: reconcile + re-render wall time **per camera** | product constraint |
| 15 | `ts` is **receive time** (stamped after `cap.read()` returns), not capture time; no PTS ever read. Quantify per-camera jitter vs nominal 40/66.7 ms; probe `CAP_PROP_POS_MSEC` | verified |
| 16 | Measure 224↔219 offset **as a bound on apparent disagreement**, never as a timestamp-error estimate — see decision D2 | decision D2 |
| 17 | Per non-co-visible pair, negative-gap distribution for high-confidence same-person transitions (prototype cosine ≥ 0.85). Report sample sizes | decision D2 |
| 18 | ~~Fold the verification scripts used to produce Part H into `tests/` as a calibration harness~~ — **DONE 2026-07-30**, see [`tests/calibration/`](tests/calibration/). Building it immediately caught two methodological errors in Part H (see the note there) | housekeeping |

**Gate:** existing 8 test files pass; outputs byte-identical.

#### Phase 1 progress — 2026-07-30

**Landed.** `src/identity/decision_log.py` (new: gates, annotated candidate vector,
both margin variants, aggregates, JSONL writer, summary-consistency verifier) and
`reconcile.py` instrumented additively. Config under `identity.reconcile`:
`decision_log`, `top2_margin.threshold` (null = inert), `top2_margin.basis`. Wired
into `LivePipeline._decision_log_kwargs`, fail-soft so a bad log config never costs
the run's ids. Covers items **2–12, 18**.

`tests/live/test_phase1_decision_log.py` — **28/28 checks**, auto-discovered by
`run_all.py` (now 9 files, all passing). The load-bearing check is that reconcile
returns an **identical remap with and without a log attached**, across 7 scenarios
including time-overlap, suppression, and the single-tracklet defect. Also covers
acceptance criteria 3, 4, 6 and 9.

**Two things building it caught:**

- `MIN_OBSERVATIONS` could never fail, because suppression uses the same threshold, so
  every surviving tracklet passes by construction. Suppressed tracklets now get a
  decision record too — which is the only place that gate can fail, and it makes
  "how much am I losing to `min_tracklet_observations`" visible for the first time.
- `runner_up_differs` was degenerate. With no selectable candidate there is no margin
  decision, but it still counted as a disagreement — a single-camera run reported
  **100%**. Now `None` in that case, with `no_selectable_candidate` reported
  separately, so the band interpretation stays meaningful.

**Also landed:** `tests/calibration/analyze_decision_log.py` — reads a decision JSONL
and answers the Phase 9 questions (where the bar cuts, orphan count, near-tie
classification, reciprocity breakdown, per-camera eligible-set size, cross-camera
merges vs the stranger ceiling). It produced J.6 on its first real run.

And `tests/live/test_shutdown_reaches_reconcile.py` — 14 checks pinning the J.5 fix.

**Five self-caught errors so far**, all found by building the instrumentation rather
than by reasoning: the two Part H methodology bugs, `MIN_OBSERVATIONS` being
unfailable, `runner_up_differs` counting non-decisions, the analyser printing an
inverted "clean band", and — while landing #40 — the analyser's section 1 judging
**every** configured threshold against **every** subject, which with per-camera bars
blames 0.80 for subjects in a camera that never used it. Now filtered to the subjects
that actually faced each bar, and it prints which cameras used it. **That correction is
a no-op on a single-threshold log**, so run1's J.10 numbers stay directly comparable to
run2's. Assume more remain; prefer the *direction* of a comparison over its magnitude.

**Still open in Phase 1:** item 1 is partial (terminal states recorded; the
`Candidate` / `merged Candidate` distinction is implicit in `phase` + `merged_from`),
and items **13** (label-free correctness counters — these need the rendered output and
gid map, not reconcile), **14** (per-camera timing), and **15–17** (timestamp
diagnostics, which live in capture/pipeline) are not started.

### Phase 2 — Store transport + query hygiene

| # | Item | Evidence |
|---|---|---|
| 19 | **DONE** `cfc6fc9`. `store.prefer_grpc` (default `false`) + `store.grpc_port: 6334`. `docker-compose.yml` **already maps 6334** — no infra change | verified |
| 20 | **DONE** `cfc6fc9` -- server-side `run_id` filter in reconcile, `build_gid_map` and `print_run_summary`, with the Python check kept as a fallback. Was: `_gather_tracklets` scrolls **unfiltered** with `with_vectors=True`, discarding other runs in Python — cost grows with every run forever. Add server-side `scroll_filter` on `run_id` here, in `build_gid_map`, and in `print_run_summary` | verified; `scroll_filter` exists, unused |
| 21 | **DONE** `cfc6fc9`. Log active transport at startup; keep embedded `path=` mode working (no gRPC in-process) | required for tests |
| 22 | **DONE** `cfc6fc9` -- dim and distance metric are checked and warned about. Was: never validated | read |

**Gate:** bit-identical assignments over REST and gRPC on the same clip; finalization wall
time reported for both.

### Phase 3 — Offline replayability

| # | Item | Evidence |
|---|---|---|
| 23 | **DONE in the part that matters** `3dd0107`/`e51175b` -- reconcile AND the render replay from the store + clips + geometry sidecar, no model in memory (§0.3). Original: Reconcile runs end-to-end from a persisted score log with **no model in memory**, reproducing assignments bit-for-bit at identical thresholds | spec §6.2 |
| 24 | **NOT DONE** (see §0.5). Replay harness: frozen clips fed with `NewestSlot` and the 2-deep queue decimation bypassed, so runs are deterministic | prerequisite |

This turns Phase 9's sweeps into an interactive loop instead of a re-inference job.

### Phase 4 — Determinate reconcile fixes

| # | Item | Evidence |
|---|---|---|
| 25 | **FIXED** `84ed133`; the defect characteriser now reports it FIXED. Was: reconcile stamps **no identity at all** below 2 tracklets (`if len(keys) < 2: return {}`) — the whole video renders as bare `ID <n>` | **measured**: 1 tracklet, 5 observations, zero ids written |
| 26 | **FIXED** `afbfe9e` -- `span_ts` gathered per tracklet, frame-index fallback kept. Was: reconcile **never reads `ts`**; spans use per-camera frame indices, which are not comparable across 15/25 fps. Switch to `ts` with frame-index fallback | verified by grep |
| 27 | **FIXED** `84ed133` -- the merge loop re-reads the bar from CURRENT membership and skips the pair loudly if an earlier merge raised it. Snapshot and live views are separate by name so the decision log still records the round AS SCORED. Was: Phase 2 selects its threshold from a **stale `roots` snapshot** — a camera-sharing pair can merge at 0.63 instead of 0.90 | verified: `union` updates `members`, not `roots` |

`next_gid` cross-run collision **dropped from scope** — harmless with run-scoped ids, since
every consumer filters by `run_id`.

**Gate:** new regression tests; false-merge counter (#13) does not increase.

### Phase 5 — Input integrity

| # | Item | Evidence |
|---|---|---|
| 28 | ~~**No RTSP transport or timeout options anywhere.**~~ **LANDED `9f5e7fc`** (`source.rtsp`, applied before any capture opens; TCP is a sound default, not a measured fix — J.4 saw zero live decode errors). Set `OPENCV_FFMPEG_CAPTURE_OPTIONS` with `rtsp_transport;tcp` and a socket timeout | **measured**: 294/682 and 207/1573 broken H.265 references in project recordings; grep confirms none set |
| 29 | Without a socket timeout `cap.read()` can block indefinitely; the capture thread never re-checks `stop_event`, so `Ctrl-C` cannot stop that camera | read |
| 30 | **DONE** `cfc6fc9` -- `source.env_urls` names environment variables, resolved from the untracked `.env`, missing ones reported. Was: move credentials out of command-line URLs (visible in `ps` and shell history) into `.env` | observed |

**Rationale:** H.265 reference loss produces smeared frames that feed both the detector and
the ReID crop, so a corrupted crop poisons the tracklet prototype. Packet loss is random,
which no threshold explains — this is the leading hypothesis for random identity behaviour.

**Gate:** `grep -c hevc run2.log` vs run1; same-person cosine spread tightens or holds.

### Phase 6 — Renderer

| # | Item | Evidence |
|---|---|---|
| 31 | **NOT DONE** (see §0.5). Renderer becomes a **pure function of `(state, frame)`** — no state, no promotion. Both purity tests: frame *N* direct vs sequential `0..N` byte-identical; forward-then-backward label stability | spec §3 |
| 32 | **DONE** `84ed133` -- `UNRESOLVED` label in a fixed neutral grey that is deliberately NOT in the palette, so two unresolved people cannot look like one. No dashed box. Was: unresolved handles: distinct colour, dashed box, **no ID number**, never in a tally | spec §2.3 |
| 33 | **DONE** `84ed133` (same change as #32). Was: suppressed tracklets render as bare `ID <n>`, reading as the identity vanishing — replaced by #32's treatment | verified |
| 34 | **DONE** `84ed133` -- reconcile returns per-tracklet fit/margin, rendered as `REID 7 (0.94 / +0.21)`. Was: **REID confidences.** `render_final_videos` builds `SimpleNamespace` without `reid_score`, so the final MP4 shows no confidence at all. Add per-tracklet **fit** (prototype vs final cluster prototype) and **margin** (gap to nearest other cluster) → `ID 7 (0.94 / +0.21)`. Label as reconcile-scale — **not** comparable to live-engine scores | verified |
| 35 | **DONE** `afbfe9e` -- 20 distinct colours, hue-interleaved so consecutive gids are far apart. Was: palette has only **8 colours** (`id % 8`). Across four videos, where the check is "did this person keep their colour", collisions read as false merges that did not happen | verified |
| 36 | **DONE** `84ed133` -- label flips inside the box when it would clip, and x is clamped at the right edge. Was: labels clipped for boxes with `y1 < ~30` — people entering at the top have no visible id | read |
| 37 | **NOT DONE** -- a new feature, not a defect (§0.5). `as-known-at-time` playback mode reading the same state log | spec §4 |

### Phase 7 — Cross-camera simultaneity veto

| # | Item | Evidence |
|---|---|---|
| 38 | **DONE and ON** `afbfe9e` -- needs `ts` (#26); fail-open on unlisted pairs, co-visible pairs and missing timestamps, and every unlisted pair is named at startup. Was: `conflict()` skips cross-camera pairs entirely, so tracklets overlapping in time in non-co-visible cameras merge on appearance alone. This is the "id switches to another person" symptom | verified: `if a[0] != b[0]: continue` |

```yaml
identity:
  reconcile:
    covisibility:
      enabled: false              # fail-open default
      default_tolerance_sec: 1.0
      pairs:
        - [cam_224, cam_219, covisible]   # never vetoed
        - [cam_213, cam_206, 1.0]
        - [cam_213, cam_224, 1.0]
        - [cam_213, cam_219, 1.0]
        - [cam_206, cam_224, 1.0]
        - [cam_206, cam_219, 1.0]
```

Semantics: `covisible` = never vetoed. A number = veto when temporal overlap **exceeds** that
many seconds. **A pair absent from the list is unconstrained** — silence loses protection
rather than splitting a real person. Startup enumerates every camera pair present in the run
and warns on any absent, so changing camera arrangement is a config edit with a visible
checklist, never a code change.

Implemented inside `conflict()` so it inherits transitive cluster safety. A single-observation
tracklet has a zero-length span and can never be vetoed — consistent fail-open, surfaced in
metrics so it is not mistaken for the veto working.

> This is a **simultaneity** veto, not the minimum-transit-time rule that was reverted for
> pruning true matches (`topology_pruned=508`). It never constrains transition *speed*, so it
> cannot repeat that failure.

**Gate:** false-merge counter drops; identity count and fragmentation unchanged.

### Phase 8 — Feature tap

| # | Item | Evidence |
|---|---|---|
| 39 | **IMPLEMENTED, OFF BY DEFAULT** `9dafa42` (`reid.tap`; see §0.4). Verified here: post-ReLU 19.2% zero dims / 0% negative, post-BN 0% / 19.2%. Was: `fc = Sequential(Linear, BatchNorm1d, ReLU)` and eval-mode `forward` returns `self.fc(v)` — the shipped embedding is **post-ReLU**, confined to the non-negative orthant. Add post-BN behind a config flag; record the tap in the run config; refuse to compare score logs across taps | **measured** |

Measured effect (bank scoring, 14 proven-distinct pairs — see the corrected H.2/H.3):
post-BN improves the separation margin in **every** scoring mode at **both** sample sizes,
e.g. `+0.055 → +0.086` at 48 frames and `+0.108 → +0.157` at 90, and lowers the
different-person ceiling (`0.845 → 0.782`, `0.892 → 0.858`). The *direction* is the stable
result; the magnitudes are not. At matched false-accept rates recall was comparable, so this
**buys calibration robustness rather than accuracy** on available footage.

Untested hypothesis: the benefit may be larger cross-camera, where post-ReLU compression
pushes same-person scores toward the stranger band.

> **Changing the tap voids every threshold in Part H.**

### Phase 9 — Calibration sweeps (needs Phase 0 + 3)

| # | Item |
|---|---|
| 40 | ~~**`same_camera_threshold` must become PER-CAMERA**~~ — **LANDED 2026-07-30, unmeasured.** Field data (J.6) showed 0.90 yields 9.0 eligible partners per subject in cam_206 but **0.0** in cam_213 and 0.5 in cam_224, with overlapping per-subject boundaries so no global value works. Shipped: `identity.reconcile.per_camera` (cam_213 and cam_224 at **0.80**), resolved by `reconcile.resolve_same_camera_thresholds` — the single merge point both call sites use, so live and file-batch cannot drift, exactly as `resolve_detector_cfg` does for the detector. `pair_threshold()` resolves a CLUSTER pair's bar with `strictest_same_camera_bar`: **max** over the cameras the two clusters share, since merging asserts the same-person claim for every one of them and the loosest camera must not launder a merge past the tightest. Malformed config is skipped loudly (an unnoticed `80` for `0.80` would silently disable same-camera merging); a configured camera absent from the run is named in the log (D1). 36 checks in `tests/live/test_per_camera_same_camera_bar.py`, including that **no overrides reproduces the old behaviour exactly**. **Still a hypothesis from one run — it has not been measured on footage** (D9) |
| 41 | `cross_camera_threshold` — still uncalibrated. Do this only AFTER #40, since fewer orphans means fewer spurious competitors and the picture will change |
| 42 | `min_tracklet_observations` — LOW priority: only 8/97 tracklets and 0.3% of observations suppressed (J.6) |
| 43 | ~~`TOP2_MARGIN` sweep~~ — **ANSWERED, see Part A.** Accepted and rejected margin distributions are near-identical (median 0.0224 vs 0.0186), so it carries no information. Keep `threshold: null` |
| 44 | `RECIPROCAL_BEST` — rejects 31% of decisions at up to 0.905 (J.6). Re-examine only after #40 |
| 45 | Consensus vs `max(proto, exemplar)` scoring — consensus gave the **lowest different-person ceiling** of the three modes at both sample sizes (corrected H.2/H.3). Also the candidate fix for the front/back blurring in #34a |
| 45a | ~~**Reconcile compares prototype MEANS**~~ — **LANDED `a7a042c`**, default still `prototype` because the mode voids every threshold. Measured on a front/back counterexample: prototype scores one person's two visits **0.640, BELOW two strangers at 0.800**; `max_exemplar` fixes the ordering (1.000) but one bad crop in ten fuses two people; `consensus` fixes it *and* holds that bad crop to 0.880. Original text: reconcile compares prototype MEANS, which blurs front/back into a vector matching neither view.** The live engine's `_bank_score` documents avoiding exactly this with `max(prototype, best_exemplar)`; reconcile never got that fix. This is the structural cause behind cam_213's front/back split — #40 treats the symptom, this treats the cause |

**A high `TOP2_MARGIN` failure rate is a diagnostic, not evidence of under-merging.** The
discriminator is `pair_similarity_to_best`:

| `pair_similarity_to_best` | Cause | Remedy |
|---|---|---|
| ~0.95 between tied candidates | one person fragmented into two clusters | lower `same_camera_threshold` |
| ~0.55 | genuinely ambiguous embeddings (uniforms, low light, bad crops) | feature / crop-quality work, not merging |
| many candidates above the floor | over-permissive candidate generation | raise the floor |

Deliverable: **curves, not values.**

### Phase 10 — Per-camera timing

| # | Item | Evidence |
|---|---|---|
| 45 | Single global `output.fps_default: 20` applied to all four cameras ([pipeline.py:203](src/live/pipeline.py#L203), [:430](src/live/pipeline.py#L430)) → four videos with four different wrong time scales, which **blocks visual cross-camera verification** | verified |
| 46 | Output MP4 timeline **5.5× too fast** (157 frames @20fps for 42.9 s of content). `WriterStage` has correct wall-clock pacing but is never constructed in reconcile mode | **measured** |
| 47 | **DONE** `cfc6fc9` -- `reid.interval_sec: 0.4` converts per camera from its measured rate (cam_224 every 6 frames, the fast cameras every 10). Was: `reid.interval` counts processed frames → cam_224 at 15 fps accumulates ~40% fewer observations per second, giving the weakest prototypes to the hardest camera pair. Convert to seconds | arithmetic |
| 48 | **DONE** `7015ed4` -- `max_frame_staleness_periods: 2.5` converts per camera (cam_213 ~100ms, cam_224 ~169ms), falling back to the absolute bound until a rate is known. `track_buffer` untouched. Was: absolute → 2.5 vs 1.5 frame periods across cameras | read |
| 49 | `detector.per_camera` is `{}` — the intended lever for heterogeneous cameras, unused | read |

### Phase 11 — Throughput (dropped to lowest priority by J.1; **re-open if yolo11m drops frames**)

> J.1 measured real drop rates of 0.5–6.5% and declared this phase largely dead — but that
> was at `yolo11n`'s cost. The detector is now `yolo11m` (~10× the FLOPs, 2.0× measured on
> CPU: H.11). If run2's per-camera `dropped%` climbs, the items below stop being dead code
> and #50 / #51 become the cheapest way to pay for the bigger model.


| # | Item | Evidence |
|---|---|---|
| 50 | **DONE** `7015ed4` -- the lock moved INTO `ReIDExtractor` around the forward pass alone; per-camera embedders now overlap (measured 3 concurrent where the old lock pinned it at 1). Was: `_embed_lock` wraps all of `TrackEmbedder.process` — cropping, a float64 Laplacian, an O(N²) occlusion loop, all preprocessing — not just the forward pass, contradicting its own docstring | verified |
| 51 | **DONE** `7015ed4` -- 2 -> 4, raised modestly not deepened (each slot is a full batch at ~6 MB/frame). Was: drops dominated by `max_inference_queue: 2` (629 of 637 batches on a file run; `slot_drop` was 1) | **measured** |
| 52 | **DONE** `7015ed4` -- `reid.warmup_spacing: 3` frames (~0.12s) so warmup samples different moments. Was: `warmup_embeddings: 3` fires on consecutive frames → ~1 effective view for short tracklets, exactly those needing to clear 0.90. Spread it | read |
| 53 | **BLOCKED, not implementable** (§0.5) -- `NVDEC_IMPLEMENTED = False` is a stub for a GPU decode library that is not installed. 251 Mpx/s of CPU-only H.265 decode (cam_219 alone 92.2); NVDEC is a stub (`NVDEC_IMPLEMENTED = False`) | arithmetic |
| 54 | **BLOCKED, deliberately** (§0.5) -- inside vendored Ultralytics; patching it risks tracking quality to fix a defect that only bites under heavy loss (measured ~1%). ByteTrack's Kalman filter advances one step per `update` regardless of elapsed time — no `dt`. Under load-dependent dropping the predicted box is wrong by however long the gap really was | verified |

### Phase 12 — Robustness

| # | Item | Evidence |
|---|---|---|
| 55 | **DONE** `afbfe9e` -- all four stages report and re-raise, and shutdown names every stage that died. Was: no exception guard in any worker `run()`; no watchdog. A dead `InferenceStage` runs forever writing `CAMERA OFFLINE` frames | verified by grep |
| 56 | **DONE** `cfc6fc9` -- chunked at 32, verified embedding-invariant (5.2e-08) because eval-mode BatchNorm uses running stats. Was: ReID batch unbounded in number of people; no chunking → OOM risk on a weaker GPU | verified |
| 57 | **DONE** `cfc6fc9` -- retries every 30s indefinitely after the fast budget; a camera that comes back rejoins. Was: camera death permanent after ~15 s of retries; all-dead ends the run, so a 20 s switch reboot kills a session | read |
| 58 | **DONE** `afbfe9e` -- clips are KEPT on render failure, with the re-render command printed. Was: clips deleted in a `finally` even when the render failed, destroying the only copy of the footage | read |
| 59 | **DONE** `afbfe9e` -- a banner says reconcile is disabled and the ids will be provisional. Was: Qdrant down → `_build_store` returns `None` → reconcile **silently disabled** → live-annotated output instead. Make it fail loudly | verified |
| 60 | **DONE** `7015ed4` -- the project root is derived from `__file__` and put on `sys.path` first. Was: `from main import` inside `_finalize_offline` → fails outside the project root, returning without rendering | observed |
| 61 | **DONE** `cfc6fc9` -- a failure mid-start stops and joins whatever came up. Was: threads started outside `try/finally`; identity never drains on shutdown; render's drain races identity | read |
| 62 | **DONE** `cfc6fc9` -- later frames are resized to the latched size so geometry stays aligned, warned once. Was: clip writer latches frame size from the first frame → a mid-run resolution change desyncs annotations from the clip permanently | read |
| 63 | **DONE** `cfc6fc9` -- the configured codec reaches the render, falling back to mp4v if the build lacks it. Was: `output.codec: h264` ignored on the product path (mp4v hardcoded); metrics always report `written=0` in reconcile mode | verified in a real run |
| 64 | **DONE** `afbfe9e` -- says so loudly. Was: a camera with no annotations produces **no video and no message** (`if not annos: return`). With four cameras this is easy to miss. Log loudly and record per-camera outcome in the summary | verified |
| 65 | **DONE** `cfc6fc9` -- `live.reconcile.max_clip_gb: 4.0` per camera, plus a free-space report before the run. Annotation RAM untouched. Was: clip disk and annotation RAM are unbounded and scale with camera count. Measured **51.0 KB/frame** at 1920×1080 mp4v → see table below. No cap, no rotation, no free-space check | **measured** |
| 66 | **DONE** `afbfe9e` -- defaults False. Was: `_g("inference","pose_ensemble", True)` defaults **True** — deleting one config line enables the duplicate-box generator on the live path. Change the default to `False` | verified |
| 67 | **DONE** `cfc6fc9` -- points at footage that ships with the repo. Was: `source.videos` points at non-existent `placeholder_video/`, so bare `python main.py` fails | observed |

Clip growth, assuming no frames dropped:

| Camera | Rate | Per hour |
|---|---|---|
| cam_213 1920×1080@25 | ~1.2 MB/s | 4.4 GB |
| cam_224 2560×1440@15 | ~1.3 MB/s | 4.7 GB |
| cam_206 1920×1080@25 | ~1.2 MB/s | 4.4 GB |
| cam_219 2560×1440@25 | ~2.2 MB/s | 7.8 GB |
| **total** | **~5.9 MB/s** | **~21 GB/hour** |

---

## Part C — Out of scope / deferred

### Live `IdentityEngine` — off the product path

**Verified end to end:** with `live.reconcile.enabled: true` (the default and current product
mode) the live engine's ids are **computed and discarded**. `_observation_payload` stores no
`reid_id`, `RenderStage._capture` records only `x1,y1,x2,y2,track_id,confidence`, the final
MP4's ids come solely from `build_gid_map` reading what reconcile stamped, and the live
pipeline is headless. So `live.identity.*` thresholds have **no effect on the deliverable**.

Deferred defects (all **measured** by reproduction): bank poisoning driving a wrong person to
cosine 1.000 via unguarded `_reinforce`; a single bad exemplar reaching 1.000 because
`score()` takes `max(prototype, best_exemplar)`; the two-lane leak letting a stranger take
another person's id at cosine 0.633 while incrementing *both* `recam_rej_below` and `linked`;
co-active window expiry letting a look-alike inherit a gid; cross-camera co-presence not
vetoed (`_gid_coactive` skips other cameras); LRU eviction dropping a gid an active track
still holds; `O(gids × tracks)` matching with per-call `np.stack`.

**Recommendation: disable the engine in reconcile mode** to reclaim the identity thread.

### Batch path — unused

Pose-ensemble duplicate boxes (**measured**: 0 overlapping pairs without it, 2 with it, fires
on 5/40 frames, creates synthetic `track_id + 100000·k` ids whose "primary" flips between
people); `_same_camera_overlap` never returning `True` (**measured**); `_maybe_reassign_track`
dead because `_commit` adds the embedding to the bank first, forcing `current_score` to
exactly 1.0; the verifier's unbounded `observation_count` drift (accept bar 0.733 at obs=0 →
0.543 at 300 → 0.466 at 3000, against a ~0.57 stranger ceiling) and prob-gap saturation (at
obs=300, cosine 0.90 vs 0.85 gives a 0.0019 gap, below `min_prob_gap: 0.05`, so a clean winner
is rejected and a duplicate minted); unfiltered `store.search`; asymmetric
`_reciprocal_best_ok`; ~300 lines of unreachable `_assign_online`.

### Low value

`crop_saver` frame-index throttle bug and unbounded dict (disabled); `http://` static files
misclassified as streams so EOF triggers replay; empty `models/fastreid`.

---

## Part D — Verified clean, do not re-investigate

**Entire embedding path.** Checkpoint loads 550/552 keys with only the discarded classifier
head unloaded — no layer left at random init. `model.training is False`. Output `(N, 512)`,
unit norms. Determinism: same batch twice, max abs diff `0.00e+00`. Batch order preserved
(`argmax(single_i · batch_j) = [0 1 2 3 4]`). Batch invariance `6.26e-07`. BGR→RGB present and
load-bearing (cosine 0.903 if swapped). Resize is `(W=128, H=256)`, not transposed. ImageNet
normalisation applied after the channel permute. Empty crop raises `ValueError` rather than
injecting a vector. Cache row aliasing is memory retention only — no cross-track contamination.

**Qdrant.** `Distance.COSINE`; `score` **is** similarity, higher = closer; round-trip exact
(numpy 0.5846 → qdrant 0.5846).

**Reconcile `_prototype`** math correct, norm 1.0. `_prototype([])` raises but is unreachable
(line 124 filters empty vector lists).

**`InterruptGuard`** soundly engineered — the `pthread_sigmask` + `sigwait` approach with a
Python handler as backup correctly solves the blocked-signal problem. No defect found.

**Clip↔annotation alignment invariant** held in both end-to-end runs (annotations == clip
frames == output frames).

**Per-camera tracker isolation** correct: one `PersonDetector` (hence one `YOLO`, hence one
ByteTrack) per camera, one worker per camera, no cross-contamination possible.

**`_bin6`** correct — `int(0.7 * 10) == 7` in CPython; an earlier off-by-one claim was wrong.

**ByteTrack cannot invent boxes** — `_format_output` returns one row per activated track and
association is one-to-one. Across 120 frames the tracker returned more rows than detections in
**0** frames.

---

## Part E — Runs required

| Run | Purpose |
|---|---|
| ~~**run1** — baseline, Phase 0~~ | **DONE**, `20260730_093723`. Per-camera drop rate, fragmentation ratio, H.265 answer, first cross-camera data, frozen clips. Analysis in J.6–J.12 |
| **run2** — #40 + yolo11m, **the next run** | Did the orphan count fall and did the operator's four named symptoms (J.8) go away? Diff analyser sections 1/2/5 against J.10, and per-camera tracklet counts against J.9. **Also the throughput verdict on yolo11m** — per-camera `dropped%` vs J.9's 0.4–2.9% |
| ~~**run2** — #40 + yolo11m~~ | **DONE**, `20260730_111232` (3 cameras; cam_206 was dropped from the command by a shell error). Operator: reid 5, 6 and 10 each held more than one person |
| ~~**run3** — tightened bars~~ | **DONE**, `20260730_120551`. Operator rated it WORSE: one person cycled through many numbers. Both changes reverted (`530b446`) |
| **run4 — the next run.** All guards on, thresholds at run 2's values | Did Phase 1 mutual-best and the #38 veto remove the shared-reid failures WITHOUT re-splitting people? Check `grep -c "refused for not being each other's best"` and `grep "simultaneity veto"`. Keeps its clips, so any follow-up setting is testable offline |
| **run5** — after picking a scoring mode or tap (§0.4) | Only once run4 gives a baseline. Change ONE of them, re-derive its bars, re-render, watch |
| Single-person route walk through every camera | Still the cleanest data for per-pair tolerances (#17) and cross-camera calibration |
| Crowded run | Every Part H measurement tops out at 6 people; the production regime is untested |
| **Single-person route walk** through all four cameras with rough timings | cleanest data for per-pair tolerances (#17) and `cross_camera_threshold` |
| **Crowded run**, as many people as possible | every measurement in Part H tops out at 6 people; the production regime is untested |

---

## Part F — Acceptance criteria

1. Renderer purity — frame *N* direct equals frame *N* from a sequential `0..N` pass,
   byte-identical; forward-then-backward label stability.
2. No provisional handle appears in any identity count or unique-person tally.
3. A tracklet failing *n* gates has *n* entries in `gates_failed`, proven by a fixture
   constructed to fail two gates simultaneously.
4. `gate_detail` populated for every gate on every record, passing and failing alike.
5. Reconciliation runs from a persisted score log with **no model loaded** and reproduces the
   original assignments bit-for-bit at identical thresholds.
6. Candidate↔Candidate merge exercised by a fixture where two sub-threshold tracklets from
   different cameras resolve correctly **only** when merged.
7. `as-known-at-time` mode and default mode both derive from the same state log.
8. REST and gRPC produce identical assignments on the same clip.
9. The `TOP2_MARGIN` summary block matches recomputation from the annotated candidate vector.
10. The existing test files stay green throughout — **14 files pass** as of 2026-07-31.

**Status of these criteria:** 2, 3, 4, 6, 9 and 10 are met and pinned by tests. 5 is
met for reconcile (§0.3 replays it from the store with no model) but not from a
persisted *score* log. 1 and 7 are not met (#31, #37 — see §0.5). 8 is untested:
gRPC is available (#19) but REST/gRPC have not been diffed on the same clip.

---

## Part G — Design decisions log

**D1 — Co-visibility is configuration, not code.** Changing camera arrangement means editing
the `pairs` list. Startup enumerates every pair present in the run and warns on any absent, so
a forgotten pair is visible rather than silent.

**D2 — The 224↔219 offset is a bound on *apparent disagreement*, not a timestamp-error
estimate.** It conflates: timestamp error proper (network jitter, RTSP buffering, the decode
cost gap between 92.2 and 55.3 Mpx/s), frame-rate quantisation (66.7 ms vs 40 ms grids),
load-dependent frame dropping, detection latency, ByteTrack's two-frame confirmation delay,
the crop-quality gate, `min_tracklet_observations`, and — probably the largest term —
**non-identical fields of view**, since a person can be well inside one view while genuinely
outside the other. Therefore it is a **floor** for the tolerance, never a value to shrink
toward, must not be reported as "our timestamp error is X ms", and must be re-measured after
any throughput change. Because it is partly geometric it does **not** transfer to other pairs,
which is why D7 exists.

**D3 — `RECIPROCAL_BEST` comes *after* `TOP2_MARGIN`.** `best_partner[a]` is single-valued, so
at most one candidate can pass reciprocity with any cluster. Filtering by reciprocity before
ranking leaves ≤1 candidate, no runner-up exists, and the margin gate becomes dead code.
Pre-margin constraints must be **independent of the cluster's own ranking**: temporal
conflicts, `NOT_MERGEABLE_CROSS`, and the absolute floor.

**D4 — Both margins logged, exactly one operational.** `margin_eligible` (runner-up over the
post-hard-constraint set) and `margin_all_scored` (runner-up over everything above the floor)
answer different questions — *selection safety* versus *evidence quality* — and disagree on
real inputs. A single `basis: eligible | all_scored` config key makes gating on both
inexpressible. `threshold: null` in Phase 1 means computed and logged, enforcing nothing.

**D5 — Exclusion reasons live on the candidate vector.** Rather than parallel summary fields,
each candidate carries `excluded_by` (enumerated, ranking-independent) plus a
`would_fail_reciprocity` **annotation** that excludes nothing. Under D3 reciprocity cannot be
an exclusion reason, but the annotation preserves the diagnostic: if near-ties are frequent and
the runner-up almost always carries `would_fail_reciprocity: true`, reciprocal-best is already
resolving them and `TOP2_MARGIN` adds nothing.

Enumerated `excluded_by` values: `TEMPORAL_CONFLICT_SAME_CAMERA`,
`TEMPORAL_CONFLICT_CROSS_CAMERA`, `NOT_MERGEABLE_CROSS`, `BELOW_ABSOLUTE_THRESHOLD`, `null`.

**D6 — Margin disagreement is itself a metric.** `runner_up_differs` aggregated to a rate,
read against <1% (drop the distinction) / 10–20% (meaningful design choice) / ~50% (hard
constraints fundamentally reshaping the candidate space). In this system the only new
pre-margin constraint is the simultaneity veto, so the disagreement rate doubles as a direct
measure of that veto's impact on candidate generation.

**D7 — Simultaneity tolerances are per-camera-pair from the outset**, even though all default
to 1.0 s. Small cost now; the clean path if pairs prove to have materially different timing
characteristics. Overlap is symmetric so pairs are undirected. Whether tolerance should scale
with the pair's coarser frame period is left to the Phase 1 metrics.

**D8 — Ids are run-scoped.** No cross-run gallery. The spec's gallery states collapse to
cluster membership; `next_gid` collisions become harmless; no gallery-hygiene workstream. The
store still accumulates across runs, which is why #20 (query hygiene) matters.

**D9 — Thresholds are hypotheses until swept.** `same_camera_threshold` stays at 0.90.
Deliverables from Phase 9 are curves, not values.

---

## Part H — Measurement reference

**Reproduce any of this with the harness in [`tests/calibration/`](tests/calibration/)** —
see its [README](tests/calibration/README.md) for which script produces which subsection.
The scripts are the source of truth; this section is a snapshot.

All numbers below were produced on `register_file.avi` (2560×1440, **0 H.265 decode errors**),
which matches production camera resolution. **Sample: 6 people, one camera, one clip, 14
proven-distinct pairs.** The "different person" set uses only tracks that **co-occur in the
same frame**, since a person cannot be two simultaneous detections — earlier figures that
included non-co-occurring track pairs were contaminated by fragmentation and are not
reproduced here.

> **Two methodological errors were made and corrected while producing this section.** Both
> inflated the results in the optimistic direction, and both are now enforced in code by the
> harness. (1) Different-person pairs must co-occur in a frame — otherwise fragments of one
> person enter the different-person set. (2) Bank queries must be held out of the bank —
> otherwise the max-exemplar term matches a query against itself and returns 1.000. Assume
> further methodological error is possible; prefer the *direction* of a comparison over its
> magnitude.

### H.1 Raw crop-to-crop cosine (48 frames @ stride 6)

| Tap | same mean | same p5 | other mean | other p95 | other MAX | margin |
|---|---|---|---|---|---|---|
| post-ReLU (ships) | 0.885 | 0.710 | 0.552 | 0.640 | 0.819 | +0.070 |
| post-BN | 0.848 | 0.601 | 0.373 | 0.509 | 0.770 | +0.093 |

Sample-size sensitive — at 90 frames the post-ReLU ceiling rises to 0.936. See the note
under H.2/H.3.

### H.2 / H.3 Bank scoring and alternatives — **CORRECTED 2026-07-30**

> **The first version of these two tables was wrong.** Bank queries were not held out of
> the bank, so any query also present in the bank matched *itself* via the max-exemplar
> term and returned 1.000. That inflated the same-person distribution by whatever fraction
> of queries sat in the bank, which made the margin depend on frame count rather than on
> the model. `tests/calibration/measure_score_separation.py::_holdout` now splits each
> track into a bank half and a disjoint query half. The live engine documents this exact
> trap in `_reinforce`. Superseded figures were: bank post-ReLU margin +0.047 / ceiling
> 0.828; consensus +0.070 / 0.793.

Corrected, `register_file.avi`, two sample sizes to show the instability:

| Mode | Tap | margin @48f | ceiling @48f | margin @90f | ceiling @90f |
|---|---|---|---|---|---|
| raw crop-to-crop | post-ReLU | +0.072 | 0.819 | +0.075 | 0.936 |
| raw crop-to-crop | post-BN | +0.097 | 0.765 | +0.114 | 0.914 |
| **bank** `max(proto,exemplar)` (ships) | post-ReLU | +0.055 | 0.845 | +0.108 | 0.892 |
| **bank** `max(proto,exemplar)` | post-BN | +0.086 | 0.782 | +0.157 | 0.858 |
| consensus (mean of top half) | post-ReLU | +0.060 | **0.798** | +0.080 | **0.835** |
| consensus (mean of top half) | post-BN | +0.071 | **0.738** | +0.112 | **0.770** |
| prototype only | post-ReLU | +0.059 | 0.845 | +0.104 | 0.869 |
| prototype only | post-BN | +0.071 | 0.782 | +0.151 | 0.806 |

**Stable conclusions** — hold at both sample sizes, safe to act on:

1. post-BN beats post-ReLU on margin in every mode, and lowers the ceiling
2. consensus gives the **lowest different-person ceiling** of the three scoring modes
3. the different-person ceiling (0.78–0.94) sits **far above**
   `live.identity.same_camera_threshold: 0.70` — that threshold is inside the range where
   strangers score. The correction made this finding *stronger*, not weaker.

**Unstable — do not set a threshold from one run.** `other MAX` is an extreme-value
statistic and grows with sample size (raw post-ReLU: 0.819 → 0.936 between the two runs).
The margin itself moved +0.055 → +0.108. Prefer p95 over MAX, and get actual values from
the Phase 9 sweep on frozen multi-camera footage.

### H.4 Reconcile — prototype vs prototype (what Phase 1 of reconcile compares)

Same-person fragments simulated as one track split into two time-disjoint halves.

```
SAME-person fragments : mean 0.951   min 0.851   (n=6)
DIFFERENT people      : mean 0.613   p95 0.656   MAX 0.661   (n=12 pairs)

clean separation window: (0.661, 0.851)
configured same_camera_threshold = 0.90  ->  OUTSIDE that window

thr 0.85: genuine fragments merge 100.0%
thr 0.90: genuine fragments merge  83.3%   <- 1 of 6 fails (0.851)
```

### H.5 Feature space

| Tap | min | exact-zero dims | negative dims |
|---|---|---|---|
| post-ReLU (ships) | +0.0000 | 21.0% | 0% |
| post-BN | −0.0778 | 0% | 21.0% |

Post-ReLU is confined to the non-negative orthant, so cosine of any two vectors is ≥ 0.

### H.6 Detection / tracking (50 consecutive frames, 2560×1440)

| Config | distinct ids | dets | mean track len | ms/frame (CPU) |
|---|---|---|---|---|
| imgsz 640, conf .40, no drop | 5 | 183 | 36.6 | 74 |
| imgsz 640, conf .10 + post .40, no drop | 5 | 183 | 36.6 | 52 |
| imgsz 1280, conf .40, no drop | 5 | 191 | 38.2 | 175 |
| imgsz 640, conf .40, 2-of-3 dropped | 5 | 62 | 12.4 | 55 |
| imgsz 1280, conf .10 + post .40, 2-of-3 dropped | 5 | 64 | 12.8 | 145 |

### H.7 NMS

| iou | boxes (clip A) | pairs IoU≥0.5 | boxes (clip B) | pairs IoU≥0.5 |
|---|---|---|---|---|
| 0.70 (old default) | 193 | 4 | 205 | 4 |
| **0.60 (current)** | **190** | **1** | **201** | **0** |
| 0.45 | 188 | 0 | 201 | 0 |

### H.8 Quality gate on real footage

Rejected **1 crop of 214** (reason `occluded`). `bbox_quality_scalar` on accepted crops: mean
0.409, min 0.194, max 0.699 — never near 1.0, because its brightness term penalises CCTV
footage uniformly, so the aggregate weight carries little identity-relevant signal.

The occlusion gate is **inverted**: it divides intersection by the detection's *own* area, so
a person fully contained inside a nearer person's box scores exactly 1.000 and is rejected
every frame, while the occluder — whose crop genuinely holds two bodies — scores 0.246 and
passes. Mechanism verified; measured impact on this footage is only 0.5%, so it needs crowded
footage to size.

### H.9 Store transport

One 512-d vector: **2048 B binary (gRPC) vs 11265 B JSON (REST) — 5.50×.** At 50k
observations, 102 MB vs 563 MB, plus 0.114 ms/vector of `json.loads` (5.7 s).

### H.10 End-to-end product path

| Config | captured | rendered | discarded | tracklets | identities | aligned |
|---|---|---|---|---|---|---|
| `max_inference_queue: 2` (ships) | 1573 | 10 | 99.4% | 3 | 3 | yes |
| deep queue | 1060 | 157 | 85.2% | 2 | 2 | yes |

Drops occur at the **inference queue** (`infer_q` 629 dropped, peak 900), not the capture slot
(`slot_drop` 1). Both runs used a file read at ~180 fps, so these rates are **not**
representative of A100-on-RTSP — they locate the bottleneck, nothing more.

Output timeline: 157 frames at 20 fps = 7.85 s for 42.9 s of content = **5.5× too fast**.

### H.11 Detector capacity: yolo11n vs yolo11m

`tests/calibration/compare_detector_models.py`, `register_file.avi` (2560×1440),
150 consecutive frames, shipped `conf 0.40 / iou 0.60 / imgsz 640`, fresh ByteTrack
per model. **4 people in view.**

| Model | dets | mean/frame | track ids | track lengths | ms/frame (CPU) |
|---|---|---|---|---|---|
| yolo11n (was) | 552 | 3.68 | **6** | 150, 150, 150, **64, 37**, 1 | 215 |
| yolo11m (ships) | 600 | 4.00 | **4** | 150, 150, 150, 150 | 435 |

yolo11m found more boxes on 49 of 150 frames and fewer on 1. The result that matters is
the **id count**, not the box count: yolo11n's ids 4 (frames 0–55) and 6 (frames 80–149)
**never co-occur in any frame** and are separated by a 24-frame gap — one person lost and
re-minted. yolo11m covers that person with a single continuous track. yolo11n also emits a
1-frame phantom id.

So the two changes staged for run2 meet in the middle: **yolo11m creates fewer fragments;
#40 merges the fragments that remain.** Same-camera fragmentation is the mechanism behind
both the operator's cam_213 front/back split and their own diagnosis, *"this seems like a
track id change so reid change nonsense"*.

**What this does NOT establish.** One clip, one camera, 4 people, no frame dropping, and
CPU timings — the 2.0× CPU ratio is not the GPU ratio, and a file run never drops frames.
The live path does: a slower detector means more dropping, which fragments tracks, cutting
against the gain measured here. That half is only answerable from run2's per-camera
`dropped%`. Nor is this clip cam_206, whose missed people are in a crowded room with a
table — its own frozen replay clip is the only thing that reproduces that.

> Precedent worth respecting: Part A rejected `imgsz` 640→1280 as a no-op on this same
> clip with yolo11n. Capacity moved what resolution did not, so `imgsz` deserves
> re-measuring on yolo11m before it is treated as settled.

---

## Part J — Field results, 2026-07-30 (A6000, 4 cameras, live RTSP)

First real production data. **Three of my hypotheses were wrong and one new P0 defect
surfaced.** Source: `run_id 20260730_082045`, 4 cameras, 40 s of metrics, `cuda:0`,
`nvdec_usable: False`, `decode_backend: cpu`.

### J.1 Frame dropping is a NON-PROBLEM — Phase 11 largely dead

| Camera | nominal | read | rendered | dropped |
|---|---|---|---|---|
| cam_213 | 25 fps | 24.2 | 905/968 | **6.5%** |
| cam_224 | 15 fps | 14.2 | 551/567 | **2.8%** |
| cam_206 | 25 fps | 21.8 | 868/872 | **0.5%** |
| cam_219 | 25 fps | 22.7 | 877/908 | **3.4%** |

`infer_q` dropped **10 frames in 40 s**. `slot_drop` ≤ 4. `stale_skipped=0`. Inference
completed 3271 of 3315 frames read, at **44% scheduler utilisation**.

My CPU-on-file measurement said 85–99% (H.10). I flagged it unrepresentative; it is
unrepresentative by a factor of ~20. **Narrowing the ReID lock (#50), the queue depth
(#51), `imgsz` and `conf` all solve a problem that does not exist** at this crowd size.
Phase 11 drops to lowest priority. Caveat retained: the ReID batch scales with people
in frame, so this could change in a genuinely crowded scene.

### J.2 Cross-camera linking is 0-for-19, and the threshold is NOT why

```
x-camera: attempts=19 linked=0 rejected[thresh=0 margin=9 recip=10 topology=0]
histogram:  <.5:0  .5-.6:0  .6-.7:9  .7-.8:10  .8-.9:0  .9+:0
```

**`thresh=0`** — not one attempt was below `cross_camera_threshold: 0.60`. All 19 scored
0.60–0.80 and every one died on the **runner-up margin** (9, at `accept_margin: 0.03`) or
**reciprocal-best** (10).

This inverts the assumption in the config history, which lowered `cross_camera_threshold`
0.63 → 0.60 to let real matches through. The threshold was never the obstacle. Nine
near-ties plus ten reciprocity failures means several candidates are bunched in the same
0.6–0.8 band — exactly the ambiguity `TOP2_MARGIN` was added to measure, and strong
support for prioritising Phase 9's margin questions over its threshold sweeps.

### J.3 Same-camera 0.70 cuts through the middle of the distribution

```
same-cam reacquire: attempts=17 ok=10 rejected_below_thr=7 max_rejected=0.690
histogram:  .5-.6:2  .6-.7:5  .7-.8:7  .8-.9:1  .9+:2
```

Max rejected 0.690, a hair under the bar — the same pattern the config comments record at
0.85 and 0.90. `linked=0` this run, so the two-lane leak did not fire and these 7
rejections are genuine mints rather than double-counted.

### J.4 H.265 corruption does NOT occur on the live feed

**Zero** decode errors across 4 live streams. The 294/682 and 207/1573 broken references
in `test_file.avi` / `test_v2.avi` are an artefact of however those were recorded, not
something happening on RTSP. **Issue #28 downgrades** from "leading hypothesis for random
identity behaviour" to "worth setting `rtsp_transport=tcp` anyway"; #29's blocking-read
problem stands on its own merits.

### J.5 NEW P0 — a failed `print` abandoned the run's ids. Twice.

Two consecutive runs produced clips and metrics, then **nothing**: no final summary, no
reconcile, no decision log, no output video. Both launched as `python main.py ... | tee
run1.log`, which puts python and tee in one foreground process group. Ctrl-C reaches
both; tee dies first; every subsequent `print` raises `BrokenPipeError`. And
`_report(final=True)` runs **before** `_finalize_offline()`, so a cosmetic print failure
skipped the reconcile — with the traceback going into the same dead pipe, so nothing was
visible. A dropped SSH session or closed terminal breaks stdout identically.

`InterruptGuard` does not help: it protects against signals and explicitly does not
swallow exceptions.

**Fixed 2026-07-30** — `_report(final=True)` is now guarded, and stdout/stderr are wrapped
in `_QuietOnBrokenPipe` for the finalization phase so no print can raise. Pinned by
`tests/live/test_shutdown_reaches_reconcile.py` (14 checks), including the
report-raises, dead-stdout, and both-at-once cases plus the ordering invariant.

**Diagnostic trap worth remembering:** `output_cam_*.mp4` existing does **not** mean the
last run succeeded. Those files are only overwritten by a completed render, so stale
outputs from an earlier run look like success. Check their mtime against the `run_id`.

### J.6 Decision-log analysis — the per-camera finding that reframes Phase 9

Full analysis of `logs/reconcile_decisions_20260730_093723.jsonl` (286 decisions) via
`tests/calibration/analyze_decision_log.py`. **Operator ground truth, max concurrent
per camera:** cam_206 ≈ 6, cam_224 ≈ 5, cam_219 ≈ 4, cam_213 ≈ 3, with people
continuously entering and leaving, so total distinct people over the 4.5-minute run
is higher than any single maximum — plausibly 8–15.

Against that, **17 identities is roughly plausible in aggregate**, not obviously
broken. But the specific mis-assignments the operator observed are real regardless of
the count, and the analysis explains them exactly.

**`same_camera_threshold` is a PER-CAMERA problem. A global value cannot work.**

| Camera | subjects | eligible/subject @0.90 | zero eligible |
|---|---|---|---|
| cam_206 | 55 | **9.0** (max 21) | 6 |
| cam_213 | 11 | **0.0** (max 0) | **11 / 11** |
| cam_219 | 6 | 0.3 | 4 / 6 |
| cam_224 | 17 | 0.5 (max 2) | **12 / 17** |

At 0.90, **cam_213 achieves zero same-camera merges out of eleven subjects**, and
cam_224 manages it for only 5 of 17 — while in cam_206 every fragment sees nine
partners above the bar. The same value is harmless in one camera and total in another.
The analyser confirms no global bar exists: p95 of the "top different" score (0.816)
**exceeds** p5 of the "worst same" score (0.719), so the per-subject boundaries
overlap. Median boundary sits between 0.721 and 0.856.

Supporting: **50.6%** of subjects lost a candidate above 0.85, 34.5% above 0.88,
highest rejected 0.899. **Orphans — subjects with no eligible same-camera partner —
are 33/89 (37%)**, ten with a best candidate above 0.85.

This maps directly onto the observed symptoms: cam_213's "correct by his front, reid 7
by his back" is a camera with *no* same-camera merging, so front and back fragments
can never join and each is absorbed cross-camera separately. cam_224's "reid 3 becomes
reid 7 at a bad angle" is the same, with 12 of 17 orphaned. cam_206's 26-tracklet
identity is fragmentation that *self-heals* because its scores clear 0.90.

**`TOP2_MARGIN` must NOT gate — it carries no information.**

```
margins on accepted merges: median=0.0224  p5=0.0017  p95=0.1181
margins on rejected merges: median=0.0186  p5=0.0016  p95=0.0779
```

Near-identical distributions, so a gate on it would be close to random. Note this
supersedes an earlier read of mine: from two records I claimed fragmentation dominates
near-ties; at scale it is 55 fragmentation vs **65 genuine ambiguity**, roughly
balanced. The conclusion holds for a better reason than the one I first gave.
Disagreement between the two margin definitions: **63.9%**.

**`RECIPROCAL_BEST` rejects 31% of decisions**, all cross-camera, at mean 0.720 and
**max 0.905**. Rejecting a 0.905 cross-camera match is very likely rejecting a real
person. Concentrated in cam_206 (40) and cam_224 (35).

**Cross-camera merges:** 52 accepted, min 0.639, **4 below the 0.66 stranger ceiling**
and 10 below 0.70. Those four are the false-merge candidates worth eyeballing.

**Suppression is cheap here:** 8 of 97 tracklets, 0.3% of observations.

### J.7 Confirmed as expected

`written=0` on every camera (#63, cosmetic). Software H.265 decode for all four streams
and still keeping pace (#53). The store held **4157 points before this run**, and
`_gather_tracklets` scrolls the whole collection filtering by `run_id` in Python — issue
#20 now has a real number attached, and it grows every run.

### J.8 Operator observations from watching the output videos

Ground truth on the *symptoms*, which no metric replaces. Recorded verbatim in substance
because these are what "working" has to mean.

| Camera | Observation | Mechanism |
|---|---|---|
| cam_224 | "definitely sped up" | 14.8 fps tagged at 20 → plays 1.35× fast (#45/#46) |
| cam_224 | reid 3 → **reid 7** when the person moves at a bad angle | orphan fragment absorbed into the wrong cross-camera cluster; 12 of 17 subjects orphaned here (J.6) |
| cam_219 | "a little slowed"; reid 2 → **reid 4** when he moves to the other side of the table | 24.2 fps tagged 20 → 1.21× slow; the id change is a track-id change that reconcile failed to re-merge |
| cam_213 | reid 2 correct **by his front**, becomes **reid 7 by his back** | cam_213 achieves ZERO same-camera merges at 0.90, so front and back fragments can never join. Compounded by reconcile comparing prototype *means* (#45a) |
| cam_206 | 5 people present but only 1 detected at first; "randomly starts detecting the rest" | **detection recall**, not identity. Nothing downstream can invent a box YOLO never produced. Also explains cam_206's 61 tracklets |

**Responses now staged for run2:** the cam_213 / cam_224 rows are what #40 addresses; the
cam_206 recall row is why `detector.model` moved to `yolo11m` (H.11). The cam_224 and
cam_219 playback-speed rows are **not** addressed — they are #45/#46 and still open, so
expect the same "sped up / slowed" complaint from run2's videos.

Operator's own diagnosis, which was correct: *"this seems like a track id change so reid
change nonsense which should not always be the case."* Final-video ids come from
`gid_map[(camera, track_id)]`, so a new track id is looked up independently — reconcile's
entire job is to re-merge those, and it is failing.

### J.9 Raw metrics, run `20260730_093723` (final, t=+130s)

The run the analysis is based on. ~4.5 min, 4 cameras, `cuda:0`, `nvdec_usable: False`.

| Camera | read | rendered | dropped | slot_drop |
|---|---|---|---|---|
| cam_213 | 3227 (24.8 fps) | 3133 | **2.9%** | 30 |
| cam_224 | 1923 (14.8 fps) | 1909 | **0.7%** | 0 |
| cam_206 | 3135 (24.1 fps) | 3123 | **0.4%** | 7 |
| cam_219 | 3163 (24.3 fps) | 3126 | **1.2%** | 10 |

Even lower than the 082045 run in J.1. `scheduler: batches=3119 frames=11398
avg_batch=3.65 util=46% stale_skipped=0`. `infer_q=7 dropped / peak 2`,
`identity_in=0 / peak 4`, `identity_fair_drop=0`, `inference_done=11370`.

**Live engine counters** (off the product path, but they characterise the score
distributions):

```
identity: minted=16 reacquired=18 linked=2 active=16 stored=1500
x-camera:   attempts=15 linked=2 rejected[thresh=4 margin=5 recip=4] max_subthreshold=0.599
same-cam:   attempts=23 ok=18 rejected_below_thr=5 max_rejected=0.699
coactive_vetoes=49   topology_pruned=0
histograms (<.5 .5-.6 .6-.7 .7-.8 .8-.9 .9+)
  same-cam reacquire:  0  1  4  4  2 12      <- mass well above 0.70; 0.70 works here
  cross-camera:        0  4  7  4  0  0      <- nothing above 0.80
```

Note `max_subthreshold_score=0.599` — four cross-camera matches missed the 0.60 bar by a
thousandth. And unlike the 082045 run (0 links from 19 attempts), this one linked 2 of 15.

**Per-camera tracklets and observations** (from the reconcile diagnostics):

| Camera | tracklets | total obs | min | max | mean |
|---|---|---|---|---|---|
| cam_206 | **61** | 1065 | 1 | 203 | 17.5 |
| cam_213 | 11 | **85** | 3 | **13** | 7.7 |
| cam_219 | **7** | 1848 | 1 | 722 | 264.0 |
| cam_224 | 18 | 1211 | 2 | 355 | 67.3 |

**cam_213 is starved**: it read the *most* frames of any camera yet produced the *fewest*
observations (85, longest track 13 ≈ 4 s at `reid.interval: 10`). Either almost nobody was
in view, or the crop-quality gate rejects nearly everything there. Untested.

**Store:** 4209 observations for this run → 17 identities, 11 cross-camera.

### J.10 Full analyser output, run `20260730_093723`

286 decisions (8 suppressed), 97 outcomes; phase 1 = 89, phase 2 = 189.

```
gate failures : MIN_OBSERVATIONS 8, ABSOLUTE_THRESHOLD 92, RECIPROCAL_BEST 86
exclusions    : BELOW_ABSOLUTE_THRESHOLD 5576, TEMPORAL_CONFLICT_SAME_CAMERA 1008,
                NOT_MERGEABLE_CROSS 720

1. WHERE THE BAR CUTS (same-camera, n=87 subjects)
   highest score REJECTED per subject: mean=0.821 p50=0.853 p95=0.898 max=0.899
     lost a candidate above 0.75: 64/87 (73.6%)
     lost a candidate above 0.80: 55/87 (63.2%)
     lost a candidate above 0.85: 44/87 (50.6%)
     lost a candidate above 0.88: 30/87 (34.5%)
   natural boundary from 77 bimodal subjects:
     gap LOWER edge (top 'different'): mean=0.721 p95=0.816 max=0.841
     gap UPPER edge (worst 'same')   : mean=0.856 p5=0.719  min=0.609
   => boundaries OVERLAP: no global bar works (p95 lower 0.816 > p5 upper 0.719)

2. ORPHANS: 33/89 (37.1%) with zero eligible same-camera partners
   their best rejected candidate: mean=0.795 p95=0.903 max=0.908
   10/33 had a best candidate above 0.85
   by camera: cam_206 6, cam_213 11, cam_219 4, cam_224 12

3. NEAR-TIES (margin <= 0.05): 120 total
   fragmentation (runner similarity >0.85): 55 (45.8%)
   genuine ambiguity (<=0.85)            : 65 (54.2%)
   margins accepted: n=91 p5=0.0017 median=0.0224 p95=0.1181
   margins rejected: n=72 p5=0.0016 median=0.0186 p95=0.0779   <- indistinguishable
   margin-definition disagreement: 124/194 (63.9%) [+84 no selectable candidate]

4. RECIPROCAL_BEST: 86/278 (30.9%), ALL cross_camera
   scores rejected: mean=0.720 p95=0.862 max=0.905
   by camera: cam_206 40, cam_224 35, cam_213 22, cam_219 11

5. ELIGIBLE-SET SIZE @0.90
   cam_206: 55 subjects  mean=9.0  max=21  zero=6
   cam_213: 11 subjects  mean=0.0  max=0   zero=11
   cam_219:  6 subjects  mean=0.3  max=1   zero=4
   cam_224: 17 subjects  mean=0.5  max=2   zero=12

6. CROSS-CAMERA ACCEPTED: n=52 min=0.639 mean=0.801 max=0.937
   below 0.70: 10/52    below 0.66 (stranger ceiling): 4/52
   observed accepted scores: 0.937 ... 0.733 0.721 0.704 0.692 0.690 0.688 0.655 0.639

7. SUPPRESSION: 8/97 tracklets, 0.3% of observations
   by camera: cam_206 6, cam_219 1, cam_224 1

8. OUTCOME: 17 identities, tracklets/identity mean=5.24 max=26
   distribution: [26, 10, 8, 7, 5, 5, 5, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1]
   spanning >1 camera: 11/17
```

### J.11 Worked decision-log examples

Four real records, kept because they show the distributions concretely and are the
clearest single argument for #40.

**`U-0000`** (cam_206, 18 obs, frames 1–155). 54 candidates scored, **3 eligible**,
margin 0.0689. Sorted scores:

```
0.977  0.908  0.906  |  0.897  0.876  0.875  |  0.662  0.633  0.631  0.621 ... 0.495
       ELIGIBLE      |   REJECTED at 0.90    |    genuinely different people
```

A clean gap between **0.875 and 0.662**, with the bar cutting *inside* the same-person
cluster and discarding three genuine merges. A bar in ~(0.68, 0.87) would be perfect here.

**`U-0001`** (cam_206, 10 obs). **16 eligible candidates**, all above 0.90:
`0.9718 0.9694 0.9623 0.9571 0.9457 0.9451 0.9447 0.9367 0.9319 0.9289 0.9213 0.9207
0.9199 0.9185 0.9120 0.9056`. Mutual `pair_similarity_to_best` 0.92–0.97, so not a chain.
**One person, seventeen tracklets, one camera.** Top-2 margin **0.0024** — a near-tie
between two fragments of the same person, which is why a margin gate would be harmful.

**`U-0002`** (cam_206, 23 obs). 4 eligible, margin 0.0069. `U-0000` scored **0.8757**
against it and was rejected — yet both merge with `U-0017`, so they end up in the same
cluster transitively anyway. The 0.90 bar blocks the direct edge but not the outcome.

**`U-0003`** (cam_206, **203 obs**, frames 371–2379). Only 1 eligible candidate;
`TEMPORAL_CONFLICT_SAME_CAMERA: 8` — eight other tracklets overlapped it in time, so the
temporal veto is working. `margin_eligible: null`, `margin_all_scored: 0.3405`.

### J.12 Identity composition, run `20260730_093723`

The 11 cross-camera identities, as reconcile assigned them. Kept so a re-run can be
diffed against it rather than re-derived.

```
GID  1: cam_206(5,8,13,79,81,93) + cam_224(185)
GID  2: cam_206(7,19,21,30,49,54,58,66,76,83,89,96,97,131,137,138,139,140,157,162,165,210)
        + cam_213(26) + cam_219(6) + cam_224(3,43)          <- 26 tracklets, 22 in cam_206
GID  3: cam_206(80) + cam_213(120) + cam_219(143,173) + cam_224(126)
GID  4: cam_206(82) + cam_213(38) + cam_219(48) + cam_224(2,52)
GID  5: cam_206(92) + cam_213(86) + cam_224(12)
GID  7: cam_206(99,104,107,112,114,127) + cam_213(33) + cam_224(159)
GID  8: cam_206(108) + cam_224(132)
GID 11: cam_206(176,192) + cam_213(180,190) + cam_224(183)
GID 12: cam_206(195) + cam_213(116,193) + cam_224(121,189)
GID 15: cam_213(115) + cam_219(118) + cam_224(117)
GID 16: cam_219(4) + cam_224(1,20,202)
```

GID 2's 22 cam_206 members are all **mutually time-disjoint** (reconcile refuses to merge
time-overlapping same-camera tracklets) and each cleared the **0.90** bar, so it is almost
certainly one person entering and leaving that view 22 times — not a false merge.

The operator-reported failures map onto this: cam_213's front/back split is GID 2's
`cam_213(26)` versus GID 7's `cam_213(33)`; cam_224's bad-angle switch is GID 3's
`cam_224(126)` versus GID 7's `cam_224(159)`; cam_219's across-the-table switch is GID 2's
`cam_219(6)` versus GID 4's `cam_219(48)`.

---

## Part K — Library internals verified by reading source

Facts established by reading the installed packages, not from documentation or memory.
Recorded so the next session does not re-derive them. Versions: `ultralytics 8.4.89`,
`torch 2.12.1`, `qdrant-client 1.18.0`, `opencv 5.0.0`.

### K.1 ByteTrack (`ultralytics/trackers/`)

`cfg/trackers/bytetrack.yaml` ships: `track_high_thresh: 0.25`, `track_low_thresh: 0.1`,
`new_track_thresh: 0.25`, `track_buffer: 30`, `match_thresh: 0.8`, `fuse_score: True`.

- `byte_tracker.py:312-315` splits detections by those two thresholds. Since the detector
  pre-filters at `conf=0.4`, the **second-association pool (0.1–0.25) is always empty**, so
  ByteTrack's occlusion-recovery stage never runs. Mechanism real; measured as a no-op on
  available footage (Part A).
- `_format_output` (`:486-488`) returns one row per track that is in `tracked_stracks`
  **and** `is_activated`, and association is a one-to-one linear assignment. **The tracker
  cannot invent a box** — confirmed empirically across 120 frames, 0 with more rows than
  detections. Duplicate boxes therefore require duplicate *detections*.
- `activate()` (`:108-109`) sets `is_activated` immediately only when `frame_id == 1`.
  Otherwise a new track needs a **second consecutive match** before it appears in output,
  so a one-frame duplicate detection is filtered but a persistent one is not.
- `multi_predict` advances the Kalman filter **one step per `update` call with no `dt`**, so
  under irregular frame gaps the predicted box is wrong by however much real time passed.
- `on_predict_start`/`register_tracker` attach the tracker to the *predictor*, so tracker
  state is per-`YOLO`-object. One `PersonDetector` per camera therefore isolates state
  correctly (verified).

### K.2 Ultralytics predict defaults (`cfg/default.yaml`)

`imgsz: 640`, `iou: 0.7`, `max_det: 300`, `agnostic_nms: False`, `device:` unset (auto).
The pipeline never passes `imgsz` or `device`, so 640 and auto-select apply.

### K.3 torchreid OSNet feature tap

`model.fc == Sequential(Linear(512,512), BatchNorm1d(512), ReLU())`, and eval-mode
`forward()` returns `self.fc(v)` — i.e. **post-ReLU**. Measured consequence: 21.0% of dims
exactly zero, 0% negative, so cosine between any two embeddings is ≥ 0 and the usable
range is compressed upward. Post-BN is Phase 8; changing the tap voids every threshold.

### K.4 Qdrant client

`QdrantClient` accepts `prefer_grpc` (default `False`), `grpc_port` (default 6334) and
`grpc_options`. `scroll()` accepts `scroll_filter` — **currently unused everywhere**, which
is why `_gather_tracklets` pulls the entire collection and filters in Python (#20).
`docker-compose.yml` **already maps 6334**, so gRPC needs no infrastructure change.
Embedded `path=` mode is in-process and has no gRPC.

### K.5 Verifier weights (batch path, `identity/verifier.py`)

`w0=-13.5`, `w_cosine=11.0`, `w_rerank=7.0`, `w_observation_count=0.6`,
`w_time_since_last=-0.01`, `w_bbox_quality=1.0`. Because the observation-count term is
`0.6 · log1p(count)`, unbounded and never decayed, the minimum accepted cosine falls:

| observations | 0 | 10 | 30 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|---|---|
| min accepted cosine | 0.733 | 0.653 | 0.619 | **0.580** | **0.543** | **0.503** | **0.466** |

Different people top out around 0.57, so past ~100 accumulated observations the busiest
identity starts accepting strangers. Separately, `min_prob_gap: 0.05` saturates: at 300
observations, cosine 0.90 vs 0.85 gives a probability gap of **0.0019**, so a clean winner
is rejected and a duplicate minted. Batch path only, hence Part C — but do not reuse these
weights anywhere without re-deriving them.

---

## Part L — Handover notes (2026-07-31)

Things a new agent needs that are not item numbers.

### L.1 Where the code now lives

| Concern | File | Notes |
|---|---|---|
| Merge decisions | `src/identity/reconcile.py` | scoring modes, per-camera bars, both physical guards, fit/margin |
| The record of every decision | `src/identity/decision_log.py` | read by `analyze_decision_log.py` |
| Offline sweep | `tests/calibration/sweep_reconcile_thresholds.py` | read-only by construction |
| Offline re-render | `tests/calibration/rerender_from_clips.py` | needs the `.annotations.json` sidecar |
| Detector comparison | `tests/calibration/compare_detector_models.py` | produces H.11 |
| Box geometry sidecar | `src/live/render.py::_write_annotations` | written when the clip finalises |

Two resolvers exist so the live path and the file-batch path cannot drift, the same
argument as `detector.resolve_detector_cfg`: `resolve_same_camera_thresholds` and
`resolve_covisibility`. Both call sites use them. Add config the same way.

### L.2 Invariants worth not breaking

1. **Fail-soft in finalization.** Nothing between `Ctrl-C` and `[live] shutdown
   complete.` may raise. A cosmetic `print` once cost two runs their identities
   (J.5); the decision log, the geometry sidecar and the disk cap all degrade
   rather than throw.
2. **Fail-open on physical vetoes.** An unlisted camera pair, a co-visible pair or
   a missing timestamp all mean "no veto". A wrong veto manufactures a second
   identity for someone who has one, which is worse than a missed one.
3. **Read-only means read-only.** The sweep and the re-render take reconcile's
   RETURNED remap and never let it write ids into the store, so two settings run
   back to back cannot contaminate each other.
4. **The decision log records the round AS SCORED.** Do not feed it post-merge
   state; a bar logged beside a score it never applied to is worse than no log.

### L.3 Traps that have already caught someone

- **`| tee`** — see §0.2. Also `tail -f` and `kill` in one block: the interrupt
  goes to `tail`.
- **`output_cam_*.mp4` existing does not mean the last run succeeded.** They are
  only overwritten by a completed render, so stale files look like success. Check
  mtime against the `run_id`.
- **Placeholders in shell commands.** `<run_id>` and `you@host` were both pasted
  literally, and `<` is a shell redirect. Write commands that derive their own
  values: `RUN=$(grep -o 'run_id=[0-9_]*' run.log | tail -1 | cut -d= -f2)`.
- **Test fixtures that do not test what they claim.** Two caught in one session: a
  "stranger" built in the same plane as the true partner was 0.966 similar to it,
  and 0.5s tracklet spans could never exceed a 1.0s veto tolerance, so the veto
  "passed" without ever being exercised. Assert the fixture's own preconditions.
- **Editing a `__init__` by inserting methods mid-body** orphans every assignment
  after them. `verify_embedding_contract.py` caught it in seconds; run it after
  touching anything in `src/reid/`.

### L.4 The operator's own words, as ground truth

Symptoms are reported by watching, not by metrics. Recorded verbatim in substance
because these are what "working" has to mean:

- *"reid 2 correct by his front, becomes reid 7 by his back"* (cam_213)
- *"reid 3 becomes reid 7 when the person moves at a bad angle"* (cam_224)
- *"reid 1 becomes reid 6 when he leaves the cam and comes back the other side of
  the room"* (cam_219, same camera)
- *"multiple other people in the room are also called reid 6"* — a false merge
- *"cam_224 definitely sped up"*, *"cam_219 a little slowed"* — #45/#46, now fixed
- *"5 people present but only 1 detected at first"* (cam_206) — detection recall,
  which is why the detector moved to yolo11m

Their own diagnosis, which was correct: *"this seems like a track id change so reid
change nonsense which should not always be the case."*

---

## Part M — The cam_219 split: why reconcile *cannot* merge reid 1 and reid 11

Run `20260731_060425` (4 cameras, 80 s, 2216 observations, 69 tracklets → 25 identities).

### M.1 The symptom

> *"cam 219: someone is assigned reid 1 for first 15 seconds, then switches to reid 11 and
> he persists as reid 11 across cameras, only when he sits in the angle for the first 15
> seconds he is back to reid 1."*

This is the fourth report of the same failure class (L.4: *"reid 2 by his front, reid 7 by
his back"*, *"reid 3 becomes reid 7 at a bad angle"*, *"reid 1 becomes reid 6 when he comes
back the other side of the room"*). Every previous attempt moved a threshold and was
reverted. **The threshold is not what is blocking this merge.**

### M.2 What the run's own output already proves

The two clusters the operator is seeing, from the run summary:

| | cam_206 | cam_213 | cam_219 | cam_224 |
|---|---|---|---|---|
| **GID 1** | 12 | 51 | **7 + 113** | 30 |
| **GID 11** | 48 | 79 + 88 | **46** | 3 + 68 + 102 |

His sitting fragments (219/7 + 219/113, merged in Phase 1 at cosine 0.969) are in GID 1.
His standing/walking fragment (219/46) is in GID 11. For the operator's complaint to go
away, those two clusters must merge. Reconcile refuses, for **two independent and
individually sufficient reasons**, neither of which is the cross-camera threshold.

#### Reason 1 — the merge is judged as a SAME-camera claim, at 0.90

`pair_threshold` / `pair_bar_now` (reconcile.py): if two clusters share *any* camera, the
bar becomes `strictest_same_camera_bar` over *all* shared cameras. GID 1 and GID 11 share
**all four**, so the bar is

    max(cam_206 0.90, cam_213 0.80, cam_219 0.90, cam_224 0.80) = 0.90

The 0.63 cross-camera threshold is never consulted. Two clusters of 5 and 7 tracklets are
compared by the cosine of their *cluster means* — a mean of a mean, over multiple people
and multiple views — and asked to clear **0.90**. That number is unreachable by
construction. Worse, the rule is monotone in the wrong direction: **as clusters grow, the
score falls (means average out) while the bar rises (more shared cameras → strictest).**
Merging gets harder the more evidence accumulates.

#### Reason 2 — the pair is very likely never scored at all

`mergeable_cross` calls `conflict()` **before** scoring, and `conflict()` is an all-pairs
hard veto over the full member cross-product. For GID 1 × GID 11 that is 5 × 7 = **35
member pairs**, all of which must pass:

- **8 same-camera pairs** need fully disjoint frame envelopes (206: 12↔48; 213: 51↔79, 51↔88; 219: 7↔46, 113↔46; 224: 30↔3, 30↔68, 30↔102)
- **7 cross-camera pairs** are exempt (cam_219↔cam_224 is declared `covisible`)
- **20 cross-camera pairs** are vetoed on more than **1.0 s** of wall-clock overlap

One failure anywhere in that grid removes the pair from `root_scores` entirely: no score,
no log line, no gate record, and **nothing any threshold, scoring mode or reciprocal-best
setting can reach.** The run's own diagnostics say this dominates: 1537 + 1961 candidate
exclusions from the two temporal gates against 314 decisions, and the analyser's own
verdict, *"TOP2_MARGIN disagreement 79.21% (hard constraints reshaping the candidate
space)"*.

Which of the 35 pairs actually fires is the **one fact still missing**, and M.0 below
recovers it.

### M.3 How he got split in the first place

The lock-in above only matters because the two halves were separated earlier:

1. **cam_219's same-camera bar is 0.90** (it is absent from `per_camera`, so it inherits
   the global). Config's own note already flags cam_219 as orphaning 4 of 6 subjects and
   says it was left at 0.90 only to avoid confounding a comparison.
2. **`scoring: prototype`** — mean-vs-mean. Sitting at an angle is a different appearance
   *mode* from walking, and this file already documents at length (§#45a, config lines
   422-455) why a mean is the wrong summary: *"their mean sits between them, matching
   NEITHER view"*. Measured on the fixture: one person's two modes score **0.640** while
   two strangers score **0.800**. So 219/7 vs 219/46 never came close to 0.90 in Phase 1.
3. **Phase 2 could not rescue it either.** While both halves are still single-camera
   cam_219 clusters, `mergeable_cross` requires at least one cross-camera member pair, so
   the pair is `NOT_MERGEABLE_CROSS` (1564 exclusions this run). By the time each half has
   absorbed a foreign camera, Reason 1 sets the bar back to 0.90 and Reason 2 has arrived.
   **The only window in which this merge was ever possible was Phase 1, at 0.90, on
   prototype means.**
4. **Then each half was captured by a different cluster through a weak link.** From the
   run log: GID 11 absorbed cam_206/48 at **0.649** and cam_206/27+cam_219/8 at **0.677**;
   GID 1 absorbed cam_206/12 at **0.818**. The measured different-person ceiling for this
   footage is **0.661** (H.4, and `STRANGER_CEILING` in the sweep). A cross bar of 0.63
   sits *below* it, so at least one of those links is a stranger — and a cluster's hard
   constraints are the **union** of its members' constraints, so **one wrong member
   permanently removes a correct merge.** Merge order decides who gets poisoned.

### M.4 Why five rounds of tuning could not have worked

- Reason 1 is arithmetic on the *shared-camera* rule, not on the cross bar. Lowering
  `threshold` from 0.70 to 0.63 (and back) never touched it.
- Reason 2 is a hard veto evaluated *before* scoring. It is threshold-immune by design.
- And the feedback loop was measuring a different algorithm: **`sweep_reconcile_thresholds.py`
  and `rerender_from_clips.py` passed neither `covisibility` nor
  `same_camera_reciprocal_best` to `reconcile_tracklets`, both of which default OFF and are
  ON in production.** Every threshold conclusion in Part J was drawn from a run with the
  cross-camera veto disabled — i.e. from a clustering that does not ship. (Fixed in M.1.)

### M.5 The plan

Ordered so that each step is verifiable before the next one changes any decision. **No
step tunes a threshold.** Rule D9 still holds: one change, re-run offline, diff, and watch
the video before it ships.

#### M.0 — Recover the missing fact (read-only, no behaviour change) — **DONE, needs running**

`tests/calibration/explain_merge_failure.py <run_id> <id_a> <id_b>` re-runs reconcile
read-only at the **production** settings and then prints, for the two resulting clusters:
membership with observation counts and wall-clock spans, the bar that applies and why, the
pair score under all three scoring modes, and **the full 35-pair conflict grid with the
overlap in seconds for every pair** — i.e. the witness. Verdict line is one of
`BLOCKED BY <gate> at <pair>` or `SCORED <s> BELOW BAR <bar>`.

*Acceptance:* run on `20260731_060425` for ids 1 and 11, and the output names the blocking
pair. **Everything below is chosen by what it prints.**

#### M.1 — Make the offline loop reproduce production — **DONE**

- `identity.reconcile.resolve_reconcile_kwargs(cfg)` is now the single place every
  reconcile setting is read; `main.py`, `LivePipeline` and both calibration tools use it.
  This also kills two live drifts: `main.py` defaulted `min_tracklet_observations` to **1**
  where the live path used **3**, and fell back to `identity.threshold` **0.85** where the
  live path used **0.63**.
- Both tools now print `describe_reconcile_kwargs(...)` — one line naming the cross bar,
  the per-camera bars, the scoring mode, and all three switches — so a sweep can never
  again be read as production.
- `--same-reciprocal` now exists (config.yaml has been telling the operator to run it since
  #45a; it was never parsed, so the test it documents measured *nothing changing*).
- `--no-covisibility` added, for measuring the veto's cost deliberately rather than by
  accident.

*Acceptance:* `sweep_reconcile_thresholds.py 20260731_060425` with no flags reproduces the
live run exactly — **25 identities, 7 spanning >1 camera**, same composition. That equality
is the proof the loop is trustworthy. Until it holds, no other measurement counts.

#### M.2 — Same-camera veto: occupancy, not envelope

`_spans_disjoint` compares `(min_frame, max_frame)` of the **sampled** observations
(`reid.interval_sec: 0.4`, so ~10 frames apart). A tracklet with a hole — occlusion, or a
duplicate box on one body — claims its entire envelope, so a fragment living *inside* that
hole is declared "provably a different person" and hard-vetoed forever. The axiom *"one
body cannot be two simultaneous tracks"* is also simply false under duplicate detection,
and cam_206 produced 42 tracklets for ~5 people this run.

Replace with an interval set: each observation contributes `[frame - h, frame + h]` for
`h ≈ ` that camera's sampling period; union adjacent intervals; conflict only on
**sustained** intersection (≥ 2 sample periods), never a single instant. Two genuinely
co-present people overlap densely, so the real guard is untouched; a phantom envelope and a
one-frame duplicate stop vetoing. Fail-open behaviour unchanged.

*Acceptance:* unit tests for (a) a gap-nested pair that must NOT conflict, (b) a densely
co-present pair that must, (c) a single-instant duplicate that must not. Then re-run M.0
and diff the grid.

#### M.3 — Judge the shared-camera claim on the fragments it is about

Today: shared camera → strictest bar over *all* shared cameras → applied to the *cluster
mean* score. Both halves of that are wrong (M.2 Reason 1).

Change to: for each shared camera `c`, the same-camera claim is tested **within `c`** — the
best score between A's members in `c` and B's members in `c` must clear `c`'s bar — while
the cross-camera part keeps the cross bar. This is exactly the claim the docstring says it
wants to make, applied to the evidence that claim is about, and camera-disjoint clusters
behave exactly as today. Keep `strictest_same_camera_bar`, but apply it per camera instead
of collapsing four cameras into one number over one global score.

*Acceptance:* GID 1 × GID 11 is judged at cam_219's bar on 219/7 ↔ 219/46 (a real
comparison) instead of 0.90 on a mean of means. Re-render and watch cam_219's first 15 s.

#### M.4 — Stop the poisoning

In order of cost:

1. **Restore the cross bar to 0.70.** Not new tuning: config's own comment records it as
   *measured* on run `20260730_111232` and *"the only free move in the whole sweep"* — 0
   merges below the stranger ceiling, cross-camera identities unchanged. Commit `a77895fe`
   ("revert: back to run 2's reconcile settings") put the value back to 0.63 and **left the
   comment claiming 0.70**. Re-confirm with the now-faithful sweep, then set value and
   comment together.
2. **A cluster must not absorb a member that fits it worse than its existing members fit
   each other** — refuse a merge whose score is below the absorbing cluster's own internal
   cohesion (its minimum internal pairwise score). This removes weak-edge capture without
   moving any bar, the same argument that made reciprocal-best worth more than a sweep.
3. Only if M.0's grid says merge *order* is what costs identities: replace greedy union-find
   with must-not-link-constrained agglomerative clustering that re-evaluates. Large; do not
   start it in this pass.

*Acceptance:* the 0.649 and 0.677 links are gone and cross-camera identities do not drop.

#### M.5 — Fix the cause: scoring mode

The split originates in a mean-vs-mean comparison of two appearance modes, and this repo has
already measured the fix (`consensus`: same person 1.000, strangers 0.800, one bad crop only
0.880). Changing the mode **voids every bar**, so it ships as one operation: sweep
`--scoring consensus` over both axes with M.1's faithful settings, pick the pair of bars,
re-render, watch cam_219, then set mode *and* bars together in one commit.

#### M.6 — Regression test

`tests/live/test_reconcile_two_halves_rejoin.py`: one person with two appearance modes in
cam_219, two strangers co-present in cam_206, cam_224 covisible. Assert the two halves land
in one identity **and** assert the fixture's own preconditions (L.3's trap: that the
stranger really is a stranger, and that the spans really do overlap). Plus the M.2 unit
tests.

### M.7 Loose code found while diagnosing

| # | Finding | Status |
|---|---|---|
| M-a | Sweep + re-render passed neither `covisibility` nor `same_camera_reciprocal_best` — the offline loop measured a non-shipping algorithm | **fixed** (M.1) |
| M-b | `main.py` vs `LivePipeline`: `min_tracklet_observations` 1 vs 3, threshold fallback 0.85 vs 0.63 | **fixed** — one resolver |
| M-c | `--same-reciprocal` documented in config.yaml, never implemented; the operator would have measured a no-op | **fixed** |
| M-d | `_arg()` in both tools raises `IndexError` on a trailing boolean flag, and a boolean flag silently consumes the next flag as its value | **fixed** — shared `arg()` / `flag()` in `_common.py` |
| M-e | `identity.reconcile.threshold` is 0.63 under a comment whose first line says it was *raised to 0.70 and measured*. Stale narrative beside a live number is exactly the L.3 trap | **comment corrected**; the value is M.4.1 |
| M-f | Decision log records `TEMPORAL_CONFLICT_CROSS_CAMERA` as `_na_gate()` ("does not apply") in Phase 2 while candidates are being excluded by it — the gate-failure histogram can never show the veto that dominates the run | **fixed** — real count recorded |
| M-g | `conflict_reason` returns only the gate name and discards the witness pair, so "why did these two not merge" is unanswerable from the log | M.0 recovers it externally; plumb the witness into the log with M.2 |
| M-h | Only cam_224↔cam_219 is declared `covisible`, but all four cameras overlap, and `ts` is *receive* time (#15) carrying jitter and queue delay. 1961 cross-camera exclusions this run, each of which manufactures a second identity if wrong (violates invariant L.2.2) | **open** — decide from M.0's overlap numbers |
| M-i | `src/interrupt_guard.py:27` has `Ctrl-\ ` in the module docstring → `SyntaxWarning: invalid escape sequence` on every single run | **fixed** — raw docstring |
| M-j | `deploy.sh` target is `/home/you/seifer_work/...` while the run lives in `/home/easemyai/seifer_work/...` — the literal `you` placeholder L.3 warns about; `./deploy.sh` may be pushing to a directory nobody runs | **open** — operator to confirm |
| M-k | The file-batch path builds no `DecisionLog` at all (live only), so a `--mode file` reconcile is undiagnosable | **open**, low priority |
| M-l | Phase 2 `best_partner` picks partners with the snapshot bar while the merge re-checks `pair_bar_now`, so a cluster can pick a partner it will then be refused, wasting the round | **open** — becomes trivial with M.3 |

### M.8 What to run on the GPU box

Two commands, both read-only, both on the already-captured run — no camera time:

```bash
RUN=20260731_060425
python tests/calibration/explain_merge_failure.py $RUN 1 11      # M.0: the witness
python tests/calibration/sweep_reconcile_thresholds.py $RUN      # M.1: must reproduce 25 / 7
```

Send back both outputs in full. The first decides between M.2, M.3 and M.4; the second says
whether the measurement loop can be trusted at all.

### M.9 MEASURED, 2026-07-31 — both commands run on `20260731_060425`

**The measurement loop is now trustworthy.** The sweep's shipped row reproduces the live run
exactly: 25 identities, max cluster 7, 7 spanning >1 camera. M.1 holds.

#### M.9.1 The defect, in two numbers

cam_219, one person, same camera:

| pair | apart | pose | score | cam_219 bar | outcome |
|---|---|---|---|---|---|
| 219/7 ↔ 219/113 | **45 s** | same (sitting) | **0.969** | 0.90 | merged |
| 219/7 ↔ 219/46 | **4 s** | different (sitting → walking) | **0.830** | 0.90 | **REFUSED** |

**The bar sits inside one person's own pose variation.** Time apart is irrelevant; pose is
everything. That single refusal is the whole symptom — everything else in M.2/M.3/M.4 is
its consequence. And cam_219's 0.90 is the one bar in the config with **no measurement
behind it**: it is there because it was *"deliberately LEFT at the global 0.90 ... to avoid
confounding this run's comparison"*.

His timeline in cam_219 is `reid 1` (8.4-13.8 s, sitting) → `reid 11` (18.2-37.5 s, walking)
→ `reid 1` again (58.3-74.7 s, sitting again). The operator reported the first two; the
third confirms the mechanism is pose, not time.

#### M.9.2 What poisoned the cluster, and the correction to M.2/M.4

`reid 11` is a clean single trajectory (224/3 → 219/46 → 224/68 → 213/79 → 213/88 →
224/102, 11-70 s, 199 observations). `reid 1` is his two sitting fragments **plus three
foreign members**, and the three hard vetoes all come from the foreign ones:

| blocker | verdict | admitted to reid 1 at |
|---|---|---|
| 213/51 ↔ 219/46, 213/51 ↔ 224/3 | 1.6 s wall-clock overlap > 1.0 s tolerance | **0.757** (via 206/12) |
| 224/30 ↔ 224/3 | 0.8 s, 3 of 3 sampled instants co-occur — real | **0.760** (via 206/12) |

Both blockers are scraps: 213/51 is 6 observations over 1.6 s, 224/30 is **3 observations
over 0.8 s**. Neither is the operator's person, and 213/51 ↔ 213/88 scores 0.675, so the
cross-camera veto is probably *correct* that 213/51 is somebody else. **The veto's unit is
the cluster; the truth's unit is the member.** A 1.6-second scrap admitted on a 0.757 link
is vetoing on behalf of the 60 observations of the real person it was filed with.

**Two corrections to what M.2-M.4 assumed:**

1. **M.2 (occupancy vs envelope) is NOT the fix here.** Zero phantom overlaps: every
   blocking overlap is real co-presence, and 224/30's three sampled instants all fall inside
   224/3. The envelope bug is real but is not what blocks this merge. M.2 drops to
   opportunistic. (It still needs the duplicate-box question answered: 3 observations fully
   inside a 45-observation tracklet could be a second box on one body, which the
   `.annotations.json` sidecar could settle by IoU. Worth a tool addition, not a fix.)
2. **Thresholds DO reach the blockers, via membership.** I had this wrong: the *pair* is
   threshold-immune, but cluster *membership* is not. At cross 0.80 both 0.757 and 0.760
   links fail, reid 1 becomes `{206/12, 219/7, 219/113}`, and **all three vetoes disappear**
   — every remaining pair in the grid is `ok`. The merge becomes possible, then still fails
   on cam_219's 0.90 vs 0.830. So M.4.1 and M.3 are complementary, not alternatives.

#### M.9.3 M.4.1 reconfirmed under production settings

The sweep, now with covisibility and Phase-1 mutual-best ON as production runs them:

| cross | ids | max | multi | xmerges | below ceiling | min x |
|---|---|---|---|---|---|---|
| **0.63** (shipped) | 25 | 7 | 7 | 14 | **1** | 0.649 |
| **0.70** | 27 | 6 | **7** | 12 | **0** | 0.732 |
| 0.75 | 28 | 6 | 7 | 11 | 0 | 0.757 |
| 0.80 | 31 | 6 | 6 | 8 | 0 | 0.818 |

0.70 remains the free move: the sub-ceiling merge goes away and cross-camera identities do
not drop. This is the first time that result has been produced by a sweep that matches
production. 0.80 additionally sheds both poisoning scraps at the cost of `multi` 7 → 6.

#### M.9.4 M.5 (scoring mode) is now DOUBTFUL — do not ship it on the old evidence

Cluster vs cluster, reid 1 × reid 11: prototype **0.873**, max_exemplar 0.873, consensus
**0.653**. Consensus scores the pair *far lower*, the opposite of the synthetic fixture's
prediction — because these are two multi-person clusters with few matching view pairs, so
"the top 25% of pairs" averages in mismatches. The 0.873 is itself evidence for the
average-person effect (two means both pulled toward the population centre score higher than
any real observation pair does).

That says nothing yet about the number that matters, the *fragment* pair 219/7 ↔ 219/46
under each mode. `explain_merge_failure.py` now prints exactly that, for every shared
camera, under all three modes. **Re-run it before M.5 is touched.** If consensus moves 0.830
down, M.5 is off the table and the answer is M.3 plus a calibrated cam_219 bar.

#### M.9.5 MEASURED — M.5 (scoring mode) is DEAD, and prototype is the best of the three

Fragment-level, run `20260731_060425`, all three modes:

| camera | fragment pair | prototype | max_exemplar | consensus | bar | truth |
|---|---|---|---|---|---|---|
| cam_219 | 219/7 ↔ 219/46 | **0.830** | 0.830 | **0.681** | 0.90 | **same person** |
| cam_224 | 224/30 ↔ 224/102 | 0.773 | 0.842 **PASS** | 0.766 | 0.80 | different |
| cam_213 | 213/51 ↔ 213/88 | 0.675 | 0.675 | 0.568 | 0.80 | different |
| cam_206 | 206/12 ↔ 206/48 | 0.616 | 0.616 | 0.594 | 0.90 | different |

Consensus moves the one true-positive pair **down** 0.830 → 0.681, and it compresses
separation rather than widening it:

- prototype: same-person 0.830 vs best different-person 0.675 → **margin 0.155**
- consensus: same-person 0.681 vs best different-person 0.594 → **margin 0.087**

**Prototype separates this footage better than consensus does.** The synthetic fixture that
motivated #45a predicted the opposite; it does not survive contact with the real data. And
max_exemplar does exactly what it was warned about — it is the only mode that passes a
different-person pair (224/30 ↔ 224/102 at 0.842, on one lucky crop pair).

Item #45a is closed: **keep `scoring: prototype`.** Delete M.5 from the plan.

#### M.9.6 MEASURED — one bar fixes it, and it cascades

Sweep with cam_219's bar lowered (identical output at 0.85 and 0.80):

| same-camera bars | cross | ids | max | multi | below ceiling |
|---|---|---|---|---|---|
| 213=.80, 224=.80 (shipped) | 0.63 | 25 | 7 | 7 | **1** |
| 213=.80, 224=.80 | 0.70 | 27 | 6 | 7 | 0 |
| **213=.80, 219=.85, 224=.80** | **0.63** | 25 | 9 | **8** | **0** |
| 213=.80, 219=.85, 224=.80 | 0.70 | 27 | 9 | 7 | 0 |

At cam_219=0.85 the operator's person assembles correctly as a **9-tracklet identity**:

    GID 1: cam_206(12) cam_213(79,88) cam_219(7,46,113) cam_224(3,68,102)

which reads as one continuous walk: cam_206 (0-6.7 s) → cam_219 (8.4-13.8) → cam_224
(11.4-27.3) / cam_219 (18.2-37.5) → cam_224 (31.4-40.2) → cam_213 (41.4-51.5) → cam_224
(53.9-70.2) / cam_219 (58.3-74.7).

**And the poisoning dissolves on its own.** 213/51 leaves for GID 7, 224/30 leaves for
GID 3, and the 0.649 sub-ceiling merge stops happening — so `below ceiling` reaches 0 **at
the shipped cross bar of 0.63**. Fixing the origin removed the need for M.4.1. Cross-camera
identities go 7 → **8**, the highest of any row swept. This is the only setting measured so
far that improves every column at once.

**Caveat that must not be skipped: this is one subject in one 80-second run.** Decision D9
language applies — a hypothesis from one run, not a calibration.

#### M.9.7 The fix is ORDER-DEPENDENT, which is what M.3 is for

0.85 and 0.80 produce byte-identical clusters, which is suspicious, and the reason matters:
**Phase 1 never merges 219/7 ↔ 219/46 at either bar.** With
`same_camera_reciprocal_best: true`, each fragment must be the *other's* best above-bar
partner, and 219/7's best partner is 219/113 at **0.969** — the same pose. So the pair is
refused by mutual-best regardless of the bar, and the union happens in **Phase 2, at cluster
level**, on a mean-of-means score that lands somewhere in [0.85, 0.90).

So the 0.830 fragment score — the actual evidence that these are one person — never enters
the decision at all. The bar change works by admitting a *cluster mean*, in an order-
dependent phase. It is right on this run and it is contingent.

That is precisely the case for **M.3**: judge the shared-camera claim on the best fragment
pair within each shared camera (0.830 against cam_219's bar) instead of on a cluster mean
against the strictest bar of four cameras. With M.3, cam_219 at 0.80 admits this merge on
the evidence that actually supports it. `explain_merge_failure.py` now reports which phase
united a pair and names the mutual-best partner that blocked Phase 1, so this is checkable
on any future run.

#### M.9.9 OPERATOR VERDICT on the matched re-render (A = shipped, B = cam_219 0.85)

Watched on `20260731_060425`, both renders from the same code path, differing only in
cam_219's same-camera bar. Verbatim in substance, as Part L.4 requires:

| camera | observation | reading |
|---|---|---|
| cam_219 | *"the guy did not change, he remained reid 1 throughout"* | **the reported defect is FIXED.** 219/7 + 219/46 + 219/113 are one identity |
| cam_206 | *"the lighting is pretty bad BUT it misidentified someone there as reid 1 when the reid 1 in cam_219 never was there, then a few seconds later gets assigned to reid 6"* | **confirmed false merge.** cam_206/12 is NOT him. It joined on the 0.818 link |
| cam_224 / cam_206 | *"at the end of cam_224 the person id'ed as reid 12 is the same as the reid 12 in cam_206 for those first few seconds"* | **confirmed CORRECT** cross-camera link. GID 12 = cam_206(53,69,118) + cam_213(140) + cam_224(142) |

**The cam_206 false merge is PRE-EXISTING, not caused by the bar change.** The shipped run's
own summary already reads `GID 1: cam_206 (0012) + cam_213 (0051) + cam_219 (0007 + 0113) +
cam_224 (0030)`, and GID 6 is `cam_206(26,43)` in both A and B. cam_206's behaviour is
byte-identical between the two renders. So on the evidence so far the bar change is a strict
improvement: it fixed cam_219 and changed nothing in cam_206.

#### M.9.10 The cam_206 defect is the SAME defect, one camera over

The operator's *"reid 1 for a few seconds, then reid 6"* in cam_206 is one person split
across 206/12 and 206/26+43, with the first fragment captured by someone else's identity:

- cam_206's same-camera bar is the **global 0.90** — like cam_219's was, it has never been
  calibrated for this footage.
- 206/26 + 206/43 merged at **0.916**, so with `same_camera_reciprocal_best` on, 26's best
  partner is 43. If 206/12 scores below 0.916 against either, mutual-best refuses it and
  206/12 is orphaned — **exactly the mechanism that orphaned 219/7** (M.9.7).
- An orphan is then free to be captured cross-camera, which is what the 0.818 link did.

If that is what the numbers say, then the general finding is that **the uncalibrated 0.90
default is the root defect, not one camera's setting** — and the fix is per-camera
calibration plus M.3, not a patch for cam_206. If 206/12 instead scores *far* below its own
siblings, the bar is not the answer and M.4.2's cohesion floor is.

`explain_merge_failure.py <run> --tracklets cam_206:12,cam_206:26` decides which, and it is
the next measurement.

#### M.9.11 MEASURED — cam_206's bar is NOT the answer, and Phase 1 has a structural hole

`explain_merge_failure.py --tracklets cam_206:12,cam_206:26`:

    cam_206   206/12 vs 206/43   prototype 0.906 PASS   max_exemplar 0.948 PASS   bar 0.90

**206/12's edge to its own sibling is ABOVE cam_206's bar.** It was refused by
mutual-best: 206/26 + 206/43 merged at 0.916, so 43's best partner is 26, and 12 —
whose best partner *is* 43 — is nobody's best. It was orphaned by 0.010 of cosine, then
captured cross-camera at 0.771 into the operator's identity. That is the cam_206
mislabel, fully explained.

Sweeping cam_206's bar confirms it is the wrong lever:

| same-camera bars | ids | 206/12 lands in | new same-camera clusters |
|---|---|---|---|
| 219=.85 | 25 | GID 1 (wrong) | — |
| 219=.85, **206=.85** | **21** | GID 1 (wrong) | `GID 7: cam_206(27,29,67,75,83)` |
| 219=.85, **206=.80** | **20** | GID 1 (wrong) | + `GID 9: cam_206(32,53,69,118)` |

206/12 does not move at any bar, while cam_206 loses 4-5 identities to new
multi-tracklet clusters in the camera with five people and bad lighting. **Do not lower
cam_206's bar.** M.9.10's hypothesis is refuted.

**The real defect: Phase 1 ran exactly one pass, and its comment lied about the
consequence.** It claimed a 3+ way fragmentation would be consolidated later by Phase
2's rounds. Phase 2 cannot: `mergeable_cross` requires a cross-camera member pair, so
two clusters that both sit in one camera are excluded outright
(`NOT_MERGEABLE_CROSS`). The promised second chance never existed.

**Implemented**, behind `identity.reconcile.same_camera_rounds`, default **off**:
Phase 1 now iterates like Phase 2, re-deriving mutual-best from current clusters each
round. Round 1 is identical to the old pass by construction, so off == old behaviour,
which the full suite confirms.

**And iterating alone would have relaxed the guard** — the new regression test caught
it before it shipped. With only two clusters left in a round, mutual-best is *vacuous*
(the one remaining pair is trivially each other's best), and a stranger at 0.910 / 0.890
against the two fragments scored 0.905 against their mean and merged. So a same-camera
merge now additionally requires **every cross-member pair** to clear the bar: merging
asserts all these fragments are one person, so every fragment must agree, and a mean
must not let a majority outvote a member that disagrees. Same insight as M.3.

    chain     C vs A 0.906, C vs B 0.974  -> both clear 0.90, merges
    stranger  S vs A 0.910, S vs B 0.890  -> B disagrees, refused

For singleton clusters that is exactly the old single-pair test, so round 1 is
unaffected. `tests/live/test_reconcile_same_camera_rounds.py`, 17 checks.

**Open question the measurement must answer:** `_cluster_prototype` averages tracklet
prototypes **unweighted**, so 206/26's 4 observations pull the cluster centre as hard as
206/43's 149. That can pull the centre away from the orphan it is supposed to recover.
Rounds are *necessary* for cam_206; whether they are *sufficient* there depends on that
geometry, and observation-weighted prototypes are the follow-up if they are not.

#### M.9.12 The cam_219 fix passes by 0.003 — take 0.80, not 0.85

The merge log at cam_219=0.85 shows which merge fixed the operator's bug:

    same-camera cluster merge ('cam_213', 79) + ('cam_219', 7) (cosine 0.853 >= 0.85)

**It cleared the bar by 0.003.** At 0.86 it fails. Combined with M.9.7 (the union
happens at cluster level, in an order-dependent phase, on a mean of means) this is far
too thin to ship as-is.

0.80 produces byte-identical clusters on this run and has real margin: 0.853 against
0.80 at cluster level, and 0.830 against 0.80 at the fragment level that M.3 would use.
**Prefer cam_219 = 0.80.** This reverses M.9.6's "prefer 0.85, less permissive" — same
measured output, an order of magnitude more headroom, and it is the value M.3 will still
work at.

#### M.9.13 OPERATOR VERDICT on render C (cam_219=0.80 + `same_camera_rounds`)

| observation | reading |
|---|---|
| *"the person marked reid 10 in cam_219 is completely different from the person marked reid 10 in cam_224. These are 2 DIFFERENT people"* | **confirmed FALSE cross-camera merge.** GID 10 = cam_206(34) + cam_219(6) + cam_224(5) |
| *"in cam_206 the different guy is still assigned reid 1 for the first 8 seconds then gets reid 6"* | **rounds did NOT fix it.** 206/12 (t=0..6.7 s, 17 obs) is still captured into the walker's identity |
| *"you may have made the assumption that if reid 10 is in 224 then he HAS to be in 219; it's a possibility BUT not a 100% guarantee"* | **methodologically correct, and a caution this analysis needs.** See below |

**On the assumption.** A conflict-free timeline is *"not disproven"*, never *"confirmed"* —
and with four overlapping cameras almost any timeline is conflict-free. The GID 1 argument
in M.9.6 leaned on temporal coherence as corroboration; it is not. Only labels are. Two of
the eight multi-camera identities at cross 0.63 are now labelled WRONG (GID 1's cam_206
member, and GID 10 entirely), which means **`multi` was inflated by false merges and cannot
be read as a quality signal on its own.** Every future sweep row must be read with the
labels, not with "multi should stay healthy".

#### M.9.14 GID 10: the merge asserts a pair that was NEVER SCORED

The operator's question was "if embeddings are being checked, why is this happening?" They
were checked — just not between the two tracklets the result claims are one person:

    cross-camera cluster merge ('cam_206', 34) + ('cam_219', 6)  cosine 0.779
    cross-camera cluster merge ('cam_206', 34) + ('cam_224', 5)  cosine 0.732

**cam_219/6 vs cam_224/5 was never compared.** Both attached to the hub cam_206/34, and
union-by-transitivity produced the identity. This is the failure the cross-threshold comment
predicted — *"clusters grow by union, so two weak links in a row fuse three strangers"* —
except both links are ABOVE the measured 0.661 stranger ceiling, so the ceiling did not
catch it either.

It is the same structural defect just fixed in Phase 1 (M.9.11's `all_member_pairs_clear`),
one lane over: **a merge is judged on a cluster mean while the pairs it actually asserts go
unchecked.** The symmetric fix is to require the implied member pairs to clear a floor.
Which floor is a real question, because cross-camera appearance is legitimately weaker than
same-camera — a person's front in cam A and back in cam C can both match a side view in
cam B without matching each other. Requiring the full cross bar may be too strict;
requiring "no member pair below the measured different-person ceiling" uses a measured
number rather than a new tunable. **Measure `219/6 vs 224/5` before choosing.**

#### M.9.15 Two labelled false merges bracket the cross bar

| link | cosine | operator label |
|---|---|---|
| 206/12 → the walker's cluster | **0.771** | WRONG |
| 206/34 → 219/6 | **0.779** | WRONG |
| 206/34 → 224/5 | 0.732 | WRONG |
| the walker's own links (219/46+224/3, +224/102, 213/79+219/46, 213/79+219/7) | 0.966, 0.948, 0.887, 0.853 | RIGHT |

There is a clean gap: every labelled-correct link is **≥ 0.853**, every labelled-wrong one is
**≤ 0.779**. A cross bar anywhere in 0.78–0.85 removes all three confirmed false merges and
keeps the walker's identity whole. That is the first time this project has had *labels* on
both sides of the cross bar rather than a heuristic — and it says 0.63 is far too low, by
about 0.15, not by the 0.07 the ceiling suggested.

Cost to measure, not assume: which of the other cross-camera links die with them.

#### M.9.16 MEASURED — the cross bar is bracketed to (0.779, 0.837); take 0.80

**First, a negative result that kills M.9.14's proposed guard.** `219/6 vs 224/5`, the pair
the GID 10 merge asserts and never scored, is **0.713** — above the cross bar, above the
0.661 stranger ceiling. So an all-member-pairs guard would NOT have caught this false merge
at either candidate floor. The hub-merge structure is real but it is not what needs fixing
here; the bar is. **Do not implement the cross-camera all-pairs guard on this evidence.**

Sweep at cam_219=0.80, with `same_camera_rounds`:

| cross | ids | max | multi | xmerges | min accepted x |
|---|---|---|---|---|---|
| 0.63 | 25 | 9 | 8 | 13 | 0.662 |
| **0.78** | 31 | 8 | 5 | 7 | 0.837 |
| **0.80** | 31 | 8 | 5 | 7 | 0.837 |
| 0.85 | 33 | 8 | 4 | 5 | 0.860 |

At 0.78 and 0.80 (identical output) **all three operator-confirmed false merges are gone and
nothing labelled correct is lost**:

- the walker becomes `GID 27: cam_213(79,88) cam_219(7,46,113) cam_224(3,68,102)` — all 8 of
  his tracklets, and **cam_206(12) is out**
- **GID 10 is dissolved entirely** — 206/34, 219/6, 224/5 all separate
- `GID 12: cam_206(53,69,118) cam_213(140) cam_224(142)` survives intact — the operator's
  confirmed-CORRECT cam_206 ↔ cam_224 link, which is the positive control

`multi` falls 8 → 5, and that is the *correct* direction here. Every multi-camera identity
lost was either confirmed wrong or held together by a scrap: GID 10 (confirmed wrong),
GID 1's cam_206/12 (confirmed wrong, 0.771), GID 3's cam_224/30 (3 observations, 0.727),
GID 7's cam_213/51 (6 observations over 1.6 s, 0.696), GID 11's cam_206/48 (3 observations,
0.662). **This is why `multi` must never be read without labels** (M.9.13).

**The decision boundary is bracketed.** Labelled-wrong tops out at 0.779; labelled-right
bottoms out at 0.837 (GID 12). Any bar in that 0.058-wide gap works on this run, and 0.80
sits near its midpoint. 0.78 is only 0.001 clear of the highest false merge, so it has no
margin. **cross = 0.80.**

This is the first cross-bar value in the project's history chosen from labels on *both*
sides rather than from a one-sided ceiling. The two previous cross-bar changes (0.63 → 0.70,
reverted) had only the ceiling. Note the honest caveat unchanged: one run, one operator
session, ~11 subjects.

#### M.9.17 SHIP CANDIDATE — two lines

    identity.reconcile.threshold: 0.63 -> 0.80
    identity.reconcile.per_camera.cam_219.same_camera_threshold: (absent, 0.90) -> 0.80

`same_camera_rounds` stays **false**. It fixes a real structural hole (M.9.11) and is tested,
but it changed nothing measurable on this run — it did not move 206/12, because 206/12's
problem was capture by a 0.771 link, not failure to reach its siblings. Shipping a change
whose only measured effect is nil adds risk for no benefit. It stays in the codebase, off,
until a run exercises it. cam_206 stays at 0.90 (M.9.11: lowering it is actively harmful).

Expected on the next live run: **more identities, fewer cross-camera links** (31 vs 25, 5 vs
8 on this footage). That is the intended direction, not a regression — do not "fix" it by
lowering the bar again without labels.

#### M.9.18 RETRACTED — M.9.16's "clean bracket" is wrong; the bands OVERLAP

Two more operator labels, and they destroy the separation M.9.16 claimed:

| pair | cosine | label | how it failed |
|---|---|---|---|
| cam_219/6 ↔ cam_224/30 | **0.688** | **SAME person** | below the 0.80 bar → split |
| cam_206/107 ↔ cam_213/133 | **0.818** | **SAME person** | TEMPORAL_CONFLICT, 1.3 s > 1.0 s |
| cam_206/12 ↔ the walker | 0.771 | different | — |
| cam_206/34 ↔ cam_219/6 | 0.779 | different | — |

**A labelled-TRUE link at 0.688 sits below two labelled-FALSE links at 0.771 and 0.779.
No cross-camera threshold separates them.** M.9.16's bracket of (0.779, 0.837) was an
artifact of having labels only on the strong end. This is exactly J.6's finding — the
per-subject same/different boundaries overlap — now reproduced on the CROSS-camera axis with
operator labels instead of statistics.

The consequence is structural, and it kills the shape of the whole M.4 line of attack:
**the cross bar cannot be tuned to correctness on this footage.** Any value trades a false
merge for a false split. A third lever is required, and appearance is not it.

Note WHAT the 0.688 pair is: cam_224/30 is **3 observations over 0.8 s at a bad angle**. Its
prototype is three crops of one difficult instant. The pair is unlinkable not because the bar
is wrong but because the evidence does not exist. See M.9.20.

#### M.9.19 A CONFIRMED FALSE VETO — the covisibility config is wrong

`cam_206/107 ↔ cam_213/133` is one person (operator-labelled) and was refused by
`TEMPORAL_CONFLICT_CROSS_CAMERA`: **1.3 s of wall-clock overlap against a 1.0 s tolerance.**
This is the first *labelled* instance of the failure invariant L.2.2 exists to prevent — a
wrong veto manufacturing a second identity for someone who has one — and item M-h predicted
it: only `cam_224 ↔ cam_219` is declared `covisible`, while the room's cameras overlap and
`ts` is RECEIVE time (#15) carrying jitter, decode-cost differences and frame-rate
quantisation.

**cam_213 is implicated in every false veto found so far**, at margins of 0.3-0.6 s over a
1.0 s tolerance:

    cam_213/133 <-> cam_206/107   1.3 s   (blocks a confirmed-same person)
    cam_213/51  <-> cam_219/46    1.6 s   (blocked the original reid 1 / reid 11 merge)
    cam_213/51  <-> cam_224/3     1.6 s   (same)

cam_213's tracklets are short (6-12 observations, 1.6-3.8 s spans), so a 1.3-1.6 s "overlap"
is most of a tracklet's life — the quantity being compared is barely longer than the noise
in it. Either those camera pairs genuinely see one person at once, or the tolerance is inside
the jitter; both mean the veto is asserting an impossibility that is not one.

**This must be settled before any further threshold work**, because the veto runs BEFORE
scoring: while it is wrong, every bar is being calibrated against a candidate space that has
had true pairs removed from it. `--no-covisibility` bounds what it is costing.

#### M.9.20 The operator's own answer: prefer UNRESOLVED to a wrong number

*"in cam_224 he is at a bad angle so it goes from unresolved to reid 31 but I still don't
think it should show the wrong reid."*

That is a product requirement, and it points at a lever no threshold provides.
`cam_224/30` has **3 observations** — exactly `min_tracklet_observations`, so it survives
suppression by one observation and is then minted as its own identity, which puts a WRONG
number on screen. At `min_tracklet_observations: 4` it is suppressed instead and renders as
UNRESOLVED, which is what the operator asked for: no claim rather than a false claim.

This is the honest response to M.9.18. When the evidence cannot support an identity claim,
the fix is not to move a bar until the claim lands somewhere — it is to decline to claim.
Sweep `--min-obs 4,5` at cross 0.80 and count how many tracklets go from "wrong number" to
"no number".

#### M.9.21 The rounds change has still never run — stale deployment, twice

Both `explain` runs and both sweeps report:

    settings: ... reciprocal=on same_reciprocal=on covisibility=on/6 pairs
    tracklet reconcile: same-camera reciprocal-best ON -- 10 merged, 24 above-bar pair(s) refused

No `same_rounds=` field, and the log line lacks the `over N round(s)` text this batch added.
So the box is still running the pre-rounds code and `--same-rounds` has been silently
ignored in every run that passed it. `validate_flags` would have failed loudly on exactly
this — and it is in the same un-deployed batch. **Nothing about Phase 1 rounds has been
measured yet; treat M.9.11 as untested.**

#### M.9.22 MEASURED — the veto is LOAD-BEARING. Do not switch it off.

`--no-covisibility` at cross 0.80: 28 identities, **max 13**, min accepted x 0.805.

It does recover the confirmed-true link:

    GID 18: cam_206(107)  cam_213(133)  cam_224(137)      <- the reid 18/24 person, whole

and it creates a confirmed-FALSE one, at a score 0.013 lower:

    GID 12: cam_206(53,69,118) cam_213(79,88,140) cam_219(7,46,113) cam_224(3,68,102,142)

That 13-tracklet cluster fuses **two separately operator-confirmed people**: the walker
(cam_213/79,88 + cam_219/7,46,113 + cam_224/3,68,102) and the GID 12 man whose cam_206 ↔
cam_224 link the operator verified by eye. The veto had been the only thing keeping them
apart.

So the answer to M.9.19 is not "the veto is wrong", it is **"the veto's TOLERANCES are wrong
for at least one camera pair"**. Switching it off trades a true 0.818 for a false 0.805 —
another overlapping pair of labels, 13 thousandths apart. Global on/off is not the lever;
per-pair tolerance is.

**The open question that decides the fix:** which member pair blocks the walker ↔ GID 12
fusion when the veto is ON? If it is *not* cam_206 ↔ cam_213, then raising cam_206 ↔ cam_213
alone recovers GID 18 without enabling the fusion — a clean, targeted win. If it *is* that
pair, the two outcomes are inseparable by tolerance and this lever is exhausted too.

#### M.9.23 MEASURED — `min_tracklet_observations` is not a discriminator either

| min_obs | ids | what changed |
|---|---|---|
| 3 (shipped) | 31 | — |
| **4** | 27 | `cam_224(30)` gone (**wrong** number removed, as intended) **and `cam_224(142)` gone** (**right** link removed) |
| **5** | 24 | also `cam_206(26)` gone — **and `GID 1: cam_206(12,43)` appears** |

M.9.20's proposal half-works and half-backfires. Both `cam_224/30` and `cam_224/142` are
**3-observation cam_224 tracklets**; one was linked WRONGLY (0.688 to cam_219/6) and the
other CORRECTLY (0.860 to cam_206/69, operator-verified). Suppression cannot tell them apart,
because the thing it measures — observation count — is identical. So raising the bar removes
a false claim and a true claim together. It is a blunt instrument, not a fix.

**But min_obs 5 produced the most informative accident of the whole investigation:**

    GID 1: cam_206(12,43)

**cam_206's split closed.** Suppressing `cam_206/26` (4 observations) freed cam_206/43's
mutual-best slot, so 43's best partner became 12 at 0.906 and the pair merged. That is
M.9.11's mechanism confirmed from the opposite direction: the orphaning was caused *entirely*
by 26 occupying the slot, and removing 26 — or letting 12 compete against the {26,43} cluster
in a second round — closes it. **Phase 1 rounds should achieve the same result without
discarding a real tracklet**, and remain untested (M.9.21).

#### M.9.24 Where this leaves the whole approach: appearance is exhausted

Every global knob now trades one operator-confirmed error for another, and the margins are
inside the noise of the embedding:

| lever | recovers | costs | gap |
|---|---|---|---|
| cross bar | 0.688 true pair | 0.771 / 0.779 false pairs | **inverted** |
| covisibility off | 0.818 true link | 0.805 false merge | **0.013** |
| min_obs 4 | removes 0.688 false claim | removes 0.860 true link | identical evidence |

This is J.6's overlapping-boundaries finding, reproduced three more times on three different
axes with operator labels rather than statistics. **No global threshold on appearance
separates the remaining errors on this footage.** Four rounds of threshold tuning were
reverted before this investigation; the reason is now measured rather than suspected.

Two levers remain that are NOT global appearance thresholds:

1. **Per-pair covisibility tolerance** (M.9.22) — uses topology, not similarity. Free to
   measure offline, and the only clean win still on the table.
2. **Evidence density.** The recurring subject of every remaining defect is a **3-4
   observation tracklet**: `cam_224/30` (3), `cam_224/142` (3), `cam_206/26` (4),
   `cam_206/48` (3), `cam_213/51` (6). At `reid.interval_sec: 0.4` a one-second appearance
   yields 2-3 observations, so these prototypes are three crops of one instant. Halving the
   interval doubles them. **This is the first item in the entire investigation that genuinely
   requires a new live run**, because it changes what gets recorded rather than how it is
   clustered.

Geometry (ADR-003) is the third, and is the principled answer to an exhausted appearance
signal — but it is a subsystem, not a setting.

#### M.9.25 MEASURED — the veto's own evidence separates cleanly. The tolerance is the bug.

Every labelled veto in this investigation, sorted by overlap:

| overlap | verdict | pair | case |
|---|---|---|---|
| 1.2 s | **WRONG** | cam_219/113 ↔ cam_213/140 | one of 7 blockers of walker ↔ GID 12 |
| 1.3 s | **WRONG** | cam_206/107 ↔ cam_213/133 | splits the reid 18 / 24 man |
| 1.6 s | **WRONG** | cam_213/51 ↔ cam_219/46 | blocked the original reid 1 / reid 11 |
| 1.6 s | **WRONG** | cam_213/51 ↔ cam_224/3 | same |
| 3.0 s | RIGHT | cam_213/88 ↔ cam_206/43 | walker vs the cam_206 man |
| 3.1 s | RIGHT | cam_219/46 ↔ cam_206/69 | walker vs GID 12 |
| 3.1 s | RIGHT | cam_224/68 ↔ cam_206/69 | walker vs GID 12 |
| 3.8 s | RIGHT | cam_213/79 ↔ cam_206/43 | walker vs the cam_206 man |
| 5.5 s | RIGHT | cam_224/3 ↔ cam_206/53 | walker vs GID 12 |
| 8.0 s | RIGHT | cam_219/46 ↔ cam_206/53 | walker vs GID 12 |
| 8.6 s | RIGHT | cam_224/102 ↔ cam_206/118 | walker vs GID 12 |
| 10.5 s | RIGHT | cam_219/113 ↔ cam_206/118 | walker vs GID 12 |

**Wrong vetoes: 1.2-1.6 s. Right vetoes: 3.0-10.5 s. A gap of (1.6, 3.0), midpoint 2.3 s.**
The shipped tolerance is **1.0 s — below every wrong veto**, so it fires on all four.

This is the first lever in the whole investigation that fixes confirmed errors **without
costing another one.** Every labelled case comes out correct at any tolerance in that gap.
And the config comment already had the right intent — *"tolerances are generous on purpose:
`ts` is RECEIVE time, so it carries network jitter, decode-cost differences and frame-rate
quantisation. 1.0s vetoes only sustained overlap, not timing noise"* — the reasoning was
sound and the number was simply three times too small. Genuine co-presence in this room runs
3-10 s; 1.2-1.6 s is the noise floor of a receive-time clock.

The 7 blockers of walker ↔ GID 12 do **not** include cam_206 ↔ cam_213, and 6 of the 7 are
≥ 3.1 s, so raising the tolerance to 2.3 s releases only the 1.2 s one and the fusion stays
blocked six ways over.

`--covis-tolerance S` added to the offline tools to sweep this axis (`--no-covisibility` was
all-or-nothing and could not express it).

#### M.9.26 M.3 would block the same false fusion INDEPENDENTLY, on appearance alone

From the same output, walker vs GID 12:

    cluster mean (in force)          0.823   PASSES the 0.80 bar
    cam_213: 213/88 vs 213/140       0.712   fails
    cam_224: 224/102 vs 224/142      0.730   fails

The two clusters share cam_213 and cam_224, and **every fragment pair inside those shared
cameras scores 0.09-0.11 BELOW the bar while the mean of the cluster means passes it.** So
M.3 — judge the shared-camera claim on the fragment pairs — refuses this merge on appearance
evidence, with no temporal veto involved at all.

Checked against the labelled cases M.3 must not break:

- walker's own assembly, 213/79 + 219/7: shares cam_219, fragment 219/7 ↔ 219/46 = **0.830 ≥ 0.80** → kept
- reid 18 ↔ reid 24: camera-disjoint, no same-camera claim, 0.818 ≥ 0.80 → kept

So M.3 and the tolerance fix are independent guards that both point the right way on every
label available. That is much stronger than either alone, and it is the first time two
mechanisms have agreed rather than traded.

#### M.9.27 CONFIRMED — tolerance 2.3 s is surgical

Sweep at cross 0.80, cam_219 0.80, `--covis-tolerance 2.3`: **30 identities, max 8, multi 5,
xmerges 8, below-ceiling 0, min accepted x 0.818.**

    GID 18: cam_206(107)  cam_213(133)  cam_224(137)     <- gained 213/133, the confirmed link
    GID 27: cam_213(79,88) cam_219(7,46,113) cam_224(3,68,102)   <- walker, still exactly 8
    GID 12: cam_206(53,69,118) cam_213(140) cam_224(142)  <- still separate, still intact

`xmerges` 7 → 8. **Exactly one merge was added and it is the operator-confirmed 0.818 link.**
No cluster grew, nothing fused, no below-ceiling merge appeared. Predicted in M.9.25 and
reproduced exactly.

#### M.9.28 SHIP CANDIDATE (supersedes M.9.17)

    identity.reconcile.threshold:                        0.63  -> 0.80
    identity.reconcile.per_camera.cam_219.same_camera_threshold:  (0.90) -> 0.80
    identity.reconcile.covisibility.default_tolerance_sec:  1.0 -> 2.3
    identity.reconcile.covisibility.pairs -- all five numeric entries: 1.0 -> 2.3
    (cam_224 <-> cam_219 stays `covisible`)

Addresses four of the five operator complaints on run 20260731_060425:

| complaint | status |
|---|---|
| cam_219 reid 1 → 11 → 1 | **fixed** (M.9.6, verified on video) |
| cam_206 man wearing the walker's reid | **fixed** (cross 0.80, M.9.16) |
| reid 10 = two different people | **fixed** (cross 0.80, M.9.16) |
| reid 18 / reid 24 split | **fixed** (tolerance 2.3, M.9.27) |
| cam_219/6 ↔ cam_224/30 split | **NOT fixed** — 0.688 on 3 bad-angle crops (M.9.18/M.9.24) |

Still open and independent of the above: cam_206's own split (206/12 vs 206/26,43), for which
`same_camera_rounds` is the candidate and is now finally deployable (M.9.21 resolved).

#### M.9.8 Revised order

1. **Watch the re-render.** `cmp_cam_219_*.mp4` at 8-14 s and 58-75 s, against
   `output_cam_219.mp4` from the run. Nothing ships before this; it is the only ground truth.
2. **cam_219 same-camera bar 0.90 → 0.80** (M.9.6, revised by M.9.12). Identical measured
   output to 0.85 but the deciding merge clears by 0.053 instead of 0.003.
3. **`same_camera_rounds: true`** (M.9.11), if the sweep shows 206/12 moving to GID 6
   without cam_206 over-merging. Implemented and tested; needs one offline measurement.
4. **M.3** — judge the shared-camera claim per camera on the fragment pair (M.9.7). Turns
   step 2 from an order-dependent cluster-mean accident into a decision resting on the 0.830
   that actually supports it. Ship behind a flag, default off, measure, then enable.
5. **Observation-weighted `_cluster_prototype`** — only if step 3 does not move 206/12
   (M.9.11's open question). A 4-observation fragment should not pull a cluster centre as
   hard as a 149-observation one.
6. **M.4.2** — cohesion floor, so a 3-observation scrap cannot join on a 0.760 link and then
   veto for the cluster. Independent of the above and still worth having.
5. ~~**M.4.1** cross 0.70~~ — **no longer needed.** Step 2 takes `below ceiling` to 0 at
   0.63, and 0.70 costs a cross-camera identity (8 → 7). Leave the cross bar alone.
6. ~~**M.5** scoring mode~~ — **closed, measured dead** (M.9.5). Keep `prototype`.
7. **M.2** — opportunistic; no longer on the critical path.

---

## Part I — Historical context

Git history, parsed from every committed `config.yaml`:

- The `live.reconcile` key **did not exist** before commit `7dcf93d775` (2026-07-28), and
  `LivePipeline` defaults it to **`False`** when absent. So **every live run before that date
  produced live-annotated output, where the engine's ids *were* the product.**
- Since 2026-07-28 it has been `true` in every commit; never explicitly `false`.

Consequences for the tuning record:

- The 2026-07-24 threshold work (`same_camera_threshold` 0.90→0.85, `cross_camera_threshold`
  0.63→0.60, the "A100 calibration" commits) was calibrated for a mode that is **no longer
  shipped**.
- The later 0.85→0.70 change (2026-07-29) landed **after** reconcile was enabled, so it
  **could not have affected the deliverable at all**.
- The `recam_rej_below` counter that motivated those reductions is **corrupted by the two-lane
  leak** — the same resolve increments both `recam_rej_below` and `linked`, so it counted
  resolves that had already succeeded.

Four prior "tighten the knobs" attempts were reverted for hurting accuracy. The lesson encoded
throughout this plan: **instrument first, measure on a frozen clip, and treat every threshold
as a hypothesis.** During the audit that produced this document, four separate hypotheses
derived from reading code were falsified by measurement — including two about duplicate boxes
and one about frame dropping causing fragmentation. Assume the same failure rate for anything
tagged **read** rather than **measured**.
