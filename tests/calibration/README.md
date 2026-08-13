# Calibration harness

Measurement scripts backing [`REMEDIATION_PLAN.md`](../../REMEDIATION_PLAN.md) Part H.
They exist to **produce numbers on real footage**, not to pass. Their whole purpose is
to make threshold and model decisions falsifiable instead of argued.

Run everything from the repo root:

```bash
python tests/calibration/verify_embedding_contract.py
python tests/calibration/measure_score_separation.py     register_file.avi 90 6
python tests/calibration/compare_backbones.py          register_file.avi 60 6
python tests/calibration/measure_reconcile_thresholds.py register_file.avi 90 4
python tests/calibration/measure_detection.py            register_file.avi 50
python tests/calibration/compare_detector_models.py      register_file.avi 150
python tests/calibration/characterize_known_defects.py
python tests/calibration/audit_product_path.py           test_v2.avi 40 2
python tests/calibration/analyze_decision_log.py         logs/reconcile_decisions_<run_id>.jsonl
```

The detector these use is read from `config.yaml` (`detector.model`), not hardcoded, so a
model swap cannot leave the harness measuring the old model and reporting it as shipped
behaviour. Override with `CALIB_DETECT_WEIGHTS=yolo11n.pt`.

All are CPU-safe. Nothing writes into the repo — `audit_product_path.py` runs in a temp
directory with a distinct camera name and a local Qdrant, so it cannot overwrite a real
`output_<cam>.mp4` or touch the shared gallery.

## What each script is for

| Script | Asserts? | Produces | Run it when |
|---|---|---|---|
| `verify_embedding_contract.py` | **yes**, exits non-zero | Part D | after any change to `reid/extractor.py` or `reid/service.py` |
| `measure_score_separation.py` | no | H.1, H.2, H.3, H.5 | footage, ReID weights, or feature tap changes. **Could not run on FastReID until 2026-08-04** — it reached into torchreid OSNet's `fc` block for its tap comparison, so it raised before measuring anything. The tap axis is now discovered from the backend: OSNet gets both taps, anything else gets one (`production`) |
| `compare_backbones.py` | no | — | before changing `reid.model`; ranks backbones with threshold-FREE metrics (AUC, R@1) because a cosine bar means nothing across feature spaces. Refuses to name a winner when the footage saturates |
| `measure_reconcile_thresholds.py` | no | H.4 | before touching any `identity.reconcile.*` threshold. Backend-agnostic (uses `extract_batch`), so it always worked. **This is the one that produced the FastReID re-anchoring measurement** recorded beside `same_camera_threshold` in `config.yaml` |
| `measure_detection.py` | no | H.6, H.7 | before proposing any detector-side change |
| `compare_detector_models.py` | no | H.11 | before changing `detector.model`; answers recall AND whether extra boxes become sustained tracks |
| `analyze_decision_log.py` | no | J.6, J.10 | after every live run — it is how a threshold change is judged |
| `characterize_known_defects.py` | no, always exits 0 | — | after each phase lands, to see what moved |
| `audit_product_path.py` | no | H.10 | after any pipeline change; the only whole-path check |
| `explain_merge_failure.py` | no | Part M | when the operator reports one person carrying two reids — names the rule and the tracklet pair that refused the merge |
| `sweep_reconcile_thresholds.py` | no | #41, Part M.1 | to explore merge settings on a finished run, with no camera time |
| `rerender_from_clips.py` | no | — | to see a setting on the actual video before it ships |

Geometry has its own tooling, outside this directory, because it is not a
measurement of the appearance model:

| Tool | What it does |
|---|---|
| `tools/fit_floor_frame.py` | fits the shared floor frame from people's own foot points on a finished run — no clicking, no floor plan, no camera time. Refuses to write a calibration that fails its label-free validation |
| `tests/live/test_geometry_*.py` | the geometry maths, fully verifiable on the dev box (no GPU, no footage, no Qdrant) |

`--geometry`, `--no-geometry` and `--geometry-safety F` work on the three
reconcile-replaying tools, so the reachability veto can be judged on a finished run
with no camera time. They toggle **policy only**: they cannot conjure geometry into a
run captured without it, because positions are recorded live and only there. On such
a run reconcile says so loudly and proceeds on appearance alone.

**The three scripts that re-run reconcile** (`explain_merge_failure`,
`sweep_reconcile_thresholds`, `rerender_from_clips`) all take their settings from
`config.yaml` through `identity.reconcile.resolve_reconcile_kwargs`, and each prints the
full setting line before its first result. That is not decoration. Until 2026-07-31 the
sweep and the re-render passed neither `covisibility` nor `same_camera_reciprocal_best`,
both of which default OFF in `reconcile_tracklets` and are ON in production — so every
threshold conclusion in Part J was measured against a clustering that does not ship. **If
that setting line ever disagrees with the run you are trying to explain, nothing below it
means anything.**

`verify_embedding_contract.py` is the one genuine regression test. Everything it covers is
a **silent** failure mode — a broken BGR→RGB swap, a transposed resize, a batch/crop
misalignment — that produces no error and no crash, only worse matching.

## Two methodological rules

**1. A "different person" pair requires co-occurrence in the same frame.** A person cannot be
two simultaneous detections, so co-occurrence proves distinctness. Comparing tracks that never
co-occur silently mixes in fragments of the *same* person — a known defect here — which
inflates every different-person statistic. `_common.proven_distinct_pairs` enforces this and
prints what it excluded. The first version of this measurement got it wrong and reported a
different-person ceiling that was too high.

**2. Bank queries must be held out of the bank.** If a query observation is also in the bank,
the max-exemplar term in `ActiveIdentitySet.score` matches it against itself and returns 1.000,
inflating the same-person distribution by whatever fraction of queries sit in the bank. That
makes the measured margin depend on the frame count rather than the model.
`measure_score_separation._holdout` splits each track into a bank half and a disjoint query
half. The live engine documents this same trap in `_reinforce`. **This also got wrong first
time, and the corrected numbers are lower** — see the note in Part H.

## What these numbers can and cannot decide

**Stable across sample sizes — safe to act on:**

- post-BN beats post-ReLU on separation margin and lowers the different-person ceiling
- consensus scoring lowers the different-person ceiling versus `max(prototype, exemplar)`
- the different-person ceiling sits well above `live.identity.same_camera_threshold` (0.70)
- yolo11m produces **fewer, longer** track ids than yolo11n on the same frames (held at
  both 50 and 150 frames): 4 continuous tracks versus 6, where yolo11n splits one person
  into 64 + 37 frames. The *direction* is stable; the CPU cost ratio (2.0×) is not a GPU
  ratio, and this clip has no frame dropping, which is where the bigger model could lose

**Not stable — never set a threshold from one run:**

- `other MAX` is an extreme-value statistic and **grows with sample size**. Prefer p95. On the
  same clip, 48 versus 90 frames moved the raw MAX from 0.819 to 0.936.
- the separation margin moved +0.055 → +0.108 between those same two runs
- `KB/frame` for the temp clip is inflated below ~50 frames by container overhead

So: use these scripts to compare **options** (feature tap, scoring mode, NMS threshold) on
identical footage. Actual threshold values come from the Phase 9 sweep on frozen multi-camera
footage, which is also the only way to calibrate `cross_camera_threshold` — every measurement
here is single-camera.

## Footage

Defaults to `register_file.avi`: 2560×1440, matching production resolution, and it decodes with
**zero H.265 reference errors**. The other clips do not:

| Clip | Frames | H.265 reference errors |
|---|---|---|
| `register_file.avi` | 1428 | **0** |
| `test_v2.avi` | 1573 | 207 (13%) |
| `test_file.avi` | 682 | 294 (43%) |

Measuring feature quality on a clip with corrupted frames measures the corruption. Use
`register_file.avi` for anything appearance-related. The corruption itself is issue #28 — no
RTSP transport or timeout option is set anywhere, so packet loss over UDP produces exactly
these errors.

## Known limits of the current sample

Six people, one camera, one clip, 14 proven-distinct pairs. No cross-camera data at all. No
crowded footage. A prior 3-camera RTSP run reconciled 35 tracklets into 7 identities — 5×
fragmentation — and **none of the available clips reproduce that**, so fragmentation fixes
cannot be validated here. See Part E of the plan for the runs that would close these gaps.
