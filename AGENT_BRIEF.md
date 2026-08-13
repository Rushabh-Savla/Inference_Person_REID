# AGENT BRIEF — revision 4, 2026-08-05

**Supersedes every earlier revision.** Revision 3's central claim — "the usable window
is EMPTY, no bar satisfies both sides" — is **WITHDRAWN**. It was computed by pooling
cameras and taking a MAX on a contaminated control. Per camera, against p95, and now
against real labels, a window exists and has been found.

## THE STATE, 2026-08-05 — read this and nothing else if you are short of time

**Ground truth exists.** Run `20260805_093512` (3 cams, 92 s, 1299 obs, clips kept)
carries 32 operator-confirmed pairs in `calibration/tracklet_pairs.jsonl`, including
the project's **first cross-camera same-person labels**. Three people: A (reids 3+4+8,
3 cameras, 8 tracklets), B (7+9), C (6+11).

**At the shipped settings, 0 of 3 were recovered.** The fix, measured with
`tests/calibration/sweep_against_labels.py` and now in `config.yaml`:

| | was | now | why |
|---|---|---|---|
| `identity.reconcile.threshold` | 0.63 | **0.55** | 2 of 3 failures are cross-camera; at 0.63 NO same-camera bar exceeds 1/3 |
| per-camera same bars | 0.90 / 0.80 / 0.80 | **0.60** all three | cam_219 had no override at all and inherited 0.90 |
| `scoring` | prototype | **max_exemplar** | 3/3 with 0.10 more cross headroom; compares observation pairs, not two means |

3/3 people recovered, **zero provable false merges**, 7 identities. Label-validated,
**not yet watched** — the footage exists, so `rerender_from_clips.py 20260805_093512`
closes the loop with no camera time. THAT IS THE NEXT ACTION.

## What revision 3 got wrong, so nobody re-derives it

| rev 3 claim | status |
|---|---|
| "the window is EMPTY; no bar satisfies both" | **withdrawn** — artefact of pooling cameras + MAX. Per camera against p95 it was already positive in RAW space. |
| "same-person split-half min 0.594" | contaminated: those tracklets were chimeric (two people under one track_id). |
| "cross-camera `threshold` is never consulted" | **false in general.** It is the binding constraint for 2 of 3 labelled people. |
| "threshold tuning was never going to work" | **false.** It works when the CROSS bar moves and when settings are scored against labels rather than counts. |
| camera-bias / feature centring (hypothesis #3) | measured: **inconclusive and expensive.** 8 of 8 held-out margins improved but NOT ONE ranking changed, and the tail is a cross-space comparison that licenses nothing. Parked on cost. |
| prototype mean-pooling destroys view info (hypothesis #1) | **supported** — `max_exemplar` beats `prototype` on labels. Acted on. |

## Rules that outrank instinct — additions since revision 3

1. **Score against labels, never a count.** `sweep_against_labels.py`, not
   `sweep_reconcile_thresholds.py`. Every one of the seven reverted changes was chosen
   from a count.
2. **The grids are non-monotonic.** A plateau of adjacent clean cells is the unit of
   evidence; a single cell is an overfit.
3. **"Zero provable false merges" is necessary, not sufficient.** Co-presence only
   convicts people who share a frame.
4. **Copy `._live_src_*` aside under the run_id before every capture.** Filenames carry
   no run_id and every run overwrites them; run `20260804_064551`'s footage was lost
   this way and is unrecoverable.
5. **Never join a run's store against another run's clips.** ByteTrack renumbers per
   run; the id spaces barely intersect. It reported cam_219 100% UNRESOLVED and 84% of
   all boxes, which was a join error, not a pipeline failure. Both tools now refuse it.
6. **Every measurement here reads the STORE, so anything failing upstream of the store
   is invisible to all of them.** The annotations sidecar is the only independent
   source — `measure_unresolved.py` joins the two.

## Closed since revision 3 — do not reopen without a number

- **The unresolved/coverage question.** 14.4% of drawn boxes carried no id; **99% of
  that was ONE static false positive** (cam_219 track 10: 47.6 s, motion 4.7 px,
  `bad_aspect_hi` 100%). Furniture, correctly refused by the quality gate. Real rate
  ~0.6%. `no track_id` 0.3%, `min_tracklet_observations` 0.2%. **The quality gate is
  exonerated**; `min_box_area_ratio` fired 0% everywhere.
- **Frame dropping starving ByteTrack.** Dead: processed frames == captured frames.
- **Crop preprocessing.** Faithful to FastReID upstream.
- **`max_observations_per_side`.** Raising 64 → all changes nothing; median 8 obs/tracklet.
- **Phase-1 vs Phase-2 bar conflation.** Real, now separable via
  `cluster_same_camera_threshold` (ships `null`), pinned by 11 checks. Verified it does
  NOT fix 064551 — recorded so nobody expects it to.

## Still open, ranked

1. **WATCH the new settings.** `rerender_from_clips.py 20260805_093512`. Nothing else
   matters until this is done.
2. **Person A's cam_213 front/back pair** (`19` vs `22`) is the hardest label — it is
   the "reid 2 by his front, reid 7 by his back" symptom, in the camera with **no
   stranger reference at all** (fewer than 3 co-present pairs in every run).
3. **cam_206** — absent since 2026-07-30, never measured under yolo11m.
4. **cam_213's stream may be a SUB-stream** (`/1/1`, 1920x1080 vs 2560x1440 on the
   others). At imgsz 1280 its detections went 31 → 167 on the same 300 frames. Check
   the main stream before touching anything in the identity layer for that camera.
5. **`max_exemplar`'s single-bad-crop risk** is unfalsified, only unobserved on 23
   provable strangers.
6. Labels are 3 people on 92 seconds. Extend them on the next capture.

---

## 1. State of the world

| Fact | Consequence |
|---|---|
| **The corpus is INTACT.** `persons`, **2048-d**, Cosine, GREEN, **7071 points** — printed by the pipeline itself: `Vector store ready at http://localhost:6333 (existing points: 7071)`. | Nothing was lost. Two "the store is empty" readings were artifacts, below. |
| `20260804_064551` holds **2238 observations**, with clips + sidecars for cam_213/219/224. | The only run carrying an operator-known split. **Protect it.** |
| `RUN SUMMARY`'s `Store: 0 observations` is **run-scoped**, not global. | `print_run_summary` scrolls for *this* `run_id`. A run whose writes all failed prints 0 while 7071 points sit untouched. Caused one false alarm. |
| A **port collision on 6333** between two checkouts. | `Inference-monday`'s container can win the port with its own empty volume, so `localhost:6333/dashboard` showed an empty 2048-d `persons` while the real data sat elsewhere. Caused the second false alarm. |
| `Inference-monday` embeds at **512-d**; the collection is **2048-d**. | Every write correctly 400s. Nothing is corrupted — a rejected write changes nothing. That checkout needs its own port and volume (Task 0). |
| `geometry.*` all false, no calibration record. | Geometry is **out of scope**: it can only refuse merges, so it cannot fix a split. |
| `covisibility.enabled: true`, 6 pairs, 1.0 s. | A hard cross-camera veto **is** live. Different switch from geometry — do not conflate. |
| `scoring: prototype`, `threshold: 0.63`. | Reverted from `consensus`/`0.45`, which measured worse. Correct as-is. |
| `same_camera_reciprocal_best: true`, `same_camera_rounds: false`, `same_camera_member_quorum: 1.0`. | Current same-camera policy. |

---

## 2. The finding that reframes everything

Same statistic, same model, same script: one tracklet split in half — provably one
person, one camera, one lighting condition. The **easiest case that exists**.

| corpus | same-person split-half | stranger | gap |
|---|---|---|---|
| `register_file.avi` | min **0.810** | max **0.434** (n=6) | **+0.38 clean** |
| the real cameras | min **0.594** (mean 0.806, n=18) | **0.843** co-present, cam_219 | **−0.25 INVERTED** |

cam_219's `11+57 = 0.843` is **provably two people** — one body cannot be two
simultaneous detections. The operator's own known re-appearance pair
`0020 vs 0008` scores **0.574**. A proven stranger outscores a known same-person
pair by 0.27, inside one camera.

**To merge one person you need a bar ≤ 0.594. To reject that stranger you need
> 0.843. No bar satisfies both.** `config.yaml`'s window — "any bar in
(0.434, 0.810) is perfect" — is not narrow on this footage. It is **empty**.

Two consequences:

1. **Every threshold in `config.yaml` was derived from `register_file.avi`**, a clip
   roughly 0.6 cosine easier than the deployment, and one `compare_backbones.py`
   already refuses to score because it saturates (FastReID and OSNet both hit
   prototype AUC 1.0000, R@1 48/48 there). The bars were fitted to the wrong
   distribution.
2. **Threshold tuning was never going to work**, in either direction. That explains
   six reverted tunings, and it is not a tuning mistake.

Note also the spread *within* cam_219: `cam_219:14 = 0.594`, `cam_219:7 = 0.950`.
Something varies by 0.35 inside a single track — 8× the measured scale effect, 3× the
measured blur effect.

---

## 3. Evidence ledger

### Eliminated by measurement. Do not revisit without new evidence.

| Hypothesis | Killed by |
|---|---|
| Geometry / floor homography | Can only *refuse* merges; structurally cannot fix a split. Both fit attempts failed on **coverage** (16 collinear points → 38% inliers, 203 px error; then 106 of 122 from a stationary tracklet → 8%), not on maths. |
| Cross-camera `threshold` | **Never consulted** for the reported split: once both clusters contain cam_219, `strictest_same_camera_bar` applies (0.90). Measured on `064551`. |
| Lowering cam_219 to 0.80 | **Contraindicated**, not merely insufficient — the sweep captured `cam_219(8)` into `cam_219(38)` instead of the operator's person. |
| `same_camera_rounds` as the fix | Fixture: every `rounds=True` row identical to `rounds=False` **at quorum 1.0**, because round 2's member-pair tests are a subset of round 1's. *(It does become load-bearing below quorum 1.0 — see the C1 grid.)* |
| `scoring: consensus` | One run; lowered all three fragment pairs and crossed cam_224 from 0.907 PASS to 0.582 FAIL. |
| **Crop scale / upscale factor** | Synthetic degradation: 2× → **0.008**, 4× → **0.022**, 6× → **0.045**. Against a 0.27 gap, an order of magnitude short. The −0.490 h_ratio correlation was confounded. **Do not build a size gate.** |
| **Blur / focus** | sigma 5 → **0.119**, and sigma 5 is heavy. Real, insufficient. Laplacian variance is **not scale-invariant**, so the cross-camera comparison (cam_219 84–653 vs cam_213 602–5439) is confounded by cam_219's crops being larger. Any focus claim needs height-matched bins. Also: the operator reports cam_219 and cam_224 are the **same build**, so an optics difference between them is ruled out by hardware. |
| Preprocessing | Checked: `cv2.resize` straight to 384×128 with no aspect preservation, and `crop_person` padding 0, are **exactly what FastReID trains on**. Faithful, not broken. No free win. |

### Live hypotheses, ranked. None of #1–#4 has been measured.

| # | Hypothesis | Status |
|---|---|---|
| 1 | **Prototype mean-pooling destroys view information.** A mean over front and back matches neither. | Very high on mechanism, **zero on measurement**. Task 1 settles it. |
| 2 | **cam_219 cannot discriminate people internally.** The 0.843-vs-0.574 inversion is *within* one camera. | High. **Distinct from #3** — do not merge them. |
| 3 | **Between-camera feature offset.** cam_219 translated relative to cam_224/213. | Plausible, but **cannot explain #2**: centering subtracts the same vector from both halves of a within-camera pair. |
| 4 | **Reconcile never re-ranks.** `reranking.py` is used by `service.py` and `verifier.py`; `reconcile.py` has zero references. The live path (discarded ids) gets it; the offline path (final ids) compares raw cosine between two means. | Real gap. **Feasibility answered — see Task 2.** |
| 5 | The embedding is the ceiling on this domain. | The fallback if Task 1 comes back negative. |
| 6 | Thresholds | Low as a primary cause. §2 explains why. |

`reranking.py` is not "cosine again": mean vectors → neighbour graph → Jaccard →
blended similarity. The graph step reads relational structure no pairwise cosine
sees. It cannot recover what averaging destroyed *within* a tracklet, but it can
recover structure *between* tracklets. Different claims; keep them apart.

---

## 4. Rules that outrank your instincts

Additional to `CLAUDE.md`, which applies in full.

1. **Propose no threshold change.** Six reverted. §2 explains why none can work.
2. **`main.py --reset` is a DESTRUCTIVE RUN** — clears the store *then* runs, so it
   looks ordinary. **Never use it.** Same for `preflight.py --fix-store`,
   `docker compose down -v`, `docker system prune`. Those 7071 points include the
   only run with operator-known ground truth.
3. **One change at a time**, judged on §2's same-track-vs-stranger separation, then on
   a watched re-render. Never on a cluster count (`CLAUDE.md` §4).
4. **Pre-register the falsification criterion** in the script, before running it. This
   worked — the scale/blur hypothesis was killed by its own stated threshold. Keep
   doing it.
5. **Anything that changes a score voids every bar.** Scoring mode, re-ranking,
   centering, backbone. Sweep after, never before.
6. **Do not build a third decision logger.** `decision_log.py`,
   `verification_decisions.jsonl`, `explain_merge_failure.py` and
   `analyze_decision_log.py` all exist.
7. **Never paste a placeholder.** `<run_id>` has been run verbatim; `<` is a shell
   redirect. Derive values.
8. **The dev box is CPU-only WSL2.** Any conclusion about throughput, identity quality
   or thresholds drawn there is wrong. All model runs on the A6000.

---

## 5. Tasks

### Task 0 — Protect the corpus, isolate the old checkout (first, 10 min)

```bash
# 1. Does the ground-truth run survive?
curl -s -X POST http://localhost:6333/collections/persons/points/count \
  -H 'Content-Type: application/json' \
  -d '{"exact":true,"filter":{"must":[{"key":"run_id","match":{"value":"20260804_064551"}}]}}' \
  | python3 -m json.tool

# 2. SNAPSHOT IT. Nothing else until this returns.
curl -X POST http://localhost:6333/collections/persons/snapshots | python3 -m json.tool
curl -s  http://localhost:6333/collections/persons/snapshots | python3 -m json.tool

# 3. Which container owns 6333, and where is its storage bound?
docker inspect $(docker ps -q --filter publish=6333) \
  --format '{{.Name}} {{json .Mounts}}' | python3 -m json.tool
du -sh ~/seifer_work/*/qdrant_storage* 2>/dev/null
```

Then give `Inference-monday` its own instance, so 512-d can never meet 2048-d:

```yaml
    container_name: qdra_monday
    ports:
      - "6335:6333"
      - "6336:6334"
    volumes:
      - ./qdrant_storage_monday:/qdrant/storage
```

```bash
echo 'QDRANT_URL=http://localhost:6335' >> .env      # Inference-monday ONLY
```

**Accept when:** the count is reported, a snapshot exists and is listed, and the two
checkouts cannot share a collection. Add the snapshot command to `CLAUDE.md` §8 beside
the preflight line — a run with ground truth is the most expensive artifact this
project produces and nothing currently protects it.

---

### Task 1 — THE DECIDING MEASUREMENT

Everything forks on this. Crops come from the clips: no camera time, no labels, no
headcount.

**Built: `tests/calibration/decide_view_vs_ceiling.py`** (parts 1 and 2) and
`tests/calibration/contact_sheet_halves.py` (part 3).

For the same 18 tracklets used in the split-half control:

1. **Temporal split-half** — first half vs second half. Compute **all three** modes:
   `prototype`, `max_exemplar`, `consensus`.
2. **Random split-half** — same tracklets, random partition, `prototype` only.
3. **Contact sheets** for `cam_219:14` (0.594) and `cam_219:7` (0.950): first-half
   crops on one row, second-half on another, one PNG each.

**Pre-registered decision rules, printed by the script before its results.**

| Observation | Conclusion | Next |
|---|---|---|
| `max_exemplar` on the low controls rises to **≥ 0.85** | Mean-pooling is destroying recoverable information | Task 4. No new model needed. |
| `max_exemplar` stays **< 0.70** | **No view of that person matches any other view.** The embedding is the ceiling. | Task 5. Re-ranking over these prototypes inherits the failure. |
| random ≈ 0.95 while temporal ≈ 0.59 | Appearance genuinely changes across the track — orientation | View-aware representation |
| random ≈ temporal ≈ 0.6 | Frame-to-frame instability, not view change | Different problem; investigate before proceeding |

**The contact sheet has THREE readings, not two:**

- the person turned around → orientation, #1 confirmed;
- **two different people → ByteTrack ID switch mid-track.** A tracker bug, and it
  means the *control is contaminated* rather than the model failing. **Check this
  first** — it also invalidates any tracklet built on that track;
- halves look identical and still score 0.594 → domain failure, #5.

**Accept when:** the three-mode table, the random-split column and both contact sheets
exist, and you state which pre-registered branch fired. **Then stop.** Do not
implement the consequence in the same pass.

---

### Task 2 — Reranker feasibility — **ANSWERED 2026-08-04, no code written**

**1. What does `CameraAwareReranker` take?** Plain vectors and camera labels — **not**
an `IdentityService`-shaped object. [reranking.py:89](src/identity/reranking.py#L89):

```python
def rerank(self, query_embedding, query_camera, candidate_gids, prototypes, cameras):
    """prototypes : {gid: L2-normalized prototype vector}, ALL known identities
       cameras    : {gid: primary camera string}
       candidate_gids : the subset we actually need re-ranked scores for."""
```

Constructor is three numbers ([:33](src/identity/reranking.py#L33)):
`CameraAwareReranker(k1=8, cross_camera_k1_boost=4, lambda_=0.7)`. No bank, no
gallery signature, no identity service.

Reconcile already holds both inputs: `protos` is `{(camera, track_id): vector}` and
the camera is `key[0]`. So this is the **"close to droppable" case**, not an adapter
rewrite. One caveat on shape: `rerank` is **query-vs-gallery**, while Phase 2 needs an
all-pairs matrix — so it is a loop of N calls, not one call. `_rebuild_if_needed`
([:40](src/identity/reranking.py#L40)) caches the neighbour graph across calls with
the same `prototypes`, so the loop is cheap at N≈20.

**2. Where is the small-gallery guard evaluated?** **Inside `rerank()`, per call** —
[reranking.py:114-119](src/identity/reranking.py#L114-L119):

```python
min_gallery_for_jaccard = self.k1 + 1        # 9 at k1=8
if len(prototypes) < min_gallery_for_jaccard:
    return {gid: float(prototypes[gid] @ q) for gid in candidate_gids if gid in prototypes}
```

So the brief's worst case — *"fabricates perfect Jaccard on a shrinking population,
which manufactures merges"* — **cannot happen.** The guard is re-evaluated on every
call and fails safe to plain cosine.

The other half of the concern stands but is small on current data. Phase 2's
population on run `20260804_094039`: 38 tracklets − 11 suppressed = 27, minus 4
Phase-1 merges = **23 clusters entering Phase 2**, converging to **18**. Both above
the floor of 9, so the guard would **not** fire on either surviving run. It becomes
live on a run with fewer than ~9 surviving clusters, where re-ranking silently
degrades to raw cosine — you would have built nothing, but broken nothing.

**Still untuned, and this is the real cost:** `lambda_: 0.7` leaves 70% of the blended
score as raw cosine in the broken space, and all three parameters were fitted for the
**live** path, over live identity prototypes, at a different population size, under
**OSNet**. Treat them as unknown for reconcile and sweep after wiring, never before.

**Honest estimate:** a few hours — a loop, a `lambda_`/`k1` sweep, and a re-render.
But it is **downstream of Task 1**: if Task 1's second branch fires, re-ranking is
built on prototypes that already lost the information, and the neighbour graph
inherits the failure. Do Task 1 first.

---

### Task 3 — The re-embedding script (the unlock behind Tasks 1, 4 and 5)

Clip + sidecar + `track_id` is a complete record, so any backbone can be replayed on
frozen footage with zero camera time. It regenerates crops for Task 1, enables a
threshold-free backbone A/B, and produces the crop set a fine-tune needs. It has been
the deferred prerequisite for four rounds.

Half a day on the A6000 — FastReID R101 at 384×128 batches in the hundreds of
crops/sec there; the 0.76 s/crop figure is CPU-only.

**Partially delivered:** `degrade_crops_causal.py`, `contact_sheet_halves.py` and
`decide_view_vs_ceiling.py` all read crops from clips + sidecars already. What is
still missing is the **write** half: re-embedding into a separate collection under a
configurable `reid.model`/`reid.weights`.

**Accept when:** it re-embeds a named run's clips under a configurable
`reid.model`/`reid.weights` into a **separate** collection, and
`verify_embedding_contract.py` passes.

---

### Task 4 — Conditional on Task 1: better aggregation

Only if Task 1's first branch fired. Cheapest first:

- **Quality-weighted aggregation.** The scalars are already computed and discarded.
- **Diversity-aware subsampling.** `reid.interval` is a *time* filter, not a diversity
  filter — consecutive frames are correlated, so averaging them encodes whatever bias
  is present rather than reducing variance.
- **Multi-prototype tracklets.** Cluster each tracklet's observations into 2–4
  medoids, score by best-matching medoid pair. **Guard it:** max-over-pairs inflates
  stranger scores too (the `other MAX` trap — 0.819 → 0.936 at 48 vs 90 frames on one
  clip), so use a high quantile rather than the max, keep `require_reciprocal_best`,
  and re-derive both bars.
- **Flip TTA** — embed the crop and its mirror, average. Free, standard, orthogonal
  to all of the above.

---

### Task 5 — Conditional on Task 1: stronger representation

Only if the second branch fired. Then `ARCHITECTURE.md` §6 was right from the start,
and ADR-003B was shelved on a premise now falsified — geometry and reconcile cannot
carry the accuracy if the appearance signal is inverted at source.

Cheapest first, each A/B'd threshold-free on your own clips via Task 3 plus
`compare_backbones.py --clips`:

- **Domain-generalizing backbone** — CLIP-ReID, SOLIDER, TransReID-SSL. One
  `reid.model` line plus a checkpoint. No training.
- **BPBreID / KPR** (`github.com/VlSomers/bpbreid`,
  `github.com/VlSomers/keypoint_promptable_reidentification`) — part-based with
  **visibility scores**, so a seated person's upper body compares against a standing
  person's upper body. The right answer to an office with desks. **But it breaks the
  storage contract**: multiple part vectors plus visibility per crop, not one
  L2-normalised vector. A schema *shape* change, not just a width change.
- **Unsupervised in-domain fine-tune** — Cluster Contrast on Task 3's crops, no labels.
  Two modifications matter enormously for a small cast: cluster on **camera-debiased**
  features, and **discard single-camera clusters** (published: Cluster Contrast
  29.8 → 49.1 mAP on MSMT17). Without them, single-camera clusters dominate.

---

### Task 6 — Small, independent, cheap

- **`reid.quality.max_aspect: 1.2`** admits near-square crops that get stretched ~3×
  vertically into a shape the model never saw. In an office that is the **seated**
  case, and it hits cam_219. Measure first: what fraction of cam_219's stored crops
  had aspect > 0.8, and do those observations sit in the low-scoring halves?
- **`main.py` passes no decision log**, so file-batch reconcile runs unlogged while the
  live path is logged. Record in `REMEDIATION_PLAN.md` Part M.
- **`_gather_tracklets`' scroll fallback is bare and unlogged** (`except Exception:
  scroll_filter = None`, twice). A silent fallback scans the whole collection
  client-side — the exact cost the filter was added to remove. Log both.
- **cam_213's clip showed 0 detections across 138 frames** on the old checkout.
  Separate thread; may not reflect the current detector config.
- The `same_camera_rounds` finding from §3 belongs beside that key in `config.yaml`,
  flagged as synthetic, because its comment currently promises it fixes the cam_206
  stranding.

---

## 6. Still genuinely open

- **Task 1's branch.** Everything forks on it.
- `206/12 · 206/26` — decides whether `same_camera_rounds` can fix the cam_206
  stranding at all. cam_206 is absent from every surviving run, so this needs the
  capture.
- What the 1961 `TEMPORAL_CONFLICT_CROSS_CAMERA` exclusions on run `20260731_060425`
  actually were. One line from `analyze_decision_log.py`. Background, not a lead —
  `explain_merge_failure.py` named the *appearance* bar for the reported split.
- Whether cam_219 is optically soft. Two seconds: open `._live_src_cam_219.mp4`.
  Note the clips are deliberately CLEAN frames (no boxes — those live in the sidecar,
  and the annotated video is `output_cam_*.mp4`), which is what you want for judging
  sharpness. Ruled out between cam_219 and cam_224 by hardware; still open vs cam_213.
- Whether the split is in reconciled ids or only live provisional ids. Check
  `output_cam_*.mp4` mtime against the `run_id` — stale files look like success.
- The capture (`CAPTURE_PROTOCOL.md`): cam_206, a written headcount, a known route,
  deliberate re-appearance repeats. Not blocking; new information.
