# reconcile.py — patch set

Written against the text of `src/identity/reconcile.py` as pasted on 2026-08-04.
**Verify each `OLD` block matches your file before replacing it.**

Order matters: Part A first (no decision changes), then run
`python tests/run_all.py`. Part C is behaviour-changing and every item defaults to
the current behaviour exactly, so applying Part C changes nothing until you flip a
flag.

- **Part A** — settled from reading the code. No decision changes at all.
- **Part B** — a correction to an earlier claim of mine. No patch; read it.
- **Part C** — behaviour-changing, all gated to no-ops by default.

Run after Part A:

```bash
python tests/run_all.py
```

`verify_embedding_contract.py` is not needed — nothing here touches `src/reid/`.

---

# PART A — no decision changes

> **A2 and A3 are INERT — skip them.** `geometry_blocks` returns `False` on its
> first line when `geo_envelopes` is empty, and it is empty whenever
> `geometry.reconcile.enabled` is false — which it is, with no calibration record
> in existence. So A2's reordering never changes a reason and A3's cache never
> saves a call. They are correct, and they are the two lowest-value patches here.
> **Prefer the smaller diff: skip both.** Revisit only if geometry is ever turned
> on, at which point A2 matters (it stops a doubly-conflicting pair being
> attributed to the least-validated gate) and A3 is a real speedup.
>
> **Do not read "geometry is off" as "the physical vetoes are off."**
> `identity.reconcile.covisibility.enabled` is **true**, with all six camera pairs
> configured and non-covisible pairs at 1.0s. Separate config block, separate
> switch, separate code path — it uses only `ts` and a camera-pair table, no
> positions and no homography. It is live. See §6 of `AGENT_BRIEF.md` for its
> scope, which is narrower than it first appears.

## A1. Order observations by capture time, not Qdrant scroll order

**Why.** `scroll` returns points in point-id order. `frames` and `times` were
sorted; `vectors` and `points` were not. So `_subsample_rows`' `np.linspace`
sampling was evenly spaced in *id* space, not in time — its docstring's claim
("keeps the pose/lighting variety a random sample would keep") did not hold, and
"DETERMINISTIC, so a replay reproduces the same score" depended on the store's
internal ordering rather than on anything this module controls.

Secondary: after this function, `frames`/`times` were sorted independently of
`vectors`/`points`, so the four lists were **not index-aligned**. Nothing zips
them today. Something will.

Affects `MAX_EXEMPLAR` and `CONSENSUS` only (`PROTOTYPE` is order-invariant), so
this is a no-op on your current default — and a prerequisite for trusting either
other mode.

### OLD

```python
        frames = info["frames"]
        times = info["times"]
        out[key] = {
            "vectors": info["vectors"],
            "points": info["points"],
            "span": (min(frames), max(frames)),
```

### NEW

```python
        frames = info["frames"]
        times = info["times"]
        # OBSERVATION ORDER IS CAPTURE ORDER, not Qdrant scroll order. `scroll`
        # returns points in point-id order, so `vectors` / `points` arrived in an
        # order with no temporal meaning -- which made _subsample_rows' "evenly
        # spaced" sampling evenly spaced in ID space, and made its determinism
        # claim rest on the store's internal ordering rather than on anything this
        # module controls. Sorted by FRAME INDEX (always present; `ts` can be
        # missing), so `vectors`, `points` and `frames` are INDEX-ALIGNED.
        # `times` is deliberately NOT in that alignment -- it is only appended when
        # a usable ts exists, so it can be shorter -- and stays its own sorted list.
        order = sorted(range(len(frames)), key=lambda i: frames[i])
        vectors = [info["vectors"][i] for i in order]
        points = [info["points"][i] for i in order]
        frames = [frames[i] for i in order]
        out[key] = {
            "vectors": vectors,
            "points": points,
            "span": (frames[0], frames[-1]),
```

Then, further down in the same dict literal:

### OLD

```python
            "frames": sorted(frames),
```

### NEW

```python
            "frames": frames,          # already sorted above, with vectors/points
```

---

## A2. Check same-camera co-presence before geometry

**Why.** A pair that violates *both* rules was attributed to
`GEOMETRIC_UNREACHABLE`. Same-camera co-presence is the certain rule ("one body
cannot be two simultaneous detections"); the geometric veto has never run on real
data (ADR-003D changelog). Evaluating geometry first sends the next investigation
to the least-validated gate — the same class of misattribution you already fixed
in the Phase 1 instrumentation, re-entering via evaluation order.

Also cheap-before-expensive: `_spans_disjoint` is two comparisons,
`geometry_blocks` is a binary search plus up to `MAX_GEOMETRY_PAIRINGS` envelope
evaluations.

`conflict()` only reads truthiness, so **no merge decision changes** — only the
reason string, and only for pairs that violate both.

### OLD

```python
        for a in set_a:
            for b in set_b:
                if geometry_blocks(a, b):
                    return dlog.GEOMETRIC_UNREACHABLE
                if a[0] == b[0]:
                    if not _spans_disjoint(tracklets[a]["span"],
                                           tracklets[b]["span"]):
                        return dlog.TEMPORAL_CONFLICT_SAME_CAMERA
                    continue
```

### NEW

```python
        for a in set_a:
            for b in set_b:
                # CERTAIN AND CHEAP RULE FIRST. Two comparisons, and the claim is
                # provable. Geometry costs a binary search plus up to
                # MAX_GEOMETRY_PAIRINGS envelope evaluations and, as of 2026-08-04,
                # has never run on real data. Evaluating it first attributed every
                # pair that violated BOTH rules to geometry, pointing the next
                # investigation at the least-validated gate.
                if a[0] == b[0] and not _spans_disjoint(tracklets[a]["span"],
                                                        tracklets[b]["span"]):
                    return dlog.TEMPORAL_CONFLICT_SAME_CAMERA
                if geometry_blocks(a, b):
                    return dlog.GEOMETRIC_UNREACHABLE
                if a[0] == b[0]:
                    continue
```

---

## A3. Cache fail-open geometry verdicts

**Why.** `if detail is not None` means every fail-open path (no recorded
positions, mixed floor groups, unmeasured group, `< MIN_GEOMETRY_PAIRINGS`)
memoises **nothing** and is fully recomputed on every call — nested member loops,
every Phase 2 round, plus once per pair in the instrumentation. Fail-open is
currently the *common* case. Kept in a separate set so `geo_details` stays exactly
what the summary reports: pairs geometry could actually judge.

### OLD

```python
    geo_details = {}          # (key_a, key_b) -> detail dict, for the log/summary
    geo_vetoes = 0
```

### NEW

```python
    geo_details = {}          # (key_a, key_b) -> detail dict, for the log/summary
    geo_unavailable = set()   # pairs geometry could NOT judge -- cached, see below
    geo_vetoes = 0
```

### OLD

```python
        pair = (a, b) if a <= b else (b, a)
        if pair in geo_details:
            return geo_details[pair].get("verdict") == IMPOSSIBLE
        impossible, detail = reachability_verdict(
            tracklets[a].get("floor"), tracklets[b].get("floor"), geo_envelopes)
        if detail is not None:
            geo_details[pair] = detail
        if impossible:
            geo_vetoes += 1
        return impossible
```

### NEW

```python
        pair = (a, b) if a <= b else (b, a)
        if pair in geo_details:
            return geo_details[pair].get("verdict") == IMPOSSIBLE
        # FAIL-OPEN VERDICTS ARE CACHED TOO. reachability_verdict returns
        # detail=None for every fail-open path -- no recorded positions, mixed floor
        # groups, an unmeasured group, too few pairings -- and those are the COMMON
        # cases today. Caching only judged pairs meant recomputing each of those on
        # every call. Separate set so geo_details stays "pairs geometry judged",
        # which is what the run summary reports.
        if pair in geo_unavailable:
            return False
        impossible, detail = reachability_verdict(
            tracklets[a].get("floor"), tracklets[b].get("floor"), geo_envelopes)
        if detail is None:
            geo_unavailable.add(pair)
        else:
            geo_details[pair] = detail
        if impossible:
            geo_vetoes += 1
        return impossible
```

---

## A4. Log and clear degenerate prototypes

**Why.** A zero-norm prototype dropped the tracklet from `keys` silently: it never
entered `remap`, its points were never `clear_global_id`'d, so it kept whatever
provisional gid the live run gave it — with no line anywhere saying so. Exactly
the silent-failure class `tools/preflight.py` exists to catch.

### OLD

```python
    protos = {k: _prototype(tracklets[k]["vectors"]) for k in keys}
    keys = [k for k in keys if protos[k] is not None]
```

### NEW

```python
    protos = {k: _prototype(tracklets[k]["vectors"]) for k in keys}
    # A zero-norm prototype cannot be scored, so the tracklet cannot be clustered.
    # It used to vanish from `keys` SILENTLY -- never entering `remap`, its points
    # never cleared, keeping whatever provisional gid the live run gave it, with
    # nothing in the output saying so.
    for k in [k for k in keys if protos[k] is None]:
        store.clear_global_id(tracklets[k]["points"])
        log(f"  tracklet reconcile: DEGENERATE prototype for {k} "
            f"({len(tracklets[k]['vectors'])} obs, unnormalizable) -> id cleared; "
            f"it cannot be clustered")
        _record_outcome(k, dlog.SUPPRESSED)
    keys = [k for k in keys if protos[k] is not None]
```

---

## A5. Fix the Phase 1 instrumentation key mismatch

**Why.** `accepted_same` holds **root** pairs at merge time; the instrumentation
looks up **tracklet** keys. Identical on round 1 (every root is a singleton) and
divergent the moment `same_camera_rounds` is on — so `accepted_partner` was
silently wrong for exactly the multi-round merges rounds were added to produce.

Log-only; cannot change a decision. Note the semantic shift, which is deliberate:
"did this subject and its best peer end up as one identity in Phase 1" rather than
"was this exact root pair accepted", which is the meaningful question for a
subject-centric record.

Insert immediately **before** the `if dl is not None:` block that begins the
Phase 1 instrumentation (i.e. after the `same_camera_reciprocal_best` log):

### NEW (insert)

```python
    # Phase 1 cluster membership, captured BEFORE Phase 2 touches it. The
    # instrumentation below iterates TRACKLET keys, while `accepted_same` holds
    # ROOT pairs as they were at merge time -- identical on round 1, divergent once
    # same_camera_rounds is on. Log-only.
    same_camera_root = {k: find(k) for k in keys}
```

### OLD

```python
            partner = None
            if best is not None:
                bkey = next(b for b in peers if handles[b] == best.handle)
                partner = (best.handle if ((a, bkey) in accepted_same
                                           or (bkey, a) in accepted_same) else None)
```

### NEW

```python
            partner = None
            if best is not None:
                bkey = next(b for b in peers if handles[b] == best.handle)
                partner = (best.handle
                           if same_camera_root[a] == same_camera_root[bkey]
                           else None)
```

---

## A6. `reid_id` present-and-null falls through to `global_id`

### OLD

```python
            gid = pl.get("reid_id", pl.get("global_id"))
```

### NEW

```python
            # Explicit, not a dict default: a payload carrying reid_id=null would
            # take the None and never fall through to global_id.
            gid = pl.get("reid_id")
            if gid is None:
                gid = pl.get("global_id")
```

---

## A7. `resolve_covisibility` — type-check instead of duck-typed indexing

**Why.** A YAML string where a list was meant indexes *cleanly* — `"abc"[0]`,
`[1]`, `[2]` give `'a'`, `'b'`, `'c'` — so a typo silently became a camera pair
named after two characters. A dict raises `KeyError`, which is **not** in the
`except` clause, so that case crashed the run instead of failing soft. Both now
take the same visible path.

### OLD

```python
    for entry in (cfg.get("pairs") or []):
        try:
            a, b, spec = entry[0], entry[1], entry[2]
        except (TypeError, IndexError):
            log(f"  [reconcile] covisibility: skipping malformed pair {entry!r} "
                f"(expected [cam_a, cam_b, seconds|'covisible']).")
            continue
```

### NEW

```python
    for entry in (cfg.get("pairs") or []):
        # EXPLICIT TYPE CHECK. A bare string indexes cleanly ("abc" -> 'a','b','c')
        # so a YAML typo silently became a camera pair named after two characters;
        # a dict raised KeyError, which was not caught, so it killed the run.
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            log(f"  [reconcile] covisibility: skipping malformed pair {entry!r} "
                f"(expected a 3-item list [cam_a, cam_b, seconds|'covisible']).")
            continue
        a, b, spec = entry[0], entry[1], entry[2]
```

---

## A8. `describe_reconcile_kwargs` — print everything that moves a number

**Why.** Its own stated purpose is that "a sweep's output can never again be read
as if it described production." `safety_factor` — the one knob ADR-003D says to
raise when in doubt — and `max_observations_per_side` were both absent.

### OLD

```python
        + f" geometry={'on' if (kw.get('geometry') or {}).get('enabled') else 'OFF'}")
```

### NEW

```python
        + f" cap={kw.get('max_observations_per_side')}"
        + (f" geometry=on/safety="
           f"{(kw.get('geometry') or {}).get('safety_factor')}"
           f"/clock={(kw.get('geometry') or {}).get('clock_error_sec')}"
           if (kw.get('geometry') or {}).get('enabled') else " geometry=OFF"))
```

---

## A9. Signature default `min_tracklet_observations`: 1 → 3

**Why.** The resolver defaults to 3; the signature to 1. The drift you documented
and fixed is still latent for any caller that bypasses `resolve_reconcile_kwargs`.

**This one can break existing tests.** If a test in `tests/live/` relied on the
`1` default, it was pinning the drift — that break is informative, not a
regression. Read it before "fixing" it.

### OLD

```python
                        min_tracklet_observations=1, log=print,
```

### NEW

```python
                        min_tracklet_observations=3, log=print,
```

---

## A10. Two stale comments

### OLD

```python
    """Seconds two wall-clock spans overlap (0.0 when they do not, None if either
    span is missing). A single-observation tracklet has a zero-length span, so it
    can never register an overlap -- consistent fail-open."""
```

### NEW

```python
    """Seconds two wall-clock spans overlap. NEGATIVE when they do not (the size of
    the gap), None if either span is missing -- callers compare `> tol`, so a
    negative value fails every tolerance, which is the intended fail-open. A
    single-observation tracklet has a zero-length span, so it can never register an
    overlap."""
```

### OLD

```python
        # Prototypes are unit vectors and the shipped embedding is post-ReLU, so a
        # cosine bar outside [0, 1] is a typo, not a choice.
```

### NEW

```python
        # A same-camera bar outside [0, 1] is a typo, not a choice -- an unnoticed
        # `80` where `0.80` was meant disables same-camera merging for that camera
        # entirely, which is the failure this whole resolver exists to prevent.
        # NOTE the old justification here -- "the shipped embedding is post-ReLU" --
        # is NO LONGER TRUE. FastReID's post-bnneck features are ~60.5% negative
        # dimensions, so a cosine can legitimately be negative. The clamp stays as a
        # typo guard; nothing may assume non-negativity from it.
```

---

# PART B — a correction to my earlier review

My code review claimed the `best_partner` / `pair_bar_now` inconsistency could
**permanently strand** a cluster, because "if that was the round's only accepted
pair, `merged_this_round` stays `False` and the loop breaks."

**That was wrong, and I'm withdrawing it.** Tracing it properly:

`pair_bar_now(a, b)` differs from `pair_threshold(a, b)` only when live membership
differs from the round snapshot — which requires a **successful `union` earlier in
the same round**. So a bar-raised skip *implies* `merged_this_round == True`, the
loop cannot break on that path, and the next round re-snapshots, re-scores and
re-derives mutual-best with correct bars. The pair gets a fresh, correct
evaluation.

I also suggested "compute `best_partner` against `pair_bar_now`" as a fix. That is
a **no-op**: `best_partner` is built before any merge in the round, at which point
both bar functions agree by construction.

What survives is smaller and self-healing: a bar-raised skip costs both sides
their reciprocity slot for that round, so a third cluster that was a viable
partner waits an extra round. Correct, just slower. Worth **seeing** rather than
fixing:

### OLD

```python
            if s < bar:
                log(f"  tracklet reconcile: SKIP {a} + {b} (cosine {s:.3f} < "
                    f"{bar:.2f}) -- an earlier merge this round made this a "
                    f"same-camera claim (#27)")
                continue
```

### NEW

```python
            if s < bar:
                # Not lost -- the next round re-scores from current membership and
                # re-derives mutual-best, so this pair is re-judged against the bar
                # that actually applies. The cost is that both sides spent this
                # round's reciprocity slot on each other, so any third cluster that
                # was a viable partner waits a round too.
                log(f"  tracklet reconcile: SKIP {a} + {b} (cosine {s:.3f} < "
                    f"{bar:.2f}) -- an earlier merge this round made this a "
                    f"same-camera claim (#27); retried next round")
                continue
```

This is what a synthetic fixture would have caught in seconds, which is the
argument for `tests/live/test_same_camera_chain.py` (shipped alongside this file)
over a third round of static review.

---

# PART C — behaviour-changing, all default to today's behaviour

## C1. `all_member_pairs_clear` as a quorum (default 1.0 = unchanged)

**Why.** Requiring *every* cross-member pair to clear the bar is complete-linkage
clustering. Both counterexamples in your comment are **two-member** cases, where
complete linkage and "no weak edge" coincide. At ≥3 members they diverge.

The sharp version: **round 2's member-pair tests are a subset of round 1's.** In
round 1 every cluster is a singleton, so every same-camera pair was already tested
pairwise against the bar. In round 2, admitting F3 to {F1,F2} re-tests F1·F3 and
F2·F3 — the same two numbers. So `same_camera_rounds` can only change outcomes via
mutual-best re-derivation; it can **never admit an edge round 1 rejected on
score**.

Applied to your own cam_206 measurement, `same_camera_rounds=True` fixes the
stranding only if `206/12 · 206/26 ≥ 0.90` — a number that appears nowhere in the
comment, the plan reference, or the measurement. `tools/inspect_tracklet_pairs.py`
prints it.

`quorum=1.0` reproduces current behaviour exactly.

### OLD

```python
    def all_member_pairs_clear(ma, mb, bar):
        for x in ma:
            for y in mb:
                if pair_score({x}, {y}, protos[x], protos[y]) < bar:
                    return False
        return True
```

### NEW

```python
    def all_member_pairs_clear(ma, mb, bar, quorum=None):
        """Does the member-pair agreement requirement hold for this cluster pair?

        quorum=1.0 (the default) is COMPLETE LINKAGE: every cross-member pair must
        clear the bar. That is what shipped, and both counterexamples justifying it
        are TWO-member cases, where complete linkage and "no weak edge" are the same
        rule. At three or more members they are not, and complete linkage forbids
        the case the scoring-modes comment says the design is for: a chain
        F1(front) .. Fn(back) where every adjacent pair clears the bar but the ends
        do not can never consolidate, because admitting Fn requires F1 x Fn.

        Note also that round 2's member-pair tests are a SUBSET of round 1's -- in
        round 1 every cluster is a singleton, so every same-camera pair was already
        tested pairwise. So same_camera_rounds can only change outcomes via
        mutual-best re-derivation; it can never admit an edge round 1 rejected on
        score. A quorum below 1.0 is what actually opens the chain case, and its
        cost is that a fraction of members may disagree -- so it trades directly
        against the weak-edge capture this guard was added to stop. Sweep it; do not
        guess it.
        """
        if quorum is None:
            quorum = same_camera_member_quorum
        pairs = [(x, y) for x in ma for y in mb]
        if not pairs:
            return True
        clear = sum(1 for x, y in pairs
                    if pair_score({x}, {y}, protos[x], protos[y]) >= bar)
        if quorum >= 1.0:
            return clear == len(pairs)
        return clear >= max(1, int(round(len(pairs) * quorum)))
```

Add the parameter to the signature:

### OLD

```python
                        same_camera_reciprocal_best=False,
                        same_camera_rounds=False,
```

### NEW

```python
                        same_camera_reciprocal_best=False,
                        same_camera_rounds=False,
                        same_camera_member_quorum=1.0,
```

And to the resolver:

### OLD

```python
        "same_camera_rounds": bool(recon.get("same_camera_rounds", False)),
```

### NEW

```python
        "same_camera_rounds": bool(recon.get("same_camera_rounds", False)),
        "same_camera_member_quorum": float(
            recon.get("same_camera_member_quorum", 1.0)),
```

And to the summary line, since it moves a number:

### OLD

```python
        f" same_rounds={'on' if kw.get('same_camera_rounds') else 'OFF'}"
```

### NEW

```python
        f" same_rounds={'on' if kw.get('same_camera_rounds') else 'OFF'}"
        f" member_quorum={kw.get('same_camera_member_quorum', 1.0)}"
```

---

## C2. Per-camera-mean feature centering (default off)

**Why.** `PER_CAMERA_KEYS` already notes that `threshold` "is a property of a
camera PAIR, not a camera." Camera-mean centering attacks the same problem from
the feature side rather than the bar side: if one camera's features carry a
consistent offset, cosine on uncentered vectors is dominated by that offset and
the metric fails to express a link the data does contain.

This is the cheapest possible home for the experiment: between
`_gather_tracklets` returning and `protos`/`rows` being built you have the whole
gallery in memory with camera labels attached. It is **in-memory only** — nothing
is written back to Qdrant, so it is non-destructive and re-runnable.

**Two things this can get wrong on your deployment**, both handled:

- **Small cast.** If a camera only ever sees three people, its "camera mean" is
  substantially *their* mean and subtracting it deletes identity information. The
  damage appears as proven-distinct **within-camera** pairs moving closer. Hence
  `min_samples` (25 is the published floor below which it degrades) and `dims`
  (centre only the top-k highest-variance dimensions, which bounds the damage).
- **Norm.** Centred vectors are not unit norm, so cosine ≠ dot product and every
  bar moves. Re-normalized here, and **re-derive the bars after** turning it on.

Add near the other module-level helpers:

### NEW (insert, e.g. after `_prototype`)

```python
def center_vectors_per_camera(tracklets, dims=0, min_samples=25, log=print):
    """Subtract each camera's mean feature from its observations, in memory only.

    A per-camera translation in feature space makes tracklets from one camera
    retrieve each other rather than the same identity elsewhere, and cosine on
    uncentered vectors is dominated by that shared direction -- so the link can be
    present in the data and unreadable by the metric. Centering removes the
    translation; it does not change the model.

    NOTHING IS WRITTEN BACK TO THE STORE. This mutates the in-memory tracklet dict
    for this reconcile only, so it is non-destructive and re-runnable, and the
    gallery stays comparable across experiments.

    dims        -- 0 centres every dimension. N > 0 centres only the N dimensions
                   with the highest across-camera variance of the camera means,
                   which is where the effect concentrates. Use this when a camera
                   sees few people: it bounds how much identity information a
                   camera mean can absorb.
    min_samples -- a camera with fewer observations than this is LEFT ALONE, since
                   its mean is mostly noise (and mostly whoever happened to be
                   there). Reported, never silent.

    Returns the set of cameras actually centred.
    """
    by_cam = defaultdict(list)
    for key, info in tracklets.items():
        by_cam[key[0]].extend(info["vectors"])
    if len(by_cam) < 2:
        log("  camera centering: fewer than 2 cameras in this run -- nothing to "
            "centre against; skipped.")
        return set()

    means, skipped = {}, []
    for cam, vecs in by_cam.items():
        if len(vecs) < min_samples:
            skipped.append((cam, len(vecs)))
            continue
        means[cam] = _unit_rows(vecs).mean(axis=0)
    for cam, n in sorted(skipped, key=lambda t: str(t[0])):
        log(f"  camera centering: {cam} has {n} observation(s) < {min_samples} "
            f"-> NOT centred (its mean would be noise, and mostly whoever "
            f"happened to be in frame)")
    if len(means) < 2:
        log("  camera centering: fewer than 2 cameras have enough observations "
            "-> skipped entirely.")
        return set()

    mask = None
    if dims and dims > 0:
        stacked = np.stack([means[c] for c in sorted(means, key=str)])
        var = stacked.var(axis=0)
        keep = np.argsort(var)[::-1][:int(dims)]
        mask = np.zeros(stacked.shape[1], dtype=bool)
        mask[keep] = True
        log(f"  camera centering: restricted to the {int(dims)} highest-variance "
            f"dimension(s) of the camera means (of {stacked.shape[1]}) -- bounds "
            f"how much identity information a camera mean can absorb")

    cams = sorted(means, key=str)
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            log(f"  camera centering: cos(mean_{a}, mean_{b}) = "
                f"{float(means[a] @ means[b] / (np.linalg.norm(means[a]) * np.linalg.norm(means[b]) + 1e-12)):.4f}")

    for key, info in tracklets.items():
        m = means.get(key[0])
        if m is None:
            continue
        offset = m if mask is None else np.where(mask, m, 0.0)
        out = []
        for v in info["vectors"]:
            v = np.asarray(v, dtype=np.float32)
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            v = v - offset
            n = float(np.linalg.norm(v))
            # Degenerate only if an observation sat exactly on its camera mean.
            out.append(v / n if n > 1e-12 else np.asarray(info["vectors"][0],
                                                          dtype=np.float32))
        info["vectors"] = out
    log(f"  camera centering: ON -- centred {len(means)} camera(s). "
        f"EVERY THRESHOLD IS NOW IN A DIFFERENT SCALE; re-derive the bars with "
        f"sweep_reconcile_thresholds.py before reading anything into a number.")
    return set(means)
```

Call it immediately after gathering:

### OLD

```python
    dl = decision_log
    tracklets = _gather_tracklets(store, run_id)
    all_keys = sorted(tracklets)
```

### NEW

```python
    dl = decision_log
    tracklets = _gather_tracklets(store, run_id)
    # Feature-side companion to the per-camera BARS above: bars compensate for a
    # camera's score distribution, centering removes the offset that distorted it.
    # In-memory only; nothing is written back to the store. Off by default because
    # it changes every score in the run.
    ccfg = camera_centering or {}
    if ccfg.get("enabled"):
        center_vectors_per_camera(
            tracklets,
            dims=int(ccfg.get("dims", 0)),
            min_samples=int(ccfg.get("min_samples", 25)),
            log=log)
    all_keys = sorted(tracklets)
```

Signature and resolver:

### OLD

```python
                        geometry=None,
                        quality_out=None):
```

### NEW

```python
                        geometry=None,
                        camera_centering=None,
                        quality_out=None):
```

### OLD

```python
        "geometry": resolve_geometry_policy(cfg, log=log),
    }
```

### NEW

```python
        "geometry": resolve_geometry_policy(cfg, log=log),
        "camera_centering": {
            "enabled": bool((recon.get("camera_centering") or {})
                            .get("enabled", False)),
            "dims": int((recon.get("camera_centering") or {}).get("dims", 0)),
            "min_samples": int((recon.get("camera_centering") or {})
                               .get("min_samples", 25)),
        },
    }
```

And the summary line:

### NEW (append inside `describe_reconcile_kwargs`)

```python
        + (" centering=on/dims="
           f"{(kw.get('camera_centering') or {}).get('dims')}"
           if (kw.get('camera_centering') or {}).get('enabled')
           else " centering=OFF")
```

Config block (add under `identity.reconcile`):

```yaml
    camera_centering:
      enabled: false      # in-memory per-camera mean subtraction. Turning this ON
                          # voids every bar below -- re-derive with the sweep.
      dims: 0             # 0 = all dimensions; N = only the N highest-variance
                          # dimensions of the camera means (small-cast mitigation)
      min_samples: 25     # a camera below this is left uncentred
```

**How to judge it** — not on identity count. Two numbers, both from
`tools/inspect_tracklet_pairs.py` before and after:

1. same-person cross-camera cosine (known identity) must go **up**;
2. proven-distinct **within-camera** cosine must not go up with it.

If both rise together, the camera mean is absorbing identity information — set
`dims` to a few dozen and re-measure, or stop.

---

## C3. A size-stable consensus mode — **DO NOT APPLY**

> **Superseded 2026-08-04 by evidence already in `config.yaml`.** `scoring:
> consensus` ran for exactly one run (`20260804_064551`) and made the reported
> defect **worse**: `explain_merge_failure.py` showed it lowered all three
> fragment-pair scores, and cam_224's pair crossed **PASS → FAIL** (0.907
> prototype → 0.582 consensus against a 0.80 bar). Sixth reverted tuning change.
> A size-stable variant might behave differently, but the evidence points away
> from the consensus family, and `max_exemplar` was weakly best on every row *and*
> is what the live engine already uses.
>
> Kept below for the record only. If you try anything here, try `max_exemplar` —
> and note it moves the blocking cam_219 pair only 0.574 → 0.600, still far below
> any defensible bar, so it is not the fix either.

### Original text (not to be applied)

**Why.** `CONSENSUS`'s `k = round(|A|·|B|·top_frac)` scales with the *product* of
cluster sizes, so the statistic's meaning shifts as Phase 2 clusters grow, and
saturates once both sides hit `cap`. Your own reasoning for 0.25 over 0.5 was
about *what fraction of view pairs match* — a property of the two tracklets, not
of the product of their lengths. A bar derived on singleton pairs will not
transfer to cluster pairs.

Added as a **new mode** rather than changing `CONSENSUS`, so no existing bar is
silently voided.

### OLD

```python
PROTOTYPE = "prototype"
MAX_EXEMPLAR = "max_exemplar"
CONSENSUS = "consensus"
SCORING_MODES = (PROTOTYPE, MAX_EXEMPLAR, CONSENSUS)
```

### NEW

```python
PROTOTYPE = "prototype"
MAX_EXEMPLAR = "max_exemplar"
CONSENSUS = "consensus"
# Size-stable variant of CONSENSUS. `consensus` takes the top fraction of the FULL
# |A|x|B| product, so its meaning drifts as Phase 2 clusters grow and saturates at
# `cap`. This one asks each observation on the SMALLER side for its best match on
# the other, then averages the top fraction of THOSE -- so k scales with
# min(|A|,|B|) and a bar derived on singleton pairs still means the same thing
# between clusters. Its own scale, so its own bars.
CONSENSUS_BEST = "consensus_best"
SCORING_MODES = (PROTOTYPE, MAX_EXEMPLAR, CONSENSUS, CONSENSUS_BEST)
```

### OLD

```python
    if mode == CONSENSUS:
        flat = np.sort(sims.ravel())[::-1]
        k = max(1, int(round(flat.size * float(top_frac))))
        return float(flat[:k].mean())
```

### NEW

```python
    if mode == CONSENSUS:
        flat = np.sort(sims.ravel())[::-1]
        k = max(1, int(round(flat.size * float(top_frac))))
        return float(flat[:k].mean())
    if mode == CONSENSUS_BEST:
        # Best partner per observation on the smaller side, then the top fraction
        # of those -- k scales with min(|A|, |B|), not the product.
        best = sims.max(axis=1) if sims.shape[0] <= sims.shape[1] else sims.max(axis=0)
        best = np.sort(best)[::-1]
        k = max(1, int(round(best.size * float(top_frac))))
        return float(best[:k].mean())
```

---

# Order of operations

1. Commit first. `git add -A && git commit -m "pre-patch checkpoint"`.
2. Apply **Part A**, run `python tests/run_all.py`. Investigate any A9 break
   before changing it.
3. Run `tools/inspect_tracklet_pairs.py` — that is where the numbers that decide
   C1 and C2 come from.
4. Run `tests/live/test_same_camera_chain.py` — settles C1's grid with no data.
5. Apply **Part C**. Nothing changes until you edit `config.yaml`.
6. Any flag you flip **voids the bars**. Sweep, re-render, and *watch it*
   (`rerender_from_clips.py`) before believing an identity count.
