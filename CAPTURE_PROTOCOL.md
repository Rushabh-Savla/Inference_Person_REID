# CAPTURE PROTOCOL — the labelled re-appearance run

**Status: valuable, but NOT the critical path.** Corrected 2026-08-04. An earlier
revision of this document said the Qdrant collection was empty and nothing could be
measured until this run existed. That described the **dev box**. The A6000 holds
6017 points including run `20260804_064551` (2238 observations) with surviving
clips and sidecars for cam_213 / cam_219 / cam_224 — so the measurements run today,
and some of what is described below is partly recoverable from that footage by
watching it (`review_links.py --label`).

What this capture uniquely supplies, and nothing existing can:

- **cam_206**, absent from every run since `20260730_093723`;
- a **written headcount**, agreed before recording;
- a **known route** for a known person;
- **deliberate** re-appearance repeats, rather than whatever happened to occur.

Do it *after* the measurements on the existing run, so it is designed around what
those show. Budget: ~20 minutes of room time.

---

## 0. This is NOT the floor-frame calibration walk

Easy to conflate, so stated first. There is a separate, well-documented recording
protocol for fitting a floor homography — 8–12 distinct standing spots, spread
over the room, 3–4 s pauses, **nobody seated**, feet visible. **That is not this.**

Geometry ships disabled (`geometry.enabled: false`), no calibration record exists,
and geometry can only *refuse* merges so it cannot fix the reported split anyway.

**This capture has different requirements, and sitting is fine here** — it is what
the deployment actually looks like. Do not let anyone impose the calibration
walk's constraints on it.

---

## 1. What this run has to produce

Five jobs, one recording:

1. **A known headcount**, agreed and written down *before* the cameras start.
2. **A known identity** — you, with a route you recorded — so
   `explain_merge_failure.py` can be run on a split *you know* is one person, with
   no labelling ambiguity.
3. **Labelled same-camera re-appearance pairs.** The distribution nobody has ever
   measured. The published same-camera window (min 0.810) came from splitting one
   continuous track in half, which is the easy case; the real failure measured
   **0.574**. This run is how that distribution gets characterised.
4. **cam_206 back in.** Missing since `20260730_093723`, and the only camera whose
   detection recall has never been measured under `yolo11m`.
5. **Frozen clips** for unlimited offline replay (`keep_frames: true`, already set).

---

## 2. Pre-flight — before anyone enters the room

### 2.1 Land the `prototype`/`0.63` revert, then change nothing else

**Corrected 2026-08-04.** An earlier revision said "change no thresholds." That was
backwards: `HEAD` ships `scoring: consensus` and `threshold: 0.45`, the pair
measured to make the defect **worse** (cam_224's fragment pair crossing 0.907 PASS
→ 0.582 FAIL). Capturing against that would measure a configuration already known
to be wrong.

`config.yaml`'s own `scoring:` comment records that the operator's front/back
symptom is what *motivated* the consensus change — so the symptom was observed
under **prototype**, and `prototype`/`0.63` is the configuration worth measuring
against. Confirm the revert is committed and on the A6000 before capturing.

Then leave everything else alone: `same_camera_threshold`, `per_camera`,
`same_camera_rounds`, `same_camera_reciprocal_best`, `covisibility`. In particular
**do not** lower cam_219 to 0.80 — measured on `20260804_064551`, that captured
`cam_219(8)` into `cam_219(38)` instead of the operator's person.

### 2.2 Configure four cameras

Credentials in the untracked `.env`, never on the command line — a URL in `argv`
is visible to every user via `ps` and lands in shell history.

```
CAM_206=rtsp://...
CAM_213=rtsp://...
CAM_219=rtsp://...
CAM_224=rtsp://...
```

Then in `config.yaml`, `source.env_urls: [CAM_206, CAM_213, CAM_219, CAM_224]`.
It is currently `[]`, and `source.videos` points at `register_file.avi`, so
`--mode live` would read a **file**.

### 2.3 Preflight

```bash
python tools/preflight.py --load-model
```

Must exit zero. The check that matters most is the collection width: a collection
left at the previous backbone's 512-d accepts nothing at 2048-d, the store only
*warns*, and `IdentityStage` **swallows** the error — so the run would persist
nothing, reconcile would have nothing to reconcile, and no error would appear
anywhere. `--fix-store` rebuilds it and is destructive.

The run banner must print `fastreid_sbs_R101_ibn (2048-d, 384x128, post-bnneck, …)
on cuda:0`.

### 2.4 Confirm cam_206 actually detects

It has an open complaint of *"5 people present but only 1 detected at first."* A
20-second throwaway run with all four cameras, checking cam_206 produces
detections at all, is worth doing before the real capture — a cam_206 that detects
nothing wastes the whole recording.

### 2.5 Do not use `--reset`

It clears the store *and then runs the whole pipeline*. The store is already empty.

### 2.6 Disk

`keep_frames: true` costs ~1.2–2.2 MB/s per camera, so four cameras for 15 minutes
is roughly 5–8 GB. `max_clip_gb: 4.0` caps each camera. Confirm free space first —
hitting the cap silently truncates the offline re-render.

---

## 3. The route

Roughly 10–15 minutes. Longer is better for the stranger sample; the current
proven-distinct set is **n=6**.

### 3.1 Non-negotiable: the cam_219 re-appearance repeats

**At least four, ideally six.** This is the single most important part of the
capture.

For each repeat:

1. Stand in cam_219 **facing the camera**, still, 4–5 seconds.
2. Leave cam_219's field of view entirely.
3. Wait 15–30 seconds out of frame.
4. Return to cam_219 **facing away from the camera**, still, 4–5 seconds.

That produces a front fragment and a back fragment of one person in one camera,
time-disjoint — exactly the `0020 vs 0008` pair that scored 0.574. Six repeats give
six labelled positives in the distribution the project has never measured.

Vary the return: different position in the room, different distance, sitting on one
of them.

### 3.2 Cross-camera transits

Walk the loop several times, pausing 4–5 s in each camera:

```
cam_219  ->  cam_224  ->  cam_213  ->  cam_206  ->  back
```

cam_219 and cam_224 share a room, so include moments where you are **visible in
both at once** — that pair is configured `covisible` and is never vetoed, so those
are clean cross-camera positives.

### 3.3 Sit down

The deployment is an office. Sit at a desk for 60–90 seconds at least twice, in
different cameras, then get up and walk. Do not exclude this — it is the real
distribution.

### 3.4 Others in frame

At least one moment with **two people simultaneously in one camera**, held for
several seconds, in each camera you can manage. Same-frame co-occurrence is the
*only* proof of "different person" that needs no labels — one body cannot be two
simultaneous detections. Across cameras it proves nothing here, because cam_219
and cam_224 share a room.

More people = more proven-distinct pairs = a defensible stranger ceiling instead
of n=6.

---

## 4. The written record

**Write this down during the run, not after.** Without it the capture is just
another unjudgeable run.

- **Total headcount** — every person who entered any camera's view. Agreed before
  starting.
- **Roster** — who they were, and roughly what each was wearing. Two people in
  similar clothing is the hard case and worth noting.
- **Your timeline** — wall-clock time of each cam_219 exit and return, and which
  way you were facing on return. A phone voice memo works; live-camera `ts` is wall
  clock, so your notes align directly with the payloads.
- **Anything odd** — a camera that looked frozen, someone entering unplanned,
  lighting changing.

Put it in `calibration/` beside the run, not in a chat message. Labels belong in
`calibration/link_labels.jsonl` via `review_links.py --label` so they accumulate
with provenance — there are only **11** in the project's history, and they are
orphaned because the runs they referenced were wiped.

---

## 5. Running and stopping

```bash
python main.py --mode live
```

**Press Ctrl-C exactly once.** The most important work happens after you stop:
reconcile decides the cross-camera ids, then re-renders `output_<cam>.mp4`. Extra
presses are ignored with a `[guard]` line; if you truly must abort, Ctrl-\
(SIGQUIT), which aborts with provisional per-camera ids.

Capture the `run_id` the moment it prints — it is only printed when reconcile is
enabled, and every downstream command needs it. Note it in the written record.

Also confirm per-camera positioned/observation counts in the run summary look
sane, and that all four cameras appear.

---

## 6. Immediately after — before drawing any conclusion

```bash
cd "$(git rev-parse --show-toplevel)"
RUN=$(python - <<'PY'
import sys; sys.path.insert(0,'src')
from database.store import PersonVectorStore
s = PersonVectorStore(url='http://localhost:6333')
runs = {}
off = None
while True:
    pts, off = s.client.scroll(s.collection, limit=1000, offset=off,
                              with_payload=True, with_vectors=False)
    for p in pts:
        runs[(p.payload or {}).get('run_id')] = runs.get((p.payload or {}).get('run_id'), 0) + 1
    if off is None:
        break
print(max(runs, key=runs.get))
PY
)
echo "run_id = $RUN"
```

Derived, never pasted — `<run_id>` has been run verbatim before and `<` is a shell
redirect.

Then, **in this order**:

```bash
# 1. WATCH THE VIDEOS FIRST. Check mtime against the run_id -- output_cam_*.mp4
#    are only overwritten by a completed render, so stale files look like success.
ls -la output_cam_*.mp4

# 2. Does any one person carry more than one id? Does any one id cover more
#    than one person? Record the verdicts so they accumulate.
python tests/calibration/review_links.py "$RUN" --label

# 3. YOUR OWN SPLIT, if it reproduced. Substitute the two reids you actually
#    carry, read off the videos.
python tests/calibration/explain_merge_failure.py "$RUN" <reid_a> <reid_b>

# 4. The same-camera and cross-camera score landscape.
python tools/inspect_tracklet_pairs.py --run "$RUN"

# 5. The 1961-exclusion question: how many candidates did covisibility veto?
python tests/calibration/analyze_decision_log.py "logs/reconcile_decisions_$RUN.jsonl"

# 6. ONLY NOW sweep bars -- and sweep BOTH together; they are coupled through
#    the cluster prototypes, so one cannot be attributed alone.
python tests/calibration/sweep_reconcile_thresholds.py "$RUN" --cross 0.45,0.55,0.63 \
  --same "cam_219=0.90,cam_224=0.80 ; cam_219=0.80,cam_224=0.80 ; cam_219=0.65,cam_224=0.65"

# 7. Re-render the best candidate and WATCH IT before believing any number.
python tests/calibration/rerender_from_clips.py "$RUN" --cross <chosen> --same "<chosen>"
```

Step 3 is the one that matters most. It is what produced the 0.574 table, and this
time you will know for certain that the two reids are one person.

---

## 7. What would invalidate this capture

- **The capture ran with `scoring: consensus` / `threshold: 0.45`** — the revert was
  never landed, so the run measures a configuration already known to be worse.
- **Any other threshold was changed beforehand** — the run no longer measures the
  configuration that produced the symptom.
- **The headcount was not written down before starting** — every count-based
  conclusion becomes unprovable, which is exactly what happened to run
  `20260803_121136`.
- **cam_206 absent again** — its recall problem stays unmeasured for a third week.
- **Fewer than four cam_219 re-appearance repeats** — the primary deliverable is
  missing.
- **Preflight not run, or the collection at the wrong width** — the run persists
  nothing and reports no error.
- **Ctrl-C pressed repeatedly** — finalization aborts and the videos keep
  provisional per-camera ids.
- **Nobody watched the videos** — a cluster count cannot tell you whether a cluster
  is one person or three. Two settings previously chosen from numbers alone both
  made the output worse.
