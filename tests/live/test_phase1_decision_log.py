"""
Phase 1 acceptance: the reconcile decision log.

The load-bearing test here is the FIRST one -- reconcile must produce byte-identical
assignments with and without a decision log attached. Phase 1 is instrumentation
only, so anything else is a regression regardless of how good the diagnostics look.

Also covers the plan's acceptance criteria:
  3 -- a subject failing n gates has n entries in gates_failed (two-gate fixture)
  4 -- gate_detail is populated for every gate on every record
  6 -- Candidate<->Candidate merge: two sub-threshold tracklets that resolve only
       when merged (this is reconcile's ONLY mechanism -- there is no gallery)
  9 -- the TOP2_MARGIN summary matches recomputation from the candidate vector
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.pop("QDRANT_URL", None)
os.environ.pop("QDRANT_API_KEY", None)

from database.store import EMBEDDING_DIM as DIM, PersonVectorStore
from identity import decision_log as dlog
from identity.decision_log import DecisionLog, verify_summary_matches_vector
from identity.reconcile import reconcile_tracklets

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def build_store(td, people, jitter=0.02, seed=0):
    """people = [(camera, track_id, base_index, n_obs, frame_start), ...]"""
    rng = np.random.default_rng(seed)
    bases = {}
    st = PersonVectorStore(path=os.path.join(td, "q"), url=None)
    for cam, tid, base_ix, nobs, f0 in people:
        if base_ix not in bases:
            bases[base_ix] = _unit(rng.normal(size=DIM))
        for i in range(nobs):
            # jitter is a RELATIVE noise magnitude, so it must be divided by
            # sqrt(DIM): rng.normal(size=DIM) has norm ~sqrt(DIM), so without the
            # scaling the perturbation grows with the embedding width and swamps
            # a unit base vector. At 512-d the unscaled version happened to leave
            # same-person cosine ~0.91; at 2048-d it fell to ~0.74 and the
            # cross-camera merge this test asserts stopped happening. Same lesson
            # _synth.observe() already documents -- see its docstring.
            v = _unit(bases[base_ix]
                      + (jitter / np.sqrt(DIM)) * rng.normal(size=DIM))
            st.add_many([v], [{"camera": cam, "track_id": tid,
                               "frame": f0 + i * 10, "run_id": "R"}])
    return st


def run(st, log=None, **kw):
    kw.setdefault("threshold", 0.63)
    kw.setdefault("run_id", "R")
    kw.setdefault("same_camera_threshold", 0.90)
    kw.setdefault("require_reciprocal_best", True)
    kw.setdefault("min_tracklet_observations", 3)
    kw.setdefault("log", lambda *_: None)
    return reconcile_tracklets(st, decision_log=log, **kw)


# ---------------------------------------------------------------------------
print("\n1. BEHAVIOUR IS UNCHANGED when a decision log is attached")

SCENES = {
    "two distinct people, one camera": [
        ("cam_a", 1, 0, 6, 0), ("cam_a", 2, 1, 6, 500)],
    "same person fragmented, one camera": [
        ("cam_a", 1, 0, 6, 0), ("cam_a", 2, 0, 6, 500)],
    "same person across two cameras": [
        ("cam_a", 1, 0, 6, 0), ("cam_b", 7, 0, 6, 0)],
    "three people, two cameras": [
        ("cam_a", 1, 0, 6, 0), ("cam_a", 2, 1, 6, 500),
        ("cam_b", 7, 0, 6, 0), ("cam_b", 8, 2, 6, 0)],
    "time-overlapping same camera (must never merge)": [
        ("cam_a", 1, 0, 6, 0), ("cam_a", 2, 0, 6, 0)],
    "a suppressed short tracklet": [
        ("cam_a", 1, 0, 6, 0), ("cam_a", 2, 1, 6, 500), ("cam_a", 3, 2, 1, 900)],
    "single tracklet (known defect #25)": [("cam_a", 1, 0, 6, 0)],
}

for name, people in SCENES.items():
    with tempfile.TemporaryDirectory() as td:
        plain = run(build_store(td, people, seed=1))
    with tempfile.TemporaryDirectory() as td:
        logged = run(build_store(td, people, seed=1), log=DecisionLog(run_id="R"))
    check(f"identical remap: {name}", plain == logged,
          f"{len(plain)} tracklet(s) -> {len(set(plain.values()))} identity(ies)")


# ---------------------------------------------------------------------------
print("\n2. CRITERION 4 -- gate_detail populated for EVERY gate on EVERY record")

with tempfile.TemporaryDirectory() as td:
    log = DecisionLog(run_id="R")
    run(build_store(td, SCENES["three people, two cameras"], seed=2), log=log)

missing = [(r.handle, g) for r in log.records for g in dlog.ALL_GATES
           if g not in r.gates]
check("every record carries all six gates", not missing,
      f"{len(log.records)} record(s), {len(dlog.ALL_GATES)} gates each")
check("records were produced at all", len(log.records) > 0, f"n={len(log.records)}")


# ---------------------------------------------------------------------------
print("\n3. CRITERION 9 -- TOP2_MARGIN summary matches the candidate vector")

drift = []
for r in log.records:
    try:
        verify_summary_matches_vector(r)
    except AssertionError as e:                              # noqa: PERF203
        drift.append(str(e))
check("no summary/vector drift across all records", not drift,
      drift[0] if drift else f"{len(log.records)} record(s) verified")


# ---------------------------------------------------------------------------
print("\n4. CRITERION 3 -- a subject failing TWO gates lists BOTH")

# A SUPPRESSED tracklet is the natural two-gate failure: it fails
# MIN_OBSERVATIONS, and it also fails ABSOLUTE_THRESHOLD because it is removed
# before any candidate is scored. Note MIN_OBSERVATIONS can ONLY fail here --
# suppression uses the same threshold, so every surviving tracklet passes it by
# construction. That is exactly why suppressed tracklets get a decision record.
with tempfile.TemporaryDirectory() as td:
    log3 = DecisionLog(run_id="R")
    run(build_store(td, [("cam_a", 1, 0, 2, 0), ("cam_a", 2, 1, 6, 500),
                         ("cam_b", 7, 2, 6, 0)], seed=3),
        log=log3, min_tracklet_observations=3)

multi = [r for r in log3.records if len(r.gates_failed) >= 2]
check("at least one record fails >=2 gates", multi,
      f"e.g. {multi[0].handle} failed {multi[0].gates_failed}" if multi
      else "none found")
if multi:
    r = multi[0]
    check("gates_failed length == count of failing gate_detail entries",
          len(r.gates_failed) == sum(1 for g in r.gates.values() if not g.passed),
          f"{r.gates_failed}")
    check("no short-circuit: every failing gate carries its own value+threshold",
          all(r.gates[g].threshold is not None for g in r.gates_failed),
          f"{[(g, r.gates[g].value, r.gates[g].threshold) for g in r.gates_failed]}")


# ---------------------------------------------------------------------------
print("\n5. CRITERION 6 -- Candidate<->Candidate merge is what resolves identity")

# Two tracklets of ONE person in DIFFERENT cameras, neither individually matched
# against any gallery (there is no gallery). They can only end up with a shared id
# by being compared to EACH OTHER and merged.
with tempfile.TemporaryDirectory() as td:
    log4 = DecisionLog(run_id="R")
    remap = run(build_store(td, [("cam_a", 1, 0, 6, 0), ("cam_b", 7, 0, 6, 0)],
                            jitter=0.05, seed=4), log=log4)
ids = set(remap.values())
check("two cross-camera tracklets of one person share ONE id",
      len(remap) == 2 and len(ids) == 1, f"remap={remap}")
accepted = [r for r in log4.records if r.accepted_partner]
check("the merge is recorded as an accepted candidate pair", accepted,
      f"{[(r.handle, r.accepted_partner) for r in accepted]}")

# And the negative control: two DIFFERENT people across cameras must not merge.
with tempfile.TemporaryDirectory() as td:
    remap2 = run(build_store(td, [("cam_a", 1, 0, 6, 0), ("cam_b", 7, 1, 6, 0)],
                             seed=5))
check("two different people across cameras stay separate",
      len(set(remap2.values())) == 2, f"remap={remap2}")


# ---------------------------------------------------------------------------
print("\n6. TOP2_MARGIN ships INERT (threshold None => enforces nothing)")

inert = all(r.gates[dlog.TOP2_MARGIN].threshold is None
            and r.gates[dlog.TOP2_MARGIN].passed
            for r in log.records)
check("every TOP2_MARGIN gate is logged-only and passing", inert,
      "threshold=None on all records")
computed = [r for r in log.records
            if r.gates[dlog.TOP2_MARGIN].extra.get("margin_all_scored") is not None]
check("margins are still COMPUTED while inert", computed,
      f"{len(computed)}/{len(log.records)} record(s) have a runner-up")

# Enforcing it must be expressible, and must actually bite.
with tempfile.TemporaryDirectory() as td:
    log5 = DecisionLog(run_id="R")
    run(build_store(td, SCENES["three people, two cameras"], seed=2),
        log=log5, top2_margin_threshold=0.99)
enforced = [r for r in log5.records if r.gates[dlog.TOP2_MARGIN].threshold == 0.99]
check("a threshold can be set and is recorded per gate", enforced,
      f"{len(enforced)} record(s) at threshold 0.99")


# ---------------------------------------------------------------------------
print("\n7. AGGREGATES are produced")

s = log.summary()
check("summary has all sections",
      all(k in s for k in ("gate_failures", "candidate_exclusions",
                           "margin_disagreement", "fragmentation", "cross_camera")))
check("fragmentation counts tracklets and identities",
      s["fragmentation"]["tracklets"] > 0 and s["fragmentation"]["identities"] > 0,
      f"{s['fragmentation']['tracklets']} -> {s['fragmentation']['identities']}")
check("per-camera observation stats present",
      len(s["fragmentation"]["observations_per_camera"]) >= 2,
      str(list(s["fragmentation"]["observations_per_camera"])))
check("margin disagreement rate has a band",
      "band" in s["margin_disagreement"],
      f"{s['margin_disagreement']['rate_pct']}% "
      f"({s['margin_disagreement']['band']})")

# Provisional handles must never be counted as identities.
handles = {o["handle"] for o in log.tracklet_outcomes.values()}
check("provisional handles are namespaced, never bare integers",
      all(isinstance(h, str) and h.startswith("U-") for h in handles), str(handles))

# Suppressed tracklets are visible rather than silent.
with tempfile.TemporaryDirectory() as td:
    log6 = DecisionLog(run_id="R")
    run(build_store(td, SCENES["a suppressed short tracklet"], seed=6), log=log6)
supp = [o for o in log6.tracklet_outcomes.values() if o["state"] == dlog.SUPPRESSED]
check("a suppressed tracklet is recorded with its state", len(supp) == 1,
      f"{[o['tracklet'] for o in supp]}")

# #25 IS NOW FIXED. This check previously asserted the DEFECT -- that a lone
# surviving tracklet was left expired_unresolved, so the whole video rendered as a
# bare "ID <track_id>" with no reid at all. There is nothing to merge with one
# tracklet, but there is still an identity to ASSIGN, and now it gets one.
with tempfile.TemporaryDirectory() as td:
    log7 = DecisionLog(run_id="R")
    remap7 = run(build_store(td, SCENES["single tracklet (known defect #25)"],
                             seed=7), log=log7)
resolved7 = [o for o in log7.tracklet_outcomes.values()
             if o["state"] == dlog.RESOLVED]
check("#25 fixed: a lone tracklet is ASSIGNED an identity, not left unresolved",
      len(remap7) == 1 and len(resolved7) == 1,
      f"remap={remap7}, states="
      f"{[o['state'] for o in log7.tracklet_outcomes.values()]}")
check("#25: and the assigned id is a real gid, not None",
      all(v is not None for v in remap7.values()), f"{remap7}")


# ---------------------------------------------------------------------------
print("\n8. JSONL round-trips")

import json
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "logs", "decisions.jsonl")
    log8 = DecisionLog(path=p, run_id="R")
    run(build_store(td, SCENES["three people, two cameras"], seed=8), log=log8)
    written = os.path.exists(p)
    lines = [json.loads(l) for l in open(p)] if written else []
check("log file written", written, p if written else "missing")
check("contains decision, outcome and summary rows",
      {"decision", "outcome", "summary"} <= {l.get("type") for l in lines},
      f"{len(lines)} line(s)")


# ---------------------------------------------------------------------------
failed = [n for n, ok in CHECKS if not ok]
print(f"\nPhase 1 decision log: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
if failed:
    print("FAILED: " + "; ".join(failed))
    raise SystemExit(1)
print("OK")
