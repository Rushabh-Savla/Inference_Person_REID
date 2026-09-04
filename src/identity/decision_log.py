"""
decision_log.py  --  Phase 1: the reconciliation decision record.

============================ WHY THIS EXISTS ================================
Reconcile decides every id in the deliverable, and until now it emitted FOUR log
lines -- all of them successes. There was no way to ask why a merge did not
happen, which made every threshold question unanswerable and is why four rounds
of threshold tuning were reverted without ever learning anything.

This module owns the record. `reconcile.py` owns the decisions. Keeping them apart
matters because the record has to be complete enough to REPLAY the decisions
offline with no model in memory (REMEDIATION_PLAN.md Phase 3), and that is a
different concern from clustering.

------------------------------ TWO HARD RULES -------------------------------
1. NO SHORT-CIRCUITING. Every gate is evaluated on every candidate, even after
   one has already failed. Short-circuiting produces misleading histograms: a
   tracklet with 2 observations AND a 0.01 margin gets logged only as
   MIN_OBSERVATIONS, so the histogram says "raise the frame requirement" when a
   meaningful fraction of those cases are genuinely ambiguous pairs that would
   still fail afterwards.

2. SAME SCHEMA FOR ACCEPTED AND REJECTED. Instrumenting only failures leaves no
   baseline -- there is no way to tell whether a winning score of 0.84 is normal
   or suspiciously low without the distribution of scores that passed.

--------------------------- THE TWO MARGIN VARIANTS -------------------------
`TOP2_MARGIN` asks "is there a UNIQUELY plausible match", which is a different
question from the absolute threshold's "is there a plausible match". It is
computed two ways because the right answer is an empirical question:

  margin_eligible   -- runner-up over candidates that survived the
                       ranking-independent hard constraints. Selection safety: a
                       candidate that can never be picked should not suppress a
                       good merge.
  margin_all_scored -- runner-up over everything above the floor. Evidence
                       quality: a near-tie means the embedding cannot separate
                       the two, whether or not the runner-up is available.

Both are logged; exactly ONE can ever gate, chosen by `basis`. `runner_up_differs`
aggregated over a run is itself a metric -- it measures how much the hard
constraints reshape the candidate space.

RECIPROCAL_BEST is deliberately NOT a pre-margin constraint. `best_partner` is
single-valued, so filtering by reciprocity leaves at most one candidate, no
runner-up exists, and the margin gate becomes dead code. It is recorded per
candidate as `would_fail_reciprocity` -- an annotation that excludes nothing --
so the diagnostic survives: if near-ties are common and the runner-up almost
always carries that flag, reciprocal-best is already resolving them.
============================================================================
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ---------------------------------------------------------------------- gates

MIN_OBSERVATIONS = "MIN_OBSERVATIONS"
ABSOLUTE_THRESHOLD = "ABSOLUTE_THRESHOLD"
TOP2_MARGIN = "TOP2_MARGIN"
RECIPROCAL_BEST = "RECIPROCAL_BEST"
TEMPORAL_CONFLICT_SAME_CAMERA = "TEMPORAL_CONFLICT_SAME_CAMERA"
TEMPORAL_CONFLICT_CROSS_CAMERA = "TEMPORAL_CONFLICT_CROSS_CAMERA"
# The merge would require a person to move faster than anyone on this floor has
# been observed to move -- computed from the floor positions the LIVE RUN recorded
# (src/geometry/), never re-derived here. See geometry/__init__.py invariant 1.
GEOMETRIC_UNREACHABLE = "GEOMETRIC_UNREACHABLE"

ALL_GATES = (
    MIN_OBSERVATIONS,
    ABSOLUTE_THRESHOLD,
    TOP2_MARGIN,
    RECIPROCAL_BEST,
    TEMPORAL_CONFLICT_SAME_CAMERA,
    TEMPORAL_CONFLICT_CROSS_CAMERA,
    GEOMETRIC_UNREACHABLE,
)

# Reasons a candidate is removed from the ELIGIBLE set. All are independent of the
# subject's own score ranking -- that independence is what makes it sound to
# compute the margin over what remains.
NOT_MERGEABLE_CROSS = "NOT_MERGEABLE_CROSS"
BELOW_ABSOLUTE_THRESHOLD = "BELOW_ABSOLUTE_THRESHOLD"

EXCLUSION_REASONS = (
    TEMPORAL_CONFLICT_SAME_CAMERA,
    TEMPORAL_CONFLICT_CROSS_CAMERA,
    GEOMETRIC_UNREACHABLE,
    NOT_MERGEABLE_CROSS,
    BELOW_ABSOLUTE_THRESHOLD,
)

# Terminal states, mirroring the spec's state machine.
RESOLVED = "resolved"
EXPIRED_UNRESOLVED = "expired_unresolved"
SUPPRESSED = "suppressed"          # below min_tracklet_observations


# ---------------------------------------------------------------- data classes

@dataclass
class Candidate:
    """One rival cluster, with everything needed to reinterpret the decision later."""
    handle: str
    score: float
    excluded_by: Optional[str] = None
    would_fail_reciprocity: Optional[bool] = None
    cameras: list = field(default_factory=list)
    cluster_size: int = 0
    observations: int = 0
    # Cosine between this candidate's prototype and the BEST candidate's. This is
    # what separates "one person fragmented into two clusters" (high, ~0.95) from
    # "genuinely ambiguous embeddings" (low, ~0.55) when a margin gate fails.
    pair_similarity_to_best: Optional[float] = None

    @property
    def eligible(self) -> bool:
        return self.excluded_by is None


@dataclass
class GateResult:
    value: Any
    threshold: Any
    passed: bool
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"value": self.value, "threshold": self.threshold, "passed": self.passed}
        d.update(self.extra)
        return d


@dataclass
class DecisionRecord:
    """One merge decision. `handle` is the subject; `candidates` are its rivals."""
    handle: str
    state: str
    phase: str                              # "same_camera" | "cross_camera"
    round_index: int = 0
    assigned_id: Optional[int] = None
    merged_from: list = field(default_factory=list)
    accepted_partner: Optional[str] = None

    observations: int = 0
    cameras: list = field(default_factory=list)
    frame_range: Optional[list] = None
    time_range: Optional[list] = None
    context: str = "same_camera"            # which threshold set applied

    gates: dict = field(default_factory=dict)      # name -> GateResult
    candidates: list = field(default_factory=list)  # list[Candidate]
    candidates_truncated: bool = False
    scored_count: int = 0
    eligible_count: int = 0

    # -------------------------------------------------------------- rendering
    @property
    def gates_failed(self) -> list:
        return [n for n, g in self.gates.items() if not g.passed]

    def to_dict(self) -> dict:
        return {
            "handle": self.handle,
            "state": self.state,
            "phase": self.phase,
            "round": self.round_index,
            "assigned_id": self.assigned_id,
            "merged_from": list(self.merged_from),
            "accepted_partner": self.accepted_partner,
            "observations": self.observations,
            "cameras": list(self.cameras),
            "frame_range": self.frame_range,
            "time_range": self.time_range,
            "context": self.context,
            "gates_failed": self.gates_failed,
            "gate_detail": {n: g.to_dict() for n, g in self.gates.items()},
            "candidates": [asdict(c) for c in self.candidates],
            "candidates_truncated": self.candidates_truncated,
            "scored_count": self.scored_count,
            "eligible_count": self.eligible_count,
        }


# ------------------------------------------------------------ margin machinery

def compute_top2_margin(candidates, floor, basis="eligible", threshold=None):
    """Build the TOP2_MARGIN GateResult from an annotated candidate list.

    `candidates` must already carry `excluded_by`. Returns (GateResult, best,
    runner_eligible, runner_all_scored) so the caller can reuse the picks.

    `threshold=None` means LOGGED ONLY -- the gate reports `passed=True` and
    enforces nothing. That is the Phase 1 default and it is why adding this gate
    cannot change any decision.
    """
    eligible = sorted((c for c in candidates if c.eligible),
                      key=lambda c: c.score, reverse=True)
    # "above the floor" ignores the hard constraints but still excludes candidates
    # that were only removed FOR being below the floor.
    all_scored = sorted((c for c in candidates
                         if c.excluded_by != BELOW_ABSOLUTE_THRESHOLD),
                        key=lambda c: c.score, reverse=True)

    best = eligible[0] if eligible else None
    runner_elig = eligible[1] if len(eligible) > 1 else None
    # The all-scored runner-up is the best OTHER candidate, so skip the winner.
    runner_all = next((c for c in all_scored
                       if best is None or c.handle != best.handle), None)

    def _margin(runner):
        if best is None or runner is None:
            return None
        return round(best.score - runner.score, 6)

    m_elig, m_all = _margin(runner_elig), _margin(runner_all)
    chosen = m_elig if basis == "eligible" else m_all

    # None means "no runner-up exists", i.e. nothing to be ambiguous against.
    passed = True if (threshold is None or chosen is None) else chosen >= threshold

    extra = {
        "basis": basis,
        "best": None if best is None else {"handle": best.handle,
                                           "score": round(best.score, 6)},
        "runner_eligible": None if runner_elig is None else {
            "handle": runner_elig.handle, "score": round(runner_elig.score, 6)},
        "runner_all_scored": None if runner_all is None else {
            "handle": runner_all.handle, "score": round(runner_all.score, 6),
            "excluded_by": runner_all.excluded_by,
            "would_fail_reciprocity": runner_all.would_fail_reciprocity},
        "margin_eligible": m_elig,
        "margin_all_scored": m_all,
        "margin_ratio": (None if (chosen is None or not best or best.score == 0)
                         else round(chosen / best.score, 6)),
        # None when there is no selectable candidate at all: with no `best` there is
        # no margin decision, so counting it as a "disagreement" would inflate the
        # rate with non-decisions. A single-camera run reported 100% before this
        # guard, purely because every eligible set was empty.
        "runner_up_differs": (
            None if best is None else
            (runner_elig.handle if runner_elig else None)
            != (runner_all.handle if runner_all else None)),
        "enforcing": threshold is not None,
    }
    return (GateResult(value=chosen, threshold=threshold, passed=passed, extra=extra),
            best, runner_elig, runner_all)


def verify_summary_matches_vector(record: DecisionRecord) -> None:
    """Acceptance criterion 9: the precomputed TOP2_MARGIN summary must equal what
    recomputing from `candidates` gives. Raises AssertionError on drift.

    Exists because the summary is precomputed for convenience, and computing the
    same thing twice in two places is how a logger and its analyser silently
    disagree months later.
    """
    gate = record.gates.get(TOP2_MARGIN)
    if gate is None:
        return
    floor = record.gates.get(ABSOLUTE_THRESHOLD)
    recomputed, _, _, _ = compute_top2_margin(
        record.candidates,
        floor.threshold if floor else None,
        basis=gate.extra.get("basis", "eligible"),
        threshold=gate.threshold,
    )
    for key in ("margin_eligible", "margin_all_scored", "runner_up_differs"):
        a, b = gate.extra.get(key), recomputed.extra.get(key)
        assert a == b, (f"TOP2_MARGIN summary drifted from the candidate vector for "
                        f"{record.handle!r}: {key} logged={a!r} recomputed={b!r}")
    assert gate.value == recomputed.value, (
        f"TOP2_MARGIN value drifted for {record.handle!r}: "
        f"{gate.value!r} vs {recomputed.value!r}")


# ---------------------------------------------------------------------- the log

class DecisionLog:
    """Collects records, writes JSONL, and aggregates the run-level metrics."""

    def __init__(self, path: Optional[str] = None, run_id: Optional[str] = None,
                 max_candidates: int = 200, verify: bool = True):
        self.path = path
        self.run_id = run_id
        self.max_candidates = int(max_candidates)
        self.verify = bool(verify)
        self.records: list[DecisionRecord] = []
        self.tracklet_outcomes: dict = {}     # key -> dict

    # -------------------------------------------------------------- collection
    def add(self, record: DecisionRecord) -> DecisionRecord:
        if len(record.candidates) > self.max_candidates:
            record.candidates = sorted(record.candidates, key=lambda c: c.score,
                                       reverse=True)[:self.max_candidates]
            record.candidates_truncated = True
        if self.verify:
            verify_summary_matches_vector(record)
        self.records.append(record)
        return record

    def set_outcome(self, key, *, state, assigned_id=None, handle=None,
                    merged_from=None, observations=0, cameras=(),
                    frame_range=None, time_range=None):
        self.tracklet_outcomes[key] = {
            "tracklet": list(key) if isinstance(key, tuple) else key,
            "handle": handle,
            "state": state,
            "assigned_id": assigned_id,
            "merged_from": list(merged_from or []),
            "observations": observations,
            "cameras": list(cameras),
            "frame_range": frame_range,
            "time_range": time_range,
        }

    # -------------------------------------------------------------- aggregates
    def gate_failure_counts(self) -> dict:
        counts = {g: 0 for g in ALL_GATES}
        for r in self.records:
            for g in r.gates_failed:
                counts[g] = counts.get(g, 0) + 1
        return counts

    def exclusion_counts(self) -> dict:
        counts = {e: 0 for e in EXCLUSION_REASONS}
        for r in self.records:
            for c in r.candidates:
                if c.excluded_by:
                    counts[c.excluded_by] = counts.get(c.excluded_by, 0) + 1
        return counts

    def margin_disagreement(self) -> dict:
        """Rate at which the two margin definitions pick a different runner-up.

        Read against the agreed bands: <1% the distinction is not worth
        maintaining; 10-20% it is a meaningful design choice; ~50% the hard
        constraints are fundamentally reshaping the candidate space.
        """
        considered = differs = no_decision = 0
        deltas = []
        for r in self.records:
            g = r.gates.get(TOP2_MARGIN)
            if g is None:
                continue
            flag = g.extra.get("runner_up_differs")
            if flag is None:
                # No selectable candidate -> not a margin decision. Counted
                # separately so a run dominated by these is obvious rather than
                # masquerading as a 100% disagreement rate.
                no_decision += 1
                continue
            considered += 1
            if flag:
                differs += 1
            me, ma = g.extra.get("margin_eligible"), g.extra.get("margin_all_scored")
            if me is not None and ma is not None:
                deltas.append(abs(me - ma))
        rate = (100.0 * differs / considered) if considered else 0.0
        band = ("no margin decisions in this run" if considered == 0 else
                "not worth maintaining" if rate < 1 else
                "meaningful design choice" if rate < 30 else
                "hard constraints reshaping the candidate space")
        return {"considered": considered, "differs": differs,
                "no_selectable_candidate": no_decision,
                "rate_pct": round(rate, 2), "band": band,
                "mean_abs_delta": (round(sum(deltas) / len(deltas), 4)
                                   if deltas else None)}

    def accepted_rejected_margins(self) -> dict:
        """The two distributions needed to choose a margin threshold from data."""
        acc, rej = [], []
        for r in self.records:
            g = r.gates.get(TOP2_MARGIN)
            if g is None or g.value is None:
                continue
            (acc if r.accepted_partner else rej).append(g.value)
        return {"accepted": sorted(acc), "rejected": sorted(rej)}

    def fragmentation(self) -> dict:
        """Tracklets per final identity, and observations per tracklet per camera."""
        per_identity = {}
        per_camera_obs = {}
        suppressed = unresolved = 0
        for key, o in self.tracklet_outcomes.items():
            cam = (key[0] if isinstance(key, tuple) else None)
            per_camera_obs.setdefault(cam, []).append(o["observations"])
            if o["state"] == SUPPRESSED:
                suppressed += 1
            elif o["state"] == EXPIRED_UNRESOLVED:
                unresolved += 1
            if o["assigned_id"] is not None:
                per_identity.setdefault(o["assigned_id"], []).append(key)
        sizes = sorted((len(v) for v in per_identity.values()), reverse=True)
        total_obs = sum(o["observations"] for o in self.tracklet_outcomes.values())
        supp_obs = sum(o["observations"] for o in self.tracklet_outcomes.values()
                       if o["state"] == SUPPRESSED)
        return {
            "tracklets": len(self.tracklet_outcomes),
            "identities": len(per_identity),
            "tracklets_per_identity": sizes,
            "max_tracklets_per_identity": (sizes[0] if sizes else 0),
            "mean_tracklets_per_identity": (round(sum(sizes) / len(sizes), 2)
                                            if sizes else 0.0),
            "suppressed": suppressed,
            "suppressed_observation_share_pct": (round(100.0 * supp_obs / total_obs, 2)
                                                 if total_obs else 0.0),
            "expired_unresolved": unresolved,
            "observations_per_camera": {
                c: {"tracklets": len(v), "total": sum(v),
                    "min": min(v), "max": max(v),
                    "mean": round(sum(v) / len(v), 2)}
                for c, v in sorted(per_camera_obs.items(), key=lambda kv: str(kv[0]))
            },
        }

    def cross_camera(self) -> dict:
        """Per camera-pair: candidates considered, merged, and each rejection reason."""
        pairs = {}
        for r in self.records:
            if r.phase != "cross_camera":
                continue
            subj_cams = set(r.cameras)
            for c in r.candidates:
                for a in subj_cams:
                    for b in set(c.cameras):
                        if a == b:
                            continue
                        pk = " <-> ".join(sorted((str(a), str(b))))
                        e = pairs.setdefault(pk, {"considered": 0, "merged": 0,
                                                  "excluded": {}})
                        e["considered"] += 1
                        if r.accepted_partner == c.handle:
                            e["merged"] += 1
                        if c.excluded_by:
                            e["excluded"][c.excluded_by] = \
                                e["excluded"].get(c.excluded_by, 0) + 1
        spanning = sum(1 for o in self.tracklet_outcomes.values()
                       if len(set(o["cameras"])) > 1)
        by_identity = {}
        for key, o in self.tracklet_outcomes.items():
            if o["assigned_id"] is None:
                continue
            by_identity.setdefault(o["assigned_id"], set()).update(o["cameras"])
        return {
            "identities_spanning_multiple_cameras":
                sum(1 for cams in by_identity.values() if len(cams) > 1),
            "tracklets_spanning_multiple_cameras": spanning,
            "camera_pairs": pairs,
        }

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "decisions": len(self.records),
            "gate_failures": self.gate_failure_counts(),
            "candidate_exclusions": self.exclusion_counts(),
            "margin_disagreement": self.margin_disagreement(),
            "fragmentation": self.fragmentation(),
            "cross_camera": self.cross_camera(),
        }

    # -------------------------------------------------------------------- output
    def write(self) -> Optional[str]:
        """One JSON object per line: decisions, then outcomes, then the summary.
        Degrade rather than crash -- losing the log must never lose the run."""
        if not self.path:
            return None
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "w") as f:
                for r in self.records:
                    f.write(json.dumps({"type": "decision", "run_id": self.run_id,
                                        **r.to_dict()}) + "\n")
                for o in self.tracklet_outcomes.values():
                    f.write(json.dumps({"type": "outcome", "run_id": self.run_id,
                                        **o}) + "\n")
                f.write(json.dumps({"type": "summary", **self.summary()}) + "\n")
            return self.path
        except OSError as e:
            print(f"[decision_log] could not write {self.path}: {e}")
            return None

    def print_summary(self, log=print):
        s = self.summary()
        frag, xcam, dis = s["fragmentation"], s["cross_camera"], s["margin_disagreement"]
        log("  --- reconcile diagnostics ---")
        log(f"    decisions evaluated: {s['decisions']}")
        gf = {k: v for k, v in s["gate_failures"].items() if v}
        log(f"    gate failures: {gf or 'none'}")
        ce = {k: v for k, v in s["candidate_exclusions"].items() if v}
        log(f"    candidate exclusions: {ce or 'none'}")
        log(f"    tracklets={frag['tracklets']} -> identities={frag['identities']}"
            f"  max/identity={frag['max_tracklets_per_identity']}"
            f"  mean={frag['mean_tracklets_per_identity']}")
        log(f"    suppressed={frag['suppressed']} "
            f"({frag['suppressed_observation_share_pct']}% of observations)  "
            f"expired_unresolved={frag['expired_unresolved']}")
        for cam, st in frag["observations_per_camera"].items():
            log(f"      {cam}: {st['tracklets']} tracklet(s), "
                f"obs total={st['total']} min={st['min']} max={st['max']} "
                f"mean={st['mean']}")
        log(f"    identities spanning >1 camera: "
            f"{xcam['identities_spanning_multiple_cameras']}")
        for pk, e in sorted(xcam["camera_pairs"].items()):
            log(f"      {pk}: considered={e['considered']} merged={e['merged']} "
                f"excluded={e['excluded'] or '{}'}")
        log(f"    TOP2_MARGIN disagreement: {dis['differs']}/{dis['considered']} "
            f"= {dis['rate_pct']}% ({dis['band']})"
            + (f"  [+{dis['no_selectable_candidate']} with no selectable candidate]"
               if dis["no_selectable_candidate"] else ""))
