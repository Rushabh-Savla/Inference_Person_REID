"""
Plan item #45a -- reconcile must not compare prototype MEANS only.

THE DEFECT, in one sentence: a person seen front-on and then from behind has two
clusters of observations, and their mean sits between them matching NEITHER view,
so that person's own two fragments can score LOWER than two different people
whose means both sit in the same "average person" region.

That is not a threshold problem. J.6 measured the consequence directly: the
per-subject same/different boundaries OVERLAP (p95 of the top "different" score
0.816 is above p5 of the worst "same" score 0.719), so for a real fraction of
people NO bar merges their fragments while rejecting strangers. The operator saw
both halves of it in one run -- several people fused into one reid at a low bar,
one person cycling through many reids at a high bar.

The live engine already avoids this and documents why (ActiveIdentitySet.score:
"medoid-style ... more robust than a mean alone when a person flips front/back or
pose-shifts"). Reconcile never got that fix.

WHAT IS PINNED HERE:
  1. the front/back fixture ITSELF -- proof the defect is real: two fragments of
     one bimodal person score BELOW two look-alike strangers under prototype
     scoring, so no threshold can separate them
  2. max_exemplar and consensus both invert that ordering
  3. scoring="prototype" (the default) reproduces the old behaviour EXACTLY, so
     #45a cannot change any run that does not ask for it
  4. determinism -- the observation cap subsamples evenly, never randomly, so a
     replay reproduces the same score
  5. an unknown mode falls back loudly instead of crashing a run's finalization
"""

import sys

import numpy as np
from types import SimpleNamespace

from _synth import Check, DIM
from identity.reconcile import (CONSENSUS, MAX_EXEMPLAR, PROTOTYPE,
                                reconcile_tracklets, score_observation_sets,
                                _prototype, _subsample_rows, _unit_rows)

THRESHOLD = 0.63


def _unit(*coeffs):
    v = np.zeros(DIM, np.float32)
    for i, c in enumerate(coeffs):
        v[i] = c
    return v / np.linalg.norm(v)


# Two appearance MODES of one person: front and back. Nearly orthogonal, because
# a front view and a back view of the same body share little pixel evidence --
# that is the whole difficulty.
FRONT = _unit(1.0)
BACK = _unit(0.25, 1.0)


def _bimodal(n_front, n_back):
    """Observations of one person: some front views, some back views."""
    return [FRONT] * n_front + [BACK] * n_back


class _FakeStore:
    collection = "persons"

    def __init__(self, observations):
        self.assigned = {}
        self.cleared = []
        self.client = SimpleNamespace(
            scroll=lambda col, limit, offset, with_payload, with_vectors: (
                observations, None))

    def set_global_id(self, point_ids, global_id):
        for p in point_ids:
            self.assigned[p] = global_id

    def clear_global_id(self, point_ids):
        self.cleared.extend(point_ids)


def _observations(specs):
    """specs: [(camera, track_id, [vectors], first_frame, gid)] -> point list."""
    out = []
    for cam, tid, vecs, f0, gid in specs:
        for i, v in enumerate(vecs):
            out.append(SimpleNamespace(
                id=f"{cam}-{tid}-{f0 + i}",
                vector=np.asarray(v, dtype=np.float32).tolist(),
                payload={"camera": cam, "track_id": tid, "frame": f0 + i,
                         "run_id": "r1", "reid_id": gid},
            ))
    return out


def _run(specs, **kw):
    store = _FakeStore(_observations(specs))
    kw.setdefault("same_camera_threshold", 0.90)
    kw.setdefault("min_tracklet_observations", 1)
    remap = reconcile_tracklets(store, threshold=THRESHOLD, run_id="r1",
                               require_reciprocal_best=True,
                               log=lambda *_a, **_k: None, **kw)
    return len(set(remap.values())), remap


def _score(vecs_a, vecs_b, mode, **kw):
    return score_observation_sets(_unit_rows(vecs_a), _unit_rows(vecs_b),
                                  _prototype(vecs_a), _prototype(vecs_b),
                                  mode=mode, **kw)


def main():
    c = Check("#45a: reconcile scoring modes (prototype / max_exemplar / consensus)")

    # ---------------------------------------------------------------- 1
    print("\n1. THE DEFECT IS REAL -- prototype means invert the ordering")

    # One person, two time-disjoint visits. Visit A was mostly front views, visit
    # B mostly back views: exactly "he walked out and came back the other side of
    # the room facing away".
    visit_a = _bimodal(8, 2)
    visit_b = _bimodal(2, 8)

    # Two DIFFERENT people who dress alike. Their means sit at cosine 0.80 -- close
    # enough to be confusable, which is the regime the field data lives in (the
    # measured different-person prototype ceiling is 0.661, and cross-camera
    # strangers reached 0.86). Deliberately NOT near-identical: two people at 0.999
    # are the same person as far as any appearance model can tell, and a fixture
    # like that only proves the fixture is broken. An earlier version of this test
    # did exactly that and "failed" three checks for no reason.
    stranger_1 = [_unit(1.0, 0.5, 0.0)] * 10
    stranger_2 = [_unit(1.0, 0.0, 0.5)] * 10

    same_proto = _score(visit_a, visit_b, PROTOTYPE)
    diff_proto = _score(stranger_1, stranger_2, PROTOTYPE)
    c.ok(same_proto < diff_proto,
         f"prototype: ONE person's two visits score {same_proto:.3f}, BELOW two "
         f"strangers at {diff_proto:.3f} -- no threshold can separate these")

    for mode in (MAX_EXEMPLAR, CONSENSUS):
        same = _score(visit_a, visit_b, mode)
        diff = _score(stranger_1, stranger_2, mode)
        c.ok(same > diff,
             f"{mode}: same person {same:.3f} now ABOVE strangers {diff:.3f} "
             f"(ordering fixed)")

    # max_exemplar can never score below prototype -- it takes the max of the two.
    c.ok(_score(visit_a, visit_b, MAX_EXEMPLAR) >= same_proto,
         "max_exemplar >= prototype by construction (it is a max)")

    # ---------------------------------------------------------------- 2
    print("\n2. THE SYMPTOM -- one person's re-entry merges, strangers do not")

    # Same camera, time-disjoint: the merge reconcile is supposed to make.
    reentry = [("cam_213", 1, visit_a, 0, 1), ("cam_213", 2, visit_b, 500, 2)]
    n, _ = _run(reentry, scoring=PROTOTYPE)
    c.eq(n, 2, "prototype: the re-entering person stays SPLIT (the operator's bug)")
    for mode in (MAX_EXEMPLAR, CONSENSUS):
        n, _ = _run(reentry, scoring=mode)
        c.eq(n, 1, f"{mode}: the same re-entry MERGES into one identity")

    # And the negative control at the SAME bar: the look-alike strangers must not
    # merge just because scoring got more permissive. Time-disjoint, so only
    # appearance decides -- the temporal veto cannot help here.
    strangers = [("cam_213", 1, stranger_1, 0, 1),
                 ("cam_213", 2, stranger_2, 500, 2)]
    for mode in (MAX_EXEMPLAR, CONSENSUS):
        n, _ = _run(strangers, scoring=mode)
        c.eq(n, 2, f"{mode}: look-alike strangers stay separate at the same bar")

    # ---------------------------------------------------------------- 2b
    print("\n2b. THE PRICE OF max_exemplar -- one bad crop carries a false merge")

    # A single contaminated observation: an occluded/mis-cropped frame of person 1
    # that happens to look exactly like person 2. This is not hypothetical -- it is
    # a defect already recorded against the live bank ("a SINGLE bad exemplar
    # matches at ~1.0 because score() takes max(prototype, best_exemplar)").
    contaminated = stranger_1[:9] + [stranger_2[0]]
    polluted = [("cam_213", 1, contaminated, 0, 1),
                ("cam_213", 2, stranger_2, 500, 2)]
    n, _ = _run(polluted, scoring=MAX_EXEMPLAR)
    c.eq(n, 1, "max_exemplar: ONE bad crop out of ten fuses two people "
               "(its documented weakness, reproduced)")
    n, _ = _run(polluted, scoring=CONSENSUS)
    c.eq(n, 2, "consensus: the same bad crop does NOT -- it needs many matching "
               "view pairs, not one")
    c.ok(_score(contaminated, stranger_2, MAX_EXEMPLAR)
         > _score(contaminated, stranger_2, CONSENSUS),
         f"and the scores show why: max_exemplar "
         f"{_score(contaminated, stranger_2, MAX_EXEMPLAR):.3f} vs consensus "
         f"{_score(contaminated, stranger_2, CONSENSUS):.3f}")

    # ---------------------------------------------------------------- 3
    print("\n3. THE DEFAULT CHANGES NOTHING")

    scenes = {
        "one person re-entering": reentry,
        "look-alike strangers": strangers,
        "cross-camera pair": [("cam_213", 1, visit_a, 0, 1),
                              ("cam_224", 7, visit_a, 0, 7)],
        "three tracklets, two cameras": [
            ("cam_213", 1, visit_a, 0, 1), ("cam_213", 2, visit_b, 500, 2),
            ("cam_224", 7, stranger_1, 0, 7)],
    }
    for name, specs in scenes.items():
        explicit = _run(specs, scoring=PROTOTYPE)
        default = _run(specs)                      # scoring not passed at all
        c.ok(explicit == default,
             f"default == prototype on '{name}' ({explicit[0]} identity(ies))")

    # ---------------------------------------------------------------- 4
    print("\n4. DETERMINISM -- the observation cap subsamples EVENLY")

    long_run = [_unit(1.0, i / 100.0) for i in range(300)]
    a = _score(long_run, visit_b, CONSENSUS, cap=16)
    b = _score(long_run, visit_b, CONSENSUS, cap=16)
    c.ok(a == b, f"same inputs give the same score twice ({a:.6f})")

    rows = _unit_rows(long_run)
    capped = _subsample_rows(rows, 16)
    c.eq(capped.shape[0], 16, "the cap is respected")
    c.ok(np.allclose(capped[0], rows[0]) and np.allclose(capped[-1], rows[-1]),
         "the sample spans the whole tracklet (keeps first and last view)")
    c.ok(np.allclose(_subsample_rows(rows, 0), rows),
         "cap=0 means no cap")

    # A cap must not silently change the answer for tracklets under it.
    short = _bimodal(4, 4)
    c.ok(_score(short, visit_b, CONSENSUS, cap=64)
         == _score(short, visit_b, CONSENSUS, cap=0),
         "a tracklet smaller than the cap is unaffected by it")

    # ---------------------------------------------------------------- 5
    print("\n5. A BAD MODE FALLS BACK, IT DOES NOT CRASH THE RUN")

    lines = []
    store = _FakeStore(_observations(reentry))
    remap = reconcile_tracklets(store, threshold=THRESHOLD, run_id="r1",
                               same_camera_threshold=0.90,
                               min_tracklet_observations=1,
                               scoring="medoid-ish", log=lines.append)
    c.ok(remap == _run(reentry, scoring=PROTOTYPE)[1],
         "an unknown mode falls back to prototype (finalization survives)")
    c.ok(any("unknown scoring" in m for m in lines),
         "and says so, rather than failing silently")

    c.done()


if __name__ == "__main__":
    main()
