"""
Characterisation of known defects -- reports PRESENT / FIXED, never asserts.

The polarity here is deliberate. These are defects the audit REPRODUCED, so a
test that asserts "bug is present" would start failing the moment we fix it,
which is backwards. Instead each check reports its current state, so running this
script after a phase lands shows exactly what moved.

    python tests/calibration/characterize_known_defects.py

Always exits 0. Needs no footage and no GPU. Cross-referenced to
REMEDIATION_PLAN.md issue numbers.
"""

import numpy as np

from _common import bootstrap, header, iou

bootstrap()

from detector import PersonDetector, Detection
from live.identity_engine import IdentityEngine
from identity.reconcile import reconcile_tracklets
# Synthetic vectors here go into a real PersonVectorStore, whose dimension guard
# rejects any other width. Track the store's constant instead of hardcoding, so a
# backbone swap cannot leave this script inserting the wrong shape.
from database.store import EMBEDDING_DIM as DIM

RESULTS = []


def report(issue, name, present, detail):
    state = "PRESENT" if present else "FIXED  "
    RESULTS.append((issue, name, present))
    print(f"  [{state}] {issue:<6} {name}")
    for line in detail.strip().splitlines():
        print(f"           {line}")


QUALITY = {"blur": 100, "brightness": 120, "area": 9000,
           "box_area_ratio": 0.02, "aspect": 0.4}


def two_people(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=DIM); a /= np.linalg.norm(a)
    b = rng.normal(size=DIM); b /= np.linalg.norm(b)
    return a, b


# ---------------------------------------------------------------------------
header("RECONCILE -- on the product path")

# ---- #25: fewer than 2 tracklets => no identity stamped at all
def check_single_tracklet():
    import os
    import tempfile
    from database.store import PersonVectorStore
    os.environ.pop("QDRANT_URL", None)
    rng = np.random.default_rng(1)
    out = {}
    for ntrack in (1, 2):
        with tempfile.TemporaryDirectory() as td:
            st = PersonVectorStore(path=os.path.join(td, "q"), url=None)
            for t in range(1, ntrack + 1):
                base = rng.normal(size=DIM)
                for f in range(5):
                    # 0.01 is a RELATIVE noise level, so scale by 1/sqrt(DIM):
                    # rng.normal(size=DIM) has norm ~sqrt(DIM), so unscaled it
                    # grows with the embedding width (same trap _synth.observe
                    # documents). Keeps these vectors' cosine width-independent.
                    v = base + (0.01 / np.sqrt(DIM)) * rng.normal(size=DIM)
                    v /= np.linalg.norm(v)
                    st.add_many([v.astype(np.float32)],
                                [{"camera": "cam_a", "track_id": t,
                                  "frame": f * 10, "run_id": "R"}])
            reconcile_tracklets(st, threshold=0.63, run_id="R",
                                same_camera_threshold=0.90,
                                require_reciprocal_best=True,
                                min_tracklet_observations=3,
                                log=lambda m: None)
            gids, offset = set(), None
            while True:
                pts, offset = st.client.scroll(st.collection, limit=1000,
                                               offset=offset, with_payload=True,
                                               with_vectors=False)
                for p in pts:
                    gids.add((p.payload or {}).get("reid_id"))
                if offset is None:
                    break
            out[ntrack] = gids
    broken = out[1] == {None}
    report("#25", "reconcile stamps NO identity below 2 tracklets", broken,
           f"1 tracklet, 5 obs each -> reid_ids {out[1]}\n"
           f"2 tracklets, 5 obs each -> reid_ids {out[2]}\n"
           "cause: `if len(keys) < 2: return {}` -- the whole video renders as "
           "bare 'ID <n>'")


# ---- #26: reconcile never reads ts
def check_ts_unused():
    import inspect
    import identity.reconcile as rec
    src = inspect.getsource(rec)
    uses_ts = '"ts"' in src or "'ts'" in src
    report("#26", "reconcile ignores `ts`, uses per-camera frame indices", not uses_ts,
           "payload carries a wall-clock `ts` (identity_stage._observation_payload)\n"
           "but reconcile spans use `frame`, which is NOT comparable across\n"
           "cameras running at different frame rates (15 vs 25 fps here)")


# ---- #38: no cross-camera simultaneity veto
def check_cross_camera_conflict():
    import inspect
    import identity.reconcile as rec
    src = inspect.getsource(rec.reconcile_tracklets)
    # conflict() skips pairs from different cameras
    skips = "if a[0] != b[0]:" in src and "continue" in src
    report("#38", "conflict() skips cross-camera pairs (no simultaneity veto)", skips,
           "two tracklets overlapping in TIME in non-co-visible cameras are\n"
           "provably different people, but only appearance and reciprocal-best\n"
           "stand between them and a merge")


for fn in (check_single_tracklet, check_ts_unused, check_cross_camera_conflict):
    try:
        fn()
    except Exception as e:                                   # noqa: BLE001
        print(f"  [SKIP   ] {fn.__name__}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
header("LIVE IdentityEngine -- OFF the product path (deferred, Part C)")
print("  With live.reconcile.enabled: true these ids are computed and DISCARDED.")
print("  Recorded so the defects are not rediscovered and mistaken for urgent.\n")

# ---- gallery poisoning via unguarded _reinforce
a, b = two_people(0)
eng = IdentityEngine(min_evidence_obs=3, same_camera_threshold=0.70,
                     cross_camera_threshold=0.60, bank_size=20)
for i in range(3):
    gid = eng.assign("cam1", 1, a, QUALITY, 100.0 + i, True)
for i in range(20):
    eng.assign("cam1", 1, b, QUALITY, 110.0 + i, True)     # ByteTrack id switch
score_b = eng.store.score(gid, b)
report("Part C", "unguarded _reinforce lets a wrong person overwrite the bank",
       score_b > 0.70,
       f"cosine(personA, personB) = {float(a @ b):.3f} (different people)\n"
       f"after 20 poisoned frames, score(gid {gid}, personB) = {score_b:.3f}\n"
       "no similarity check gates what enters the bank")

# ---- one bad exemplar is enough, because score() takes a max
eng2 = IdentityEngine(min_evidence_obs=3, same_camera_threshold=0.70, bank_size=20)
for i in range(3):
    g2 = eng2.assign("cam1", 7, a, QUALITY, 200.0 + i, True)
eng2.assign("cam1", 7, b, QUALITY, 210.0, True)            # exactly ONE bad frame
s1 = eng2.store.score(g2, b)
report("Part C", "a SINGLE bad exemplar matches at ~1.0 (score takes max)", s1 > 0.9,
       f"bank = 4x personA + 1x personB; score(gid, personB) = {s1:.3f}\n"
       "ActiveIdentitySet.score returns max(prototype, best_exemplar), so one\n"
       "poisoned exemplar carries the match with no consensus requirement")

# ---- two-lane leak: rejected same-camera, then accepted cross-camera
eng3 = IdentityEngine(min_evidence_obs=3, same_camera_threshold=0.70,
                      cross_camera_threshold=0.60, accept_margin=0.03,
                      coactive_window_sec=2.0)
for i in range(3):
    g = eng3.assign("cam1", 1, a, QUALITY, 100.0 + i, True)
for i in range(3):
    eng3.assign("cam2", 2, a, QUALITY, 120.0 + i, True)    # same person, 2nd camera
mix = 0.65 * a + np.sqrt(1 - 0.65 ** 2) * b
mix /= np.linalg.norm(mix)
for i in range(3):
    g3 = eng3.assign("cam1", 9, mix, QUALITY, 200.0 + i, True)
report("Part C", "two-lane leak: same_cam reject then cross_cam accept", g3 == g,
       f"stranger scores {float(a @ mix):.3f} -- BELOW same_camera_threshold 0.70\n"
       f"stranger got gid {g3}, real person has gid {g}\n"
       f"counters: recam_rej_below={eng3.recam_rej_below} linked={eng3.linked} "
       f"minted={eng3.minted}\n"
       "the SAME resolve increments both a rejection and a link, which is why the\n"
       "recam_rej_below counter that drove past threshold tuning is untrustworthy")

# ---- cross-camera co-presence is not vetoed
import inspect
src = inspect.getsource(IdentityEngine._gid_coactive)
report("Part C", "_gid_coactive skips other cameras (no cross-cam co-presence veto)",
       "key[0] != cam" in src,
       "one identity can be displayed on two different people simultaneously in\n"
       "two cameras -- relevant because cam_224 and cam_219 share a room")


# ---------------------------------------------------------------------------
header("BATCH PATH -- unused (Part C)")

# ---- pose ensemble duplicate boxes
d = object.__new__(PersonDetector)
d.min_containment = 0.6
d.pose_cfg = {}
P1, P2 = (0.0, 0.0, 100.0, 300.0), (90.0, 0.0, 190.0, 300.0)
d._pose_bodies = lambda frame: [P1, P2]
merged = Detection(x1=0, y1=0, x2=190, y2=300, confidence=.9, class_id=0, track_id=1)
clean = Detection(x1=90, y1=0, x2=190, y2=300, confidence=.8, class_id=0, track_id=2)
out = d._split_merged_boxes(None, [merged, clean])
dupes = sum(1 for i in range(len(out)) for j in range(i + 1, len(out))
            if iou((out[i].x1, out[i].y1, out[i].x2, out[i].y2),
                   (out[j].x1, out[j].y1, out[j].x2, out[j].y2)) >= 0.5)
report("Part C", "pose ensemble emits duplicate boxes for one person", dupes > 0,
       f"2 tracker boxes in -> {len(out)} boxes out, {dupes} overlapping pair(s)\n"
       "a merged box splits into both people while a clean box on the second\n"
       "person passes through unchanged\n"
       f"live path is protected (live.inference.pose_ensemble: false) but the\n"
       "_g() default is True, so deleting that config line re-enables it (#66)")

# ---- split primary id flips between people
firsts = []
for order in ((P1, P2), (P2, P1)):
    d._pose_bodies = lambda frame, o=order: list(o)
    res = d._split_merged_boxes(None, [Detection(x1=0, y1=0, x2=190, y2=300,
                                                confidence=.9, class_id=0,
                                                track_id=1)])
    firsts.append(next(r for r in res if r.track_id == 1).x1)
report("Part C", "pose-split 'primary' track_id flips between people",
       firsts[0] != firsts[1],
       f"pose order (P1,P2) -> track_id 1 at x1={firsts[0]}\n"
       f"pose order (P2,P1) -> track_id 1 at x1={firsts[1]}\n"
       "YOLO orders by confidence, which fluctuates, so the real track_id\n"
       "alternates between two people and its bank receives both")


# ---------------------------------------------------------------------------
header("SUMMARY")
present = [r for r in RESULTS if r[2]]
print(f"  {len(present)}/{len(RESULTS)} characterised defects still PRESENT")
for issue, name, is_present in RESULTS:
    print(f"    {'PRESENT' if is_present else 'FIXED  '}  {issue:<7} {name}")
print("\n  See REMEDIATION_PLAN.md for the phase that addresses each.")
