"""
Calibration review: what this setting is SURE about, and what it is guessing.

    python tests/calibration/review_links.py <run_id> [reconcile flags]
    python tests/calibration/review_links.py <run_id> --cross 0.80 --covis-tolerance 2.3
    python tests/calibration/review_links.py <run_id> --score          # grade vs stored labels
    python tests/calibration/review_links.py <run_id> --label          # record verdicts

WHY THIS EXISTS -- it is a correction to how Part M was run.

Every threshold in Part M was picked from operator labels on merges that HAPPENED.
Nobody was ever asked about the links reconcile REFUSED. That sample is biased by
construction: it contains only links that survived the bar in force, so raising the
bar kills true links that were never in the sample, and the next video review
discovers one. That happened three rounds running (reid 18/24, then reid 23/16),
and the operator's verdict on the method was correct: *"it's like you're hardcoding
based on the shit I'm telling you, but in a live run everything will be different."*

They were right. A cosine bar chosen from a 0.058-wide gap between seven labelled
pairs on one 80-second run is fitted to noise. So this tool changes what the
calibration IS:

  * labels live in calibration/link_labels.jsonl, not in a conversation. They
    ACCUMULATE across runs, with provenance, and separate what the operator STATED
    from what was inferred on their behalf.
  * `--score` grades any candidate setting against every label ever recorded. That
    is the number that generalises, because it grows with each run instead of being
    replaced by the last one.
  * the default view separates CONFIDENT decisions from UNCERTAIN ones, so what the
    system is guessing about is visible before it ships rather than after.

A link is UNCERTAIN when the evidence does not clearly support either answer:
  * refused with a score within `--band` of the bar it needed (a hair either way
    flips it), or
  * blocked by a simultaneity veto whose overlap is inside the clock's own jitter
    (`--jitter`, default 2.5s) -- a timing rule firing on timing noise.

READ-ONLY on the store. `--label` writes only to the labels file.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

from _common import arg, bootstrap, flag, header, project_root, reconcile_settings

bootstrap()

from database.store import PersonVectorStore                        # noqa: E402
from identity import reconcile as R                                 # noqa: E402
from identity.reconcile import describe_reconcile_kwargs            # noqa: E402

# Measured different-person prototype ceiling (Part H.4), same as the sweep uses.
STRANGER_CEILING = 0.661
LABELS_PATH = os.path.join(project_root(), "calibration", "link_labels.jsonl")


class _ReadOnlyStore:
    def __init__(self, store):
        self.client = store.client
        self.collection = store.collection

    def set_global_id(self, point_ids, global_id):
        pass

    def clear_global_id(self, point_ids):
        pass


def key_str(k):
    return f"{k[0]}:{k[1]}"


def parse_key(s):
    cam, _, tid = s.partition(":")
    return (cam, int(tid))


def load_labels(run_id=None):
    """-> {frozenset({key_a, key_b}): record}. Later entries win."""
    out = {}
    if not os.path.exists(LABELS_PATH):
        return out
    with open(LABELS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_comment" in rec or "verdict" not in rec:
                continue
            if run_id is not None and rec.get("run_id") != run_id:
                continue
            out[frozenset((rec["a"], rec["b"]))] = rec
    return out


def append_label(run_id, a, b, verdict, source="operator"):
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    with open(LABELS_PATH, "a") as f:
        f.write(json.dumps({"run_id": run_id, "a": key_str(a), "b": key_str(b),
                            "verdict": verdict, "source": source,
                            "stated": True}) + "\n")


def fmt_members(members):
    by_cam = defaultdict(list)
    for cam, tid in sorted(members):
        by_cam[cam].append(tid)
    return " ".join(f"{c}({','.join(str(t) for t in ts)})"
                    for c, ts in sorted(by_cam.items()))


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        raise SystemExit(__doc__.strip().split("\n\n")[1])
    run_id = sys.argv[1]

    kw = reconcile_settings(extra_flags=("--url", "--path", "--band", "--jitter",
                                         "--top", "--score", "--label"))
    band = float(arg("--band", "0.12"))
    jitter = float(arg("--jitter", "2.5"))
    top = int(arg("--top", "15"))
    url = arg("--url", "http://localhost:6333") or None
    store = PersonVectorStore(path=arg("--path", "qdrant_data"), url=url)

    tracklets = R._gather_tracklets(store, run_id)
    if not tracklets:
        raise SystemExit(f"[review] run_id {run_id!r} has no observations here.")

    lines = []
    remap = R.reconcile_tracklets(_ReadOnlyStore(store), run_id=run_id,
                                 log=lines.append, **kw)
    if not remap:
        raise SystemExit("[review] reconcile produced no assignments.")

    by_gid = defaultdict(set)
    for k, gid in remap.items():
        by_gid[gid].add(k)
    gid_of = dict(remap)

    protos = {k: R._prototype(tracklets[k]["vectors"]) for k in remap}
    rows = {k: R._unit_rows(tracklets[k]["vectors"]) for k in remap}
    cap = kw["max_observations_per_side"]
    per_cam = kw["same_camera_thresholds"] or {}
    covis_enabled, covis_pairs = kw["covisibility"]

    def cluster_rows(members):
        mats = [rows[k] for k in sorted(members)]
        stacked = mats[0] if len(mats) == 1 else np.concatenate(mats, axis=0)
        return R._subsample_rows(stacked, cap)

    def score(ma, mb):
        return R.score_observation_sets(
            cluster_rows(ma), cluster_rows(mb),
            R._cluster_prototype(sorted(ma), protos),
            R._cluster_prototype(sorted(mb), protos),
            mode=kw["scoring"], top_frac=kw["consensus_top_frac"], cap=cap)

    def veto(ma, mb):
        """(reason, worst_overlap_seconds) or (None, None)."""
        worst = None
        for a in sorted(ma):
            for b in sorted(mb):
                if a[0] == b[0]:
                    if not R._spans_disjoint(tracklets[a]["span"],
                                             tracklets[b]["span"]):
                        return (f"same-cam overlap {a[0]}({a[1]}/{b[1]})", None)
                    continue
                if not covis_enabled:
                    continue
                pair = frozenset((a[0], b[0]))
                if pair not in covis_pairs or covis_pairs[pair] is None:
                    continue
                tol = covis_pairs[pair]
                secs = R.temporal_overlap_sec(tracklets[a]["span_ts"],
                                              tracklets[b]["span_ts"])
                if secs is not None and secs > tol:
                    if worst is None or secs > worst[1]:
                        worst = (f"{a[0]}({a[1]})<->{b[0]}({b[1]}) "
                                 f"{secs:.1f}s>{tol:.1f}s", secs)
        return worst if worst else (None, None)

    # ---- grade the setting against every label ever recorded ---------------
    labels = load_labels()
    graded = []
    for pair, rec in labels.items():
        try:
            ka, kb = (parse_key(s) for s in sorted(pair))
        except ValueError:
            continue
        if ka not in gid_of or kb not in gid_of:
            continue                      # suppressed, or a different run's data
        together = gid_of[ka] == gid_of[kb]
        want_together = rec["verdict"] == "same"
        graded.append((rec, ka, kb, together, together == want_together))

    header(f"CALIBRATION SUMMARY -- run {run_id}")
    print(f"  settings: {describe_reconcile_kwargs(kw)}\n")

    weak = [float(ln.split("cosine ")[1].split()[0]) for ln in lines
            if "cross-camera cluster merge" in ln]
    below = sum(1 for s in weak if s < STRANGER_CEILING)
    multi = sum(1 for ms in by_gid.values() if len({k[0] for k in ms}) > 1)
    print(f"  {'OK ' if below == 0 else 'BAD'} false merges below impostor "
          f"ceiling ({STRANGER_CEILING}): {below}")
    print(f"      cross-camera identities: {multi}")
    print(f"      identities total: {len(by_gid)}  (from {len(remap)} tracklets)")

    if graded:
        right = sum(1 for g in graded if g[4])
        stated = [g for g in graded if g[0].get("stated")]
        right_stated = sum(1 for g in stated if g[4])
        print(f"\n  {'OK ' if right == len(graded) else 'BAD'} "
              f"operator labels satisfied: {right}/{len(graded)}"
              f"   (operator-STATED only: {right_stated}/{len(stated)})")
        for rec, ka, kb, together, ok in graded:
            if ok:
                continue
            got = "MERGED" if together else "SPLIT"
            want = "same person" if rec["verdict"] == "same" else "different people"
            print(f"      WRONG: {key_str(ka)} <-> {key_str(kb)}  "
                  f"labelled {want}, got {got}  (score {score({ka}, {kb}):.3f})")
    else:
        print(f"\n      no labels apply to this run yet -- run with --label")

    if flag("--score"):
        return

    # ---- uncertain links ---------------------------------------------------
    gids = sorted(by_gid)
    uncertain = []
    for i, ga in enumerate(gids):
        for gb in gids[i + 1:]:
            ma, mb = by_gid[ga], by_gid[gb]
            cams_a = {k[0] for k in ma}
            cams_b = {k[0] for k in mb}
            # Only pairs that would put one identity in a camera it is missing --
            # those are the ones an operator can SEE as one person, two numbers.
            if not (cams_a - cams_b or cams_b - cams_a):
                continue
            shared = cams_a & cams_b
            bar = (R.strictest_same_camera_bar(shared, per_cam,
                                               kw["same_camera_threshold"])
                   if shared else kw["threshold"])
            reason, secs = veto(ma, mb)
            s = score(ma, mb)
            if reason is not None:
                # A veto inside the clock's own jitter is not evidence.
                if secs is not None and secs <= jitter:
                    uncertain.append((s, ga, gb, bar,
                                      f"VETO {reason} -- inside {jitter}s jitter"))
                continue
            if bar - band <= s < bar and s >= STRANGER_CEILING:
                uncertain.append((s, ga, gb, bar,
                                  f"score {s:.3f} vs bar {bar:.2f}"
                                  + (" (shared cam)" if shared else "")))
    uncertain.sort(reverse=True)

    header(f"REMAINING UNCERTAIN LINKS  ({len(uncertain)})")
    if not uncertain:
        print("  none: every refused link is either clearly below the bar or "
              "blocked by\n  a simultaneity overlap far outside the clock's "
              "jitter.")
    else:
        print(f"  Refused, but the evidence does not clearly support refusing. "
              f"Top {top}:\n")
        for n, (s, ga, gb, bar, why) in enumerate(uncertain[:top], 1):
            lbl = labels.get(frozenset((key_str(sorted(by_gid[ga])[0]),
                                        key_str(sorted(by_gid[gb])[0]))))
            mark = f"   [labelled {lbl['verdict']}]" if lbl else ""
            print(f"  {n:2}. reid {ga} <-> reid {gb}   {why}{mark}")
            print(f"      reid {ga}: {fmt_members(by_gid[ga])}")
            print(f"      reid {gb}: {fmt_members(by_gid[gb])}")

    if flag("--label") and uncertain:
        header("RECORD VERDICTS")
        print("  For each: s = same person, d = different, u = unsure/skip, q = quit\n")
        for s, ga, gb, bar, why in uncertain[:top]:
            a, b = sorted(by_gid[ga])[0], sorted(by_gid[gb])[0]
            print(f"  reid {ga} [{fmt_members(by_gid[ga])}]")
            print(f"  reid {gb} [{fmt_members(by_gid[gb])}]   ({why})")
            try:
                ans = input("    same / different / unsure [s/d/u/q]: ").strip().lower()
            except EOFError:
                break
            if ans.startswith("q"):
                break
            if ans.startswith("s"):
                append_label(run_id, a, b, "same")
                print("    -> recorded SAME")
            elif ans.startswith("d"):
                append_label(run_id, a, b, "different")
                print("    -> recorded DIFFERENT")
            else:
                print("    -> skipped")
        print(f"\n  labels file: {LABELS_PATH}")
        print("  Re-run with --score to grade any setting against all of them.")
    elif uncertain:
        header("WHAT TO DO WITH THESE")
        print("""  Four choices, and the point of the list is that you pick knowingly:

    1. ACCEPT them      -- lower the bar / raise the tolerance until they merge.
                           Costs whatever false merges sit at the same scores.
    2. REJECT them      -- what this setting does now. Costs the true links in
                           the list, which show up as one person, two numbers.
    3. LEAVE UNRESOLVED -- do not assert either way: the thin-evidence tracklets
                           render grey with no number. Needs the quality gate,
                           not a threshold.
    4. REVIEW them      -- `--label` walks this list and records your verdicts to
                           calibration/link_labels.jsonl.

  Option 4 is the only one that makes the next run easier instead of the same
  amount of guessing. `--score` then grades ANY candidate setting against every
  label ever recorded, across runs -- which is the number that generalises.""")


if __name__ == "__main__":
    main()
