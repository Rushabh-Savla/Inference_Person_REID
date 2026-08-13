"""
Analyse a reconcile decision log -- the Phase 9 calibration input.

    python tests/calibration/analyze_decision_log.py [logs/reconcile_decisions_*.jsonl]

Reads a JSONL produced by identity.reconcile with a DecisionLog attached and
answers the questions that decide every threshold, from data instead of argument:

  1. WHERE DOES THE BAR CUT?  For each subject, find the largest gap in its sorted
     candidate scores -- that gap IS the same-person / different-person boundary
     for that subject -- and report where the configured threshold sits relative
     to it. A threshold above the gap throws away genuine merges.

  2. ORPHANS.  Subjects with ZERO eligible candidates after the same-camera phase.
     These are the fragments that fail to merge with their own identity and then
     get absorbed cross-camera into whichever cluster they happen to score highest
     against -- the mechanism behind "reid 2 becomes reid 7".

  3. NEAR-TIES: fragmentation or genuine ambiguity?  Discriminated by
     `pair_similarity_to_best`: two tied candidates that are ~0.95 similar to each
     other are ONE person fragmented (enforcing a margin would reject a correct
     merge); ~0.55 means the embedding genuinely cannot separate two people.

  4. RECIPROCAL_BEST, the gate that rejects the most in practice.

  5. CROSS-CAMERA accepted merges vs the measured different-person ceiling.

Needs no GPU, no footage, no model -- just the log. Safe to run repeatedly.
"""

import glob
import json
import sys
from collections import Counter, defaultdict

import numpy as np


def load(path=None):
    if path is None:
        found = sorted(glob.glob("logs/reconcile_decisions_*.jsonl"))
        if not found:
            raise SystemExit("[analyze] no logs/reconcile_decisions_*.jsonl found; "
                             "pass a path as the first argument")
        path = found[-1]
    rows = [json.loads(l) for l in open(path)]
    return path, rows


def header(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def pct(n, d):
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def natural_gap(scores):
    """Largest gap in a descending score list -> (gap, lower_edge, upper_edge).

    The widest gap separates the same-person cluster from the different-person
    cluster far more reliably than any fixed threshold, because it is derived from
    that subject's own distribution.
    """
    s = sorted(scores, reverse=True)
    if len(s) < 2:
        return None
    gaps = [(s[i] - s[i + 1], s[i + 1], s[i]) for i in range(len(s) - 1)]
    return max(gaps)


def main():
    path, rows = load(sys.argv[1] if len(sys.argv) > 1 else None)
    dec = [r for r in rows if r["type"] == "decision"]
    out = [r for r in rows if r["type"] == "outcome"]
    summ = [r for r in rows if r["type"] == "summary"]
    live = [r for r in dec if r["state"] != "suppressed"]
    p1 = [r for r in live if r["phase"] == "same_camera"]
    p2 = [r for r in live if r["phase"] == "cross_camera"]

    print(f"{path}")
    print(f"  {len(dec)} decisions ({len(dec) - len(live)} suppressed), "
          f"{len(out)} tracklet outcomes")
    print(f"  phase 1 (same-camera): {len(p1)}   phase 2 (cross-camera): {len(p2)}")
    # Built outside the f-strings: a multi-line expression inside an f-string is a
    # syntax error before Python 3.12, and this harness has to run on both.
    states = dict(Counter(r["state"] for r in out))
    gate_fails = dict(Counter(g for r in dec for g in r["gates_failed"]))
    exclusions = dict(Counter(c["excluded_by"] for r in dec
                              for c in r["candidates"] if c["excluded_by"]))
    print(f"  outcome states: {states}")
    print(f"  gate failures : {gate_fails}")
    print(f"  exclusions    : {exclusions}")

    # ---------------------------------------------------------------- 1
    header("1. WHERE DOES THE THRESHOLD CUT?  (same-camera phase)")
    print("  For each subject: the highest score the bar REJECTED, and that subject's")
    print("  own natural same/different boundary (its widest score gap).\n")
    # `gap_thr` records, per bimodal subject, the bar THAT SUBJECT actually faced.
    # Since #40 the bar is per-camera, so a run carries several. Judging every
    # threshold against every subject would blame 0.80 for subjects in a camera
    # that never used it. On a single-threshold log this changes nothing.
    rejected_high, gap_lo, gap_hi, gap_thr = [], [], [], []
    thr_seen, cams_by_thr = set(), defaultdict(set)
    for r in p1:
        thr = r["gate_detail"]["ABSOLUTE_THRESHOLD"]["threshold"]
        if thr is not None:
            thr_seen.add(thr)
            for cam in r["cameras"]:
                cams_by_thr[thr].add(cam)
        below = [c["score"] for c in r["candidates"]
                 if c["excluded_by"] == "BELOW_ABSOLUTE_THRESHOLD"]
        if below:
            rejected_high.append(max(below))
        g = natural_gap([c["score"] for c in r["candidates"]])
        if g and g[0] > 0.05:          # only trust a clearly bimodal subject
            gap_lo.append(g[1])
            gap_hi.append(g[2])
            gap_thr.append(thr)

    print(f"  configured same_camera_threshold: {sorted(thr_seen)}")
    if len(thr_seen) > 1:
        for t in sorted(thr_seen):
            print(f"    {t:.2f}: {sorted(cams_by_thr[t])}")
    if rejected_high:
        a = np.array(rejected_high)
        print(f"\n  highest score REJECTED by the bar, per subject (n={len(a)}):")
        print(f"    mean={a.mean():.3f}  p50={np.median(a):.3f}  "
              f"p95={np.percentile(a, 95):.3f}  max={a.max():.3f}")
        for t in (0.75, 0.80, 0.85, 0.88):
            print(f"    subjects that lost a candidate above {t:.2f}: "
                  f"{sum(1 for x in a if x > t):4d}/{len(a)}  "
                  f"({pct(sum(1 for x in a if x > t), len(a))})")
    if gap_lo:
        lo, hi = np.array(gap_lo), np.array(gap_hi)
        print(f"\n  natural same/different boundary from {len(lo)} bimodal subjects:")
        print(f"    gap LOWER edge (top 'different'): mean={lo.mean():.3f} "
              f"p95={np.percentile(lo, 95):.3f} max={lo.max():.3f}")
        print(f"    gap UPPER edge (worst 'same')   : mean={hi.mean():.3f} "
              f"p5={np.percentile(hi, 5):.3f}  min={hi.min():.3f}")
        band_lo, band_hi = np.percentile(lo, 95), np.percentile(hi, 5)
        if band_lo < band_hi:
            print(f"\n    => a bar in ({band_lo:.3f}, {band_hi:.3f}) separates cleanly "
                  f"on 90% of subjects")
            for t in sorted(thr_seen):
                where = ("INSIDE the clean band" if band_lo < t < band_hi else
                         "ABOVE it -- rejecting genuine merges" if t >= band_hi else
                         "BELOW it -- admitting different people")
                print(f"    configured {t:.2f} is {where}")
        else:
            # p95(lower) exceeds p5(upper): the per-subject boundaries OVERLAP, so no
            # single global threshold can separate cleanly. That is itself the finding
            # -- it argues for a PER-CAMERA threshold rather than a better global one.
            print(f"\n    => NO SINGLE GLOBAL BAR separates cleanly: the per-subject")
            print(f"       boundaries OVERLAP (p95 of 'top different' = {band_lo:.3f} is")
            print(f"       above p5 of 'worst same' = {band_hi:.3f}).")
            print(f"       Median picture: boundary sits between {np.median(lo):.3f} "
                  f"and {np.median(hi):.3f}.")
            print(f"       Check section 5 -- if eligible-set size differs sharply by")
            print(f"       camera, the threshold should be PER-CAMERA, not global.")
        for t in sorted(thr_seen):
            # Only subjects that FACED this bar -- see the note where gap_thr is built.
            own = [x for x, ts in zip(gap_hi, gap_thr) if ts == t]
            above = sum(1 for x in own if t > x)
            where = (f" in {sorted(cams_by_thr[t])}" if len(thr_seen) > 1 else "")
            print(f"    configured {t:.2f} sits above the same-person cluster on "
                  f"{above}/{len(own)} subjects ({pct(above, len(own))}){where} "
                  f"-- those lose genuine merges")

    # ---------------------------------------------------------------- 2
    header("2. ORPHANS -- fragments with nowhere to go")
    orphans = [r for r in p1 if r["eligible_count"] == 0]
    print(f"  subjects with ZERO eligible same-camera partners: "
          f"{len(orphans)}/{len(p1)}  ({pct(len(orphans), len(p1))})")
    print("  These fail to merge with their own identity in phase 1, then phase 2")
    print("  offers them to the cross-camera lane at the LOWER bar -- which is how a")
    print("  fragment ends up in the wrong person's cluster.\n")
    if orphans:
        best_missed = [max((c["score"] for c in r["candidates"]), default=None)
                       for r in orphans]
        best_missed = [b for b in best_missed if b is not None]
        if best_missed:
            a = np.array(best_missed)
            print(f"  their best (rejected) candidate: mean={a.mean():.3f} "
                  f"p95={np.percentile(a, 95):.3f} max={a.max():.3f}")
            print(f"  orphans whose best candidate was above 0.85: "
                  f"{sum(1 for x in a if x > 0.85)}/{len(a)}  <- would merge at a lower bar")
        by_cam = Counter(c for r in orphans for c in r["cameras"])
        print(f"  orphans by camera: {dict(by_cam)}")

    # ---------------------------------------------------------------- 3
    header("3. NEAR-TIES -- fragmentation or genuine ambiguity?")
    frag = ambig = unknown = 0
    frag_margins, ambig_margins = [], []
    for r in live:
        g = r["gate_detail"]["TOP2_MARGIN"]
        m, ru = g.get("margin_eligible"), g.get("runner_eligible")
        if m is None or ru is None or m > 0.05:
            continue
        sim = next((c["pair_similarity_to_best"] for c in r["candidates"]
                    if c["handle"] == ru["handle"]), None)
        if sim is None:
            unknown += 1
        elif sim > 0.85:
            frag += 1
            frag_margins.append(m)
        else:
            ambig += 1
            ambig_margins.append(m)
    total = frag + ambig
    print(f"  near-ties (margin <= 0.05): {total} (+{unknown} unclassifiable)")
    print(f"    runner-up similarity >0.85  -> FRAGMENTATION      : {frag:4d}  "
          f"({pct(frag, total)})")
    print(f"    runner-up similarity <=0.85 -> GENUINE AMBIGUITY  : {ambig:4d}  "
          f"({pct(ambig, total)})")
    if frag and frag > ambig:
        print("\n  => Fragmentation dominates. A TOP2_MARGIN GATE WOULD REJECT CORRECT")
        print("     MERGES here. Keep it logged-only for this lane (threshold: null).")
    elif ambig > frag:
        print("\n  => Genuine ambiguity dominates. A margin gate would be protective;")
        print("     size it from the accepted-vs-rejected distributions below.")

    acc_m = [r["gate_detail"]["TOP2_MARGIN"]["value"] for r in live
             if r["accepted_partner"] and r["gate_detail"]["TOP2_MARGIN"]["value"] is not None]
    rej_m = [r["gate_detail"]["TOP2_MARGIN"]["value"] for r in live
             if not r["accepted_partner"] and r["gate_detail"]["TOP2_MARGIN"]["value"] is not None]
    for name, v in (("accepted", acc_m), ("rejected", rej_m)):
        if v:
            a = np.array(v)
            print(f"  margins on {name:8s} merges: n={len(a):4d} p5={np.percentile(a,5):.4f} "
                  f"median={np.median(a):.4f} p95={np.percentile(a,95):.4f}")

    dis = [r["gate_detail"]["TOP2_MARGIN"].get("runner_up_differs") for r in live]
    real = [d for d in dis if d is not None]
    if real:
        print(f"\n  margin-definition disagreement: {sum(real)}/{len(real)} "
              f"({pct(sum(real), len(real))})   [+{len(dis)-len(real)} with no "
              f"selectable candidate]")

    # ---------------------------------------------------------------- 4
    header("4. RECIPROCAL_BEST -- what is it rejecting?")
    rb = [r for r in live if "RECIPROCAL_BEST" in r["gates_failed"]]
    print(f"  decisions that lost their best candidate to reciprocity: "
          f"{len(rb)}/{len(live)}  ({pct(len(rb), len(live))})")
    if rb:
        sc = np.array([r["gate_detail"]["ABSOLUTE_THRESHOLD"]["value"] for r in rb
                       if r["gate_detail"]["ABSOLUTE_THRESHOLD"]["value"] is not None])
        if len(sc):
            print(f"  scores it rejected: mean={sc.mean():.3f} "
                  f"p95={np.percentile(sc,95):.3f} max={sc.max():.3f}")
        print(f"  by context: {dict(Counter(r['context'] for r in rb))}")
        print(f"  by camera : {dict(Counter(c for r in rb for c in r['cameras']))}")
        sims = []
        for r in rb:
            b = (r["gate_detail"]["TOP2_MARGIN"].get("best") or {}).get("handle")
            s = next((c["pair_similarity_to_best"] for c in r["candidates"]
                      if c["handle"] == b), None)
            if s is not None:
                sims.append(s)
        if sims:
            a = np.array(sims)
            print(f"  similarity of the rejected best to itself (sanity ~1.0): "
                  f"mean={a.mean():.3f}")

    # ---------------------------------------------------------------- 5
    header("5. ELIGIBLE-SET SIZE PER CAMERA -- where is the fragmentation?")
    per_cam = defaultdict(list)
    for r in p1:
        for c in r["cameras"]:
            per_cam[c].append(r["eligible_count"])
    for cam, v in sorted(per_cam.items()):
        a = np.array(v)
        print(f"  {cam}: subjects={len(a):4d}  eligible/subject mean={a.mean():5.1f} "
              f"max={a.max():3d}  zero={sum(1 for x in a if x == 0):3d}")
    print("\n  A high mean = one person split many ways (all fragments see each other).")
    print("  A high zero-count = fragments that found no partner -> orphans.")

    # ---------------------------------------------------------------- 6
    header("6. CROSS-CAMERA ACCEPTED MERGES vs the different-person ceiling")
    acc = [r for r in p2 if r["accepted_partner"]]
    if acc:
        s = np.array([r["gate_detail"]["ABSOLUTE_THRESHOLD"]["value"] for r in acc
                      if r["gate_detail"]["ABSOLUTE_THRESHOLD"]["value"] is not None])
        if len(s):
            print(f"  accepted merges: n={len(s)} min={s.min():.3f} "
                  f"mean={s.mean():.3f} max={s.max():.3f}")
            for t, why in ((0.70, ""), (0.66, "  <- measured stranger ceiling (n=12)")):
                print(f"  accepted below {t:.2f}: {sum(1 for x in s if x < t):3d}"
                      f"/{len(s)}{why}")
            print("\n  Merges below the stranger ceiling are candidate FALSE MERGES.")
            print("  Cross-camera stranger scores run lower than same-camera, so this is")
            print("  suggestive, not proof -- but it is where to look for id swapping.")
    else:
        print("  no accepted cross-camera merges in this log")

    # ---------------------------------------------------------------- 7
    header("7. SUPPRESSION COST")
    supp = [r for r in out if r["state"] == "suppressed"]
    tot_obs = sum(r["observations"] for r in out)
    supp_obs = sum(r["observations"] for r in supp)
    print(f"  suppressed tracklets: {len(supp)}/{len(out)}  "
          f"({pct(supp_obs, tot_obs)} of all observations)")
    if supp:
        print(f"  by camera: {dict(Counter(c for r in supp for c in r['cameras']))}")
        print("  Each renders as a bare 'ID <track_id>' with no identity in the video.")

    # ---------------------------------------------------------------- 8
    header("8. FRAGMENTATION OUTCOME")
    per_id = defaultdict(list)
    for r in out:
        if r["assigned_id"] is not None:
            per_id[r["assigned_id"]].append(r)
    sizes = sorted((len(v) for v in per_id.values()), reverse=True)
    if sizes:
        print(f"  identities: {len(per_id)}   tracklets/identity: "
              f"mean={np.mean(sizes):.2f} max={sizes[0]}  distribution={sizes}")
        multi = {i: sorted({c for r in v for c in r['cameras']})
                 for i, v in per_id.items()}
        print(f"  identities spanning >1 camera: "
              f"{sum(1 for c in multi.values() if len(c) > 1)}/{len(per_id)}")
    if summ:
        md = summ[-1].get("margin_disagreement", {})
        if md:
            print(f"  (summary row) margin disagreement {md.get('rate_pct')}% "
                  f"-- {md.get('band')}")

    header("WHAT TO DO WITH THIS")
    print("""  Section 1 gives the same_camera_threshold answer: if the configured bar sits
  ABOVE the clean band, lower it into the band and re-run. Section 2 quantifies the
  orphan population that lowering it would rescue. Section 3 decides whether
  TOP2_MARGIN should ever gate. Section 6 flags cross-camera merges worth eyeballing
  in the output videos.

  Change ONE thing, re-run, and diff these sections. Do not change a threshold and a
  scoring function in the same run -- see REMEDIATION_PLAN.md Part A on why four
  earlier tuning attempts were reverted without learning anything.""")


if __name__ == "__main__":
    main()
