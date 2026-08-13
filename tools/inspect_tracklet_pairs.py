#!/usr/bin/env python3
"""
inspect_tracklet_pairs.py -- read-only. Prints the numbers that decide the
open questions, using the SAME code path production uses.

Reads nothing but the store. Writes nothing, anywhere. Runs no model, needs no
GPU, needs no labels, needs no camera time.

    python tools/inspect_tracklet_pairs.py                 # newest run in the store
    python tools/inspect_tracklet_pairs.py --run <id>       # a specific run
    python tools/inspect_tracklet_pairs.py --list           # what runs exist

It answers, in order:

  0. WHICH SETTINGS ARE ACTUALLY IN FORCE. resolve_reconcile_kwargs +
     describe_reconcile_kwargs against the real config.yaml. If
     same_camera_rounds / same_camera_reciprocal_best / scoring are not what you
     assumed, stop reading and fix that first.

  1. SAME-CAMERA PAIRWISE COSINES, per camera, with that camera's bar applied and
     each tracklet's mutual-best partner marked. This is where the cam_206
     question lives: an edge ABOVE the bar that mutual-best refuses is a
     documented stranding, and the number that decides whether
     same_camera_rounds can fix it (member pair vs cluster) is printed here.

  2. CAMERA BIAS, three ways: pairwise cosine between camera mean features; the
     across-camera variance profile of those means; and the SAME-CAMERA
     NEIGHBOUR RATE -- for each tracklet, what fraction of its top-k nearest
     neighbours share its camera. The last is the most direct measure of camera
     bias and needs no identity labels at all.

  3. CROSS-CAMERA PAIR SCORES, grouped by camera pair, against the single global
     cross bar. If one pair's same-person scores sit systematically below another
     pair's, a single global `threshold` cannot serve both.

Nothing here is a verdict. A cluster count cannot tell you whether a cluster is
one person or three, and neither can any number below -- render and watch.
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, "src")

import numpy as np
import yaml

from database.store import PersonVectorStore
from identity.reconcile import (_gather_tracklets, _prototype, _unit_rows,
                                describe_reconcile_kwargs,
                                resolve_reconcile_kwargs,
                                resolve_same_camera_thresholds,
                                score_observation_sets)


def open_store(cfg):
    """Same precedence the pipeline uses: QDRANT_URL -> store.url -> store.path."""
    store_cfg = (cfg or {}).get("store") or {}
    url = os.environ.get("QDRANT_URL") or store_cfg.get("url")
    if url:
        return PersonVectorStore(url=url), f"url={url}"
    path = store_cfg.get("path") or "qdrant_data"
    return PersonVectorStore(path=path), f"path={path}"


def inventory(store):
    """{run_id: {camera: n_observations}} -- payload only, no vectors."""
    runs = defaultdict(lambda: defaultdict(int))
    offset = None
    while True:
        pts, offset = store.client.scroll(
            store.collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            runs[pl.get("run_id")][pl.get("camera")] += 1
        if offset is None:
            break
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="run_id. Default: the run with the most observations.")
    ap.add_argument("--list", action="store_true", help="list runs and exit")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--topk", type=int, default=5,
                    help="k for the same-camera neighbour rate (section 2c)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}

    # ---------------------------------------------------------------- section 0
    print("=" * 78)
    print("0. SETTINGS IN FORCE (from %s)" % args.config)
    print("=" * 78)
    kw = resolve_reconcile_kwargs(cfg, log=lambda m: print("   " + str(m)))
    print("   " + describe_reconcile_kwargs(kw))
    for flag in ("same_camera_rounds", "same_camera_reciprocal_best"):
        if not kw.get(flag):
            print(f"   !! {flag} is OFF. reconcile.py's own comments describe it as "
                  f"the fix for a documented failure.")
    if kw.get("scoring") == "prototype":
        print("   !! scoring=prototype. reconcile.py's own comment: 'a mean is the "
              "wrong summary for a person who changes appearance mode'.")
    print()

    store, how = open_store(cfg)
    print(f"   store: {how}  collection={store.collection}")

    runs = inventory(store)
    if not runs:
        print("\n   STORE IS EMPTY. Nothing below can run. (main.py --reset clears "
              "the store and then runs the pipeline, so it looks like a normal run.)")
        return 1

    print("\n   runs present:")
    for rid, cams in sorted(runs.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(cams.values())
        detail = ", ".join(f"{c}={n}" for c, n in sorted(cams.items(), key=lambda t: str(t[0])))
        print(f"     {rid}  {total:6d} obs   [{detail}]")
    if args.list:
        return 0

    run_id = args.run or max(runs, key=lambda r: sum(runs[r].values()))
    print(f"\n   using run_id = {run_id}"
          + ("" if args.run else "  (most observations; override with --run)"))

    tracklets = _gather_tracklets(store, run_id)
    if not tracklets:
        print("   no tracklets gathered for this run.")
        return 1

    protos, rows = {}, {}
    for k, info in tracklets.items():
        p = _prototype(info["vectors"])
        if p is None:
            print(f"   !! degenerate prototype: {k} -- skipped")
            continue
        protos[k] = p
        rows[k] = _unit_rows(info["vectors"])
    keys = sorted(protos)
    dim = rows[keys[0]].shape[1]
    print(f"   {len(keys)} tracklet(s), {dim}-d embeddings")
    if dim != 2048:
        print(f"   !! {dim}-d, not 2048. These are NOT FastReID vectors -- most "
              f"likely a pre-switch run, or a collection at the old width. Every "
              f"number below describes a feature space that is not running.")

    min_obs = kw["min_tracklet_observations"]
    bars = resolve_same_camera_thresholds(
        ((cfg.get("identity") or {}).get("reconcile") or {}), log=lambda m: None)

    def bar_for(cam):
        return bars.get(cam, kw["same_camera_threshold"])

    def score(a, b):
        return score_observation_sets(
            rows[a], rows[b], protos[a], protos[b],
            mode=kw["scoring"], top_frac=kw["consensus_top_frac"],
            cap=kw["max_observations_per_side"])

    def disjoint(a, b):
        sa, sb = tracklets[a]["span"], tracklets[b]["span"]
        return sa[1] < sb[0] or sb[1] < sa[0]

    # ---------------------------------------------------------------- section 1
    print()
    print("=" * 78)
    print("1. SAME-CAMERA PAIRS  (scoring=%s)" % kw["scoring"])
    print("=" * 78)
    print("   ABOVE-BAR + not mutual-best = a documented stranding. For each one,")
    print("   the line below it is the number that decides whether")
    print("   same_camera_rounds can fix it: round 2 must ALSO clear every member")
    print("   pair, and round 2's member tests are a subset of round 1's.")

    by_cam = defaultdict(list)
    for k in keys:
        by_cam[k[0]].append(k)

    for cam in sorted(by_cam, key=str):
        here = sorted(by_cam[cam])
        bar = bar_for(cam)
        print(f"\n   --- {cam}   bar={bar:.2f}   {len(here)} tracklet(s) ---")
        thin = [k for k in here if len(tracklets[k]["vectors"]) < min_obs]
        if thin:
            print(f"       suppressed at min_obs={min_obs}: "
                  + ", ".join(f"{k[1]}({len(tracklets[k]['vectors'])})" for k in thin))
        here = [k for k in here if k not in set(thin)]
        if len(here) < 2:
            print("       fewer than 2 usable tracklets -- nothing to pair")
            continue

        pair_s, best = {}, {}
        for i, a in enumerate(here):
            for b in here[i + 1:]:
                s = score(a, b)
                pair_s[(a, b)] = s
                if disjoint(a, b) and s >= bar:
                    if s > best.get(a, (-1, None))[0]:
                        best[a] = (s, b)
                    if s > best.get(b, (-1, None))[0]:
                        best[b] = (s, a)

        for (a, b), s in sorted(pair_s.items(), key=lambda kv: -kv[1]):
            ok = disjoint(a, b)
            mutual = (best.get(a, (0, None))[1] == b
                      and best.get(b, (0, None))[1] == a)
            if not ok:
                tag = "TIME-OVERLAP (provably 2 people)"
            elif s < bar:
                tag = "below bar"
            elif mutual:
                tag = "MUTUAL BEST -> merges"
            else:
                pa = best.get(a, (0.0, None))[1]
                pb = best.get(b, (0.0, None))[1]
                tag = ("ABOVE BAR, REFUSED (not mutual): %s prefers %s, %s prefers %s"
                       % (a[1], pa[1] if pa else "-", b[1], pb[1] if pb else "-"))
            print(f"       {a[1]:>5} <-> {b[1]:<5} {s:.4f}   {tag}")
            if ok and s >= bar and not mutual:
                # The stranded edge. Round 2 would ask: can the loser join the
                # cluster the winner formed? Complete linkage requires EVERY member
                # pair, so print them.
                for x, y in ((a, b), (b, a)):
                    partner = best.get(y, (0, None))[1]
                    if partner is None or partner == x:
                        continue
                    ms = pair_s.get((x, partner)) or pair_s.get((partner, x))
                    if ms is None:
                        continue
                    verdict = "CLEARS" if ms >= bar else "FAILS -> complete linkage blocks the chain"
                    print(f"             member pair {x[1]} <-> {partner[1]} = "
                          f"{ms:.4f}  {verdict} (bar {bar:.2f})")

    # ---------------------------------------------------------------- section 2
    print()
    print("=" * 78)
    print("2. CAMERA BIAS")
    print("=" * 78)

    cam_vecs = defaultdict(list)
    for k in keys:
        cam_vecs[k[0]].extend(tracklets[k]["vectors"])
    cams = sorted(cam_vecs, key=str)
    means = {c: _unit_rows(cam_vecs[c]).mean(axis=0) for c in cams}

    print("\n   2a. cosine between camera MEAN features")
    print("       Expect all pairs high -- the shared 'person' direction dominates.")
    print("       Read the SPREAD between pairs, not the absolute values.")
    for c in cams:
        print(f"       {c}: n={len(cam_vecs[c])}  |mean|={np.linalg.norm(means[c]):.4f}")
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            ca = means[a] / max(float(np.linalg.norm(means[a])), 1e-12)
            cb = means[b] / max(float(np.linalg.norm(means[b])), 1e-12)
            print(f"       cos(mean_{a}, mean_{b}) = {float(ca @ cb):.4f}")

    if len(cams) >= 2:
        print("\n   2b. across-camera variance of the camera means, by dimension")
        print("       If a few dimensions dominate, centering only those bounds how")
        print("       much identity information a camera mean can absorb "
              "(camera_centering.dims).")
        var = np.stack([means[c] for c in cams]).var(axis=0)
        order = np.argsort(var)[::-1]
        tot = float(var.sum()) or 1.0
        for n in (10, 50, 200, 512):
            if n <= len(var):
                frac = float(var[order[:n]].sum()) / tot
                print(f"       top {n:>4} dims carry {frac * 100:5.1f}% of the "
                      f"camera-mean variance")

    print(f"\n   2c. same-camera neighbour rate (top-{args.topk}, prototypes)")
    print("       The most direct measure of camera bias here: no identity labels")
    print("       needed. A camera whose tracklets mostly retrieve THEMSELVES is")
    print("       biased. Compare against its share of all tracklets -- a rate at")
    print("       or near that share is unbiased; well above it is not.")
    if len(keys) > args.topk:
        mat = np.stack([protos[k] for k in keys])
        sims = mat @ mat.T
        np.fill_diagonal(sims, -np.inf)
        per_cam_hits, per_cam_n = defaultdict(int), defaultdict(int)
        for i, k in enumerate(keys):
            nn = np.argsort(sims[i])[::-1][:args.topk]
            per_cam_hits[k[0]] += sum(1 for j in nn if keys[j][0] == k[0])
            per_cam_n[k[0]] += args.topk
        for c in cams:
            if not per_cam_n[c]:
                continue
            rate = per_cam_hits[c] / per_cam_n[c]
            share = (sum(1 for k in keys if k[0] == c) - 1) / max(len(keys) - 1, 1)
            flag = "  <-- well above share" if rate > share + 0.25 else ""
            print(f"       {c}: {rate * 100:5.1f}% same-camera "
                  f"(share of tracklets {share * 100:5.1f}%){flag}")
    else:
        print(f"       only {len(keys)} tracklets -- need more than topk={args.topk}")

    # ---------------------------------------------------------------- section 3
    print()
    print("=" * 78)
    print("3. CROSS-CAMERA PAIRS   (single global bar = %.2f)" % kw["threshold"])
    print("=" * 78)
    print("   PER_CAMERA_KEYS notes that `threshold` is a property of a camera PAIR,")
    print("   yet one global bar serves every pair. If these distributions differ")
    print("   between pairs, that is the gap.")
    print()
    print("   !! SCOPE. These are SINGLETON tracklet pairs, so the cross bar is the")
    print("      right one. It is NOT the bar that decides the operator's split:")
    print("      once both clusters have absorbed a shared camera, pair_threshold")
    print("      switches to strictest_same_camera_bar (0.90 for cam_219) and the")
    print("      cross bar is never consulted. Only explain_merge_failure.py can")
    print("      show that gate -- run it on the two reids of a known split person.")

    per_pair = defaultdict(list)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] == b[0]:
                continue
            per_pair[frozenset((a[0], b[0]))].append((score(a, b), a, b))

    for pair, scored in sorted(per_pair.items(), key=lambda kv: sorted(map(str, kv[0]))):
        c1, c2 = sorted(map(str, pair))
        vals = np.array([s for s, _, _ in scored])
        above = int((vals >= kw["threshold"]).sum())
        print(f"\n   --- {c1} <-> {c2}   {len(scored)} pair(s), "
              f"{above} above {kw['threshold']:.2f} ---")
        print(f"       max {vals.max():.4f}  p95 {np.percentile(vals, 95):.4f}  "
              f"median {np.percentile(vals, 50):.4f}  min {vals.min():.4f}")
        for s, a, b in sorted(scored, reverse=True)[:5]:
            mark = ">=bar" if s >= kw["threshold"] else "     "
            print(f"       {mark} {a[0]}:{a[1]} <-> {b[0]}:{b[1]}  {s:.4f}")

    print()
    print("=" * 78)
    print("Read nothing as a verdict. Section 1's refusals are mechanisms, not")
    print("causes; section 2 says whether centering is worth trying, not whether it")
    print("works. The only way to know is rerender_from_clips.py and watching it.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
