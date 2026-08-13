"""
Re-render a FINISHED run's videos at different reconcile settings -- no cameras,
no models, no walking the room again.

    python tests/calibration/rerender_from_clips.py <run_id>
    python tests/calibration/rerender_from_clips.py <run_id> --cross 0.70,0.75
    python tests/calibration/rerender_from_clips.py <run_id> --cross 0.75 \
        --same "cam_213=0.80" --out "cmp_{name}_strict.mp4"

WHY THIS EXISTS. Judging identity is a VISUAL act -- "that is the same guy" is
something an operator sees, not something a metric reports. But until now seeing
a different threshold meant capturing a whole new run: four cameras, people
walking, five minutes, a finalization that must not be interrupted, and a fresh
set of track ids that cannot be compared to the last set. Two runs were lost to
that loop, and every threshold answer cost one.

A run leaves behind everything needed to redo the last two stages:
  * ._live_src_<cam>.mp4             the CLEAN processed frames
  * ._live_src_<cam>.annotations.json  per-frame box geometry, index-aligned
  * the embeddings in Qdrant, under this run_id

So reconcile + render can be replayed at any settings, on the SAME footage and
the SAME track ids, and the results watched side by side. Requires the run to
have been captured with `live.reconcile.keep_frames: true`.

WHAT IT REPLAYS: everything downstream of the store -- every threshold,
reciprocal-best, min_tracklet_observations, and any change to reconcile's
scoring. WHAT IT CANNOT: anything that changes what got recorded -- the detector
model, imgsz, reid.interval, crop quality. Those still need a live run.

READ-ONLY. The gid map is taken from what reconcile RETURNS, never from ids
stamped into the store, so re-rendering cannot rewrite the gallery and two
settings rendered back-to-back cannot contaminate each other.
"""

import glob
import json
import os
import sys

from _common import arg, bootstrap, header, reconcile_settings

# Where the user invoked us, captured BEFORE bootstrap() -- it chdirs to the repo
# root so relative model paths resolve, which would otherwise silently look for
# clips in the wrong directory. In production the clips ARE in the repo root, so
# that would have worked by luck and broken the moment they were not.
INVOKED_FROM = os.getcwd()
ROOT = bootstrap()
sys.path.insert(0, ROOT)             # for main.render_final_videos

from database.store import PersonVectorStore                      # noqa: E402
from identity.reconcile import (SCORING_MODES,                     # noqa: E402
                                describe_reconcile_kwargs,
                                reconcile_tracklets)

class _ReadOnlyStore:
    """Real observations in, no writes out -- see sweep_reconcile_thresholds."""

    def __init__(self, store, min_visible_kp=0.0):
        self.client = (store.client if not min_visible_kp
                       else _VisibilityFilteredClient(store.client, min_visible_kp))
        self.collection = store.collection

    def set_global_id(self, point_ids, global_id):
        pass

    def clear_global_id(self, point_ids):
        pass
class _VisibilityFilteredClient:
    """A qdrant client whose `scroll` drops observations below a visibility floor.

    WHY FILTER HERE RATHER THAN GATE AT CAPTURE. A crop rejected during a run is
    never embedded, so no vector exists and the decision is permanent: you cannot
    afterwards ask what a different floor would have done without walking the
    room again. Recording the metric and filtering offline turns one capture into
    a watchable video at every candidate floor, on identical footage with
    identical track ids -- which is the only way two settings have ever been
    honestly compared here (see this module's docstring).

    FAILS OPEN on a missing `visible_kp`. A run captured before the pose patch,
    or with a box-only checkpoint, has no keypoints at all -- and treating
    "unmeasured" as "invisible" would silently delete every such run's
    observations while looking like a threshold result. Same contract as a
    missing `floor` block in reconcile and a missing `ts`: unknown is not a
    verdict.

    Delegates everything else to the real client, so this can never change what
    reconcile reads apart from which points it sees.
    """

    def __init__(self, client, floor):
        self._client = client
        self._floor = float(floor)
        self.kept = 0
        self.dropped = 0
        self.unmeasured = 0

    def __getattr__(self, name):
        return getattr(self._client, name)

    def scroll(self, *args, **kwargs):
        points, offset = self._client.scroll(*args, **kwargs)
        out = []
        for p in points:
            quality = (p.payload or {}).get("crop_quality") or {}
            value = (p.payload or {}).get("visible_kp", quality.get("visible_kp"))
            if value is None:
                self.unmeasured += 1
                out.append(p)
            elif float(value) >= self._floor:
                self.kept += 1
                out.append(p)
            else:
                self.dropped += 1
        return out, offset

    def describe(self):
        total = self.kept + self.dropped + self.unmeasured
        if not total:
            return "no observations scrolled"
        if not self.kept and not self.dropped:
            return (f"{self.unmeasured} observation(s) carry NO visible_kp -- this "
                    f"run predates the pose detector, so the floor did nothing")
        return (f"kept {self.kept}, dropped {self.dropped} "
                f"({100.0 * self.dropped / max(1, self.kept + self.dropped):.1f}% "
                f"of measured), {self.unmeasured} unmeasured and therefore kept")

def load_clips(pattern="._live_src_*.mp4", run_id=None):
    """-> [(camera, clip_path, annotations)] for every clip WITH its sidecar.

    REFUSES a clip whose sidecar names a DIFFERENT run_id, and this is a hard
    refusal rather than a warning.

    THE FAILURE IT PREVENTS, which already happened. Clip filenames carry no
    run_id -- `._live_src_<cam>.mp4` is a fixed name that every run OVERWRITES --
    while `keep_frames: true` guarantees the files persist. So the clips on disk
    belong to the LAST run, and re-rendering an EARLIER run silently drew that
    earlier run's stored ids onto the later run's pixels. ByteTrack restarts its
    numbering every run, so the two id spaces are near-disjoint: measured on
    20260804_064551 reconciled against 20260804_120409's clips, cam_219 came out
    100% UNRESOLVED and 84% of all person-boxes carried no identity. The video
    looks like a catastrophic identity failure and is really a join error.

    That is ONBOARDING 5.6's "stale files look like success" one level deeper and
    worse: a stale OUTPUT is merely stale, but a stale CLIP produces a plausible
    composite of two different runs. The run_id has been in the sidecar all along
    (render.py `_write_annotations` writes it); nothing checked it.
    """
    found, missing, wrong = [], [], []
    for clip in sorted(glob.glob(pattern)):
        side = os.path.splitext(clip)[0] + ".annotations.json"
        if not os.path.exists(side):
            missing.append(clip)
            continue
        with open(side) as f:
            blob = json.load(f)
        cam = blob.get("camera") or os.path.basename(clip)[len("._live_src_"):-4]
        clip_run = blob.get("run_id")
        if run_id is not None and clip_run is not None and clip_run != run_id:
            wrong.append((clip, clip_run))
            continue
        if run_id is not None and clip_run is None:
            print(f"[rerender] {clip}: sidecar records NO run_id (written before "
                  f"that field existed) -- CANNOT verify it belongs to {run_id}. "
                  f"Check the frame count against the run's stored frame span "
                  f"before trusting anything it renders.")
        found.append((cam, clip, blob))
    for clip in missing:
        print(f"[rerender] {clip} has no .annotations.json sidecar -- skipped. "
              f"(Runs captured before that sidecar existed cannot be re-rendered; "
              f"the box geometry was only ever in memory.)")
    if wrong:
        print(f"[rerender] REFUSED {len(wrong)} clip(s) belonging to a DIFFERENT run:")
        for clip, clip_run in wrong:
            print(f"[rerender]   {clip} is from run {clip_run}, not {run_id}")
        print(f"[rerender] Clip filenames carry no run_id and every run OVERWRITES "
              f"them, so {run_id}'s footage is gone unless it was copied aside. "
              f"Rendering it against these clips would draw {run_id}'s ids onto "
              f"another run's pixels -- which looks exactly like a total identity "
              f"failure. Re-render run {wrong[0][1]} instead, or use --clips on a "
              f"directory holding the right run's copies.")
    return found


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        raise SystemExit("usage: rerender_from_clips.py <run_id> "
                         "[--cross 0.70,0.75] [--same cam_213=0.80] "
                         "[--out 'cmp_{name}_{tag}.mp4']")
    run_id = sys.argv[1]

    # Base = exactly what production would run; only the swept axes are overridden.
    # This tool used to pass neither `covisibility` nor
    # `same_camera_reciprocal_best`, so the video it produced was rendered from a
    # clustering that does not ship -- and the video is the ONLY ground truth this
    # project has. See REMEDIATION_PLAN.md Part M.
    base_kw = reconcile_settings(owns=("--cross", "--scoring"),
                                 extra_flags=("--url", "--path", "--out",
                                              "--fps", "--dir",
                                              "--min-visible-kp", "--collection"))
    crosses = [float(x) for x in
               (arg("--cross") or f"{base_kw['threshold']:.2f}").split(",")]
    scorings = (arg("--scoring", "") or "").split(",")
    scorings = [m.strip() for m in scorings if m.strip()] or [None]
    for m in scorings:
        if m is not None and m not in SCORING_MODES:
            raise SystemExit(f"[rerender] unknown --scoring {m!r}; "
                             f"expected any of {list(SCORING_MODES)}")
    out_pattern = arg("--out", "rerender_{name}_{tag}.mp4")
    fps = float(arg("--fps", "0") or 0)

    # Work where the clips are, so the re-rendered videos land beside them.
    clip_dir = os.path.abspath(arg("--dir", INVOKED_FROM))
    os.chdir(clip_dir)
    print(f"[rerender] clips from {clip_dir}")

    clips = load_clips(run_id=run_id)
    if not clips:
        raise SystemExit(
            "[rerender] no ._live_src_*.mp4 + .annotations.json pairs in this "
            "directory.\n"
            "           Set live.reconcile.keep_frames: true and capture a run "
            "first -- without it\n"
            "           the clips are deleted as soon as the final render "
            "finishes.")
    print(f"[rerender] {len(clips)} camera(s): "
          + ", ".join(f"{c} ({b['frames']} frames @ {b['clip_fps']}fps)"
                      for c, _, b in clips))

    store = PersonVectorStore(path=arg("--path", "qdrant_data"),
                              url=arg("--url", "http://localhost:6333") or None)
    coll = arg("--collection", "")
    if coll:
        # SIDE collection (embed_run_with_onnx.py writes one). Without this the
        # render reads `persons` and you watch the SHIPPING model's identities
        # while believing they are the new model's.
        store.collection = coll
        print(f"[rerender] collection: {coll}")
    min_vk = float(arg("--min-visible-kp", "0") or 0)
    ro = _ReadOnlyStore(store, min_visible_kp=min_vk)
    print(f"[rerender] base settings: {describe_reconcile_kwargs(base_kw)}"
          + ("" if not min_vk else f" min_visible_kp={min_vk:.2f}"))

    from main import render_final_videos                          # noqa: E402

    jobs = [(cam, clip) for cam, clip, _ in clips]
    shared = {"annotations": {cam: blob["annotations"] for cam, _, blob in clips}}

    combos = [(m, c) for m in scorings for c in crosses]
    for mode, cross in combos:
        tag = f"x{cross:.2f}".replace(".", "")
        if mode is not None:
            tag += f"_{mode}"
        kw = dict(base_kw)
        kw["threshold"] = cross
        if mode is not None:
            kw["scoring"] = mode
        header(f"cross={cross:.2f}  {describe_reconcile_kwargs(kw)}")
        lines = []
        # Fresh per combo. reconcile fills these, and a dict left over from the
        # previous setting would label this render with the last one's scores.
        quality = {}
        links = {}
        remap = reconcile_tracklets(ro, run_id=run_id, log=lines.append,
                                    quality_out=quality, link_out=links, **kw)
        if not remap:
            # Print the log BEFORE bailing. An empty remap is the one case where
            # the reason matters most and the old order threw it away.
            for _ln in lines:
                print(f"  {_ln}")
            print("[rerender] reconcile produced NO assignments for this run_id "
                  "-- nothing to draw. Check the run_id against the store.")
            continue
        print(f"  {len(remap)} tracklet(s) -> {len(set(remap.values()))} identities")
        if min_vk and hasattr(ro.client, "describe"):
            # Printed per combo because reconcile scrolls the store on every call,
            # so the counters accumulate -- a reader comparing two settings needs
            # to know the second number includes the first pass.
            print(f"  visibility filter @ {min_vk:.2f}: {ro.client.describe()}")
        # PRINT THE DECISIONS, not just the count. `lines` has always collected
        # reconcile's full log via log=lines.append and then DISCARDED it, so this
        # tool reported "20 tracklets -> 12 identities" and nothing about WHICH
        # merges a setting caused. That is precisely the failure CLAUDE.md section 4
        # names -- a cluster count cannot tell you whether a cluster is one person or
        # three -- in the one offline feedback loop this project has. A comparison of
        # two settings is unreadable without these lines: 15 -> 12 is three merges,
        # and which three is the entire question.
        for _ln in lines:
            print(f"  {_ln}")

        # Each camera keeps its own recorded rate unless overridden, so the
        # re-render does not inherit the single global output fps that makes
        # cam_224 play fast and cam_219 slow (plan #45/#46).
        for cam, clip, blob in clips:
            # measured_fps is the rate the frames were really produced at;
            # clip_fps is only what the container was tagged with (the old global
            # default), which is why an earlier re-render still played cam_224 fast.
            cam_fps = fps or float(blob.get("measured_fps")
                                   or blob.get("clip_fps") or 20.0)
            render_final_videos(
                [(cam, clip)],
                {"source": {"resize_width": 0},
                 "display": {"output_fps": cam_fps}},
                shared, ro, run_id,
                gid_map=remap,
                quality=quality, links=links,
                out_pattern=out_pattern.replace("{tag}", tag))
        written = [out_pattern.replace("{tag}", tag).format(name=c)
                   for c, _, _ in clips]
        print("  wrote: " + ", ".join(written))

    header("WHAT TO DO WITH THESE")
    print("""  Play the same camera's files from two settings side by side and watch one
  person walk. You are looking for two different failures:

    * one person whose number CHANGES  -> under-merging; the bar is too high
    * one number on two people         -> over-merging; the bar is too low

  Colour is the fast read: ids are coloured by number, so a person changing
  colour mid-walk is a split, and two people sharing a colour is a merge. Watch
  the same person across two CAMERAS too -- carrying one id between them is the
  product claim.

  Note the palette only has 8 colours (plan #35), so two distant ids can collide
  by coincidence. Confirm against the printed number before calling it a merge.""")


if __name__ == "__main__":
    main()
