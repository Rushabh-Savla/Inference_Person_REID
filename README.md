# Multi-Camera Person Re-Identification

Detect and track people across multiple cameras, embed their appearance, and
assign a **reid id** that stays the same for one real person **across cameras**
and **across re-appearances**. Two input modes, one identity result:

- **Live RTSP** (`--mode live`) — reads N streams in real time, never recording
  the raw feed; on stop it runs the offline reconcile and writes corrected
  `output_<cam>.mp4`. Real-time targets a GPU server.
- **Video files** (default / `--mode batch`) — the same detect → track → embed →
  reconcile → re-render flow on recorded footage. Runs on CPU.

The pipeline is fully self-contained — it builds and owns its own Qdrant gallery;
there is no separate registration step or external service. Both modes settle
cross-camera identity with the **same offline reconcile** and produce annotated
videos where one person carries one reid id/colour in every camera.

> **How it works internally:** see **[ARCHITECTURE.md](ARCHITECTURE.md)** for the
> full data flow, every component, the concurrency model, and design rationale.

> **How cross-references are written in these docs.** "**section N**" on its own
> always means the numbered section **of the file you are reading** (this README
> has sections 1–12, listed in the [table of contents](#table-of-contents)). A
> reference to a *different* document always names that document first — e.g.
> "[ARCHITECTURE.md → section 6](ARCHITECTURE.md#6-known-limitations-model-not-plumbing)".
> Every reference is a clickable link that jumps straight to the heading. (Older
> revisions used a bare section-sign shorthand, which did not say *which*
> document it meant; there are none left.)

---

## Quickstart

Start to finish (Linux; GPU auto-detected). Details for each step are in the
numbered sections below.

```bash
# 1. clone
git clone https://github.com/seif744/Inference_PersonReid.git && cd Inference_PersonReid

# 2. OpenCV system libs (Linux/WSL/Docker only; skip on macOS/Windows)
#    -> details in section 2 of this README
sudo apt-get update && sudo apt-get install -y \
  libgl1 libglib2.0-0 libsm6 libice6 libxext6 libxrender1

# 3. Python 3.10 env + deps (torch pin is CUDA-enabled; falls back to CPU)
#    -> details in section 2
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 4. fetch the ReID checkpoint (537 MB, gitignored)  -> details in section 3
curl -L -o src/reid/weights/msmt_sbs_R101-ibn.pth \
  https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R101-ibn.pth

# 5. start the Qdrant gallery (Docker)          -> details in section 4
docker compose up -d

# 6. CHECK THE CONFIG BEFORE RUNNING ANYTHING   -> details in section 3
python tools/preflight.py --load-model

# 7. run it                                     -> details in section 6
#    (a) video files (CPU is fine):
python main.py --videos /path/cam_a.mp4 /path/cam_b.mp4
#    (b) OR live RTSP (GPU; Ctrl-C ONCE to reconcile + write the final videos).
#        Put credentials in .env + source.env_urls, not on the command line:
python main.py --mode live
```

Output: `output_<cam>.mp4` per camera (same person = same reid id/colour
everywhere) + a console run summary. `yolo11m.pt` auto-downloads on first run.

**Do not skip step 6.** Every problem it checks for is *silent* — above all a Qdrant
collection left at a previous backbone's width, which the store only warns about and
`IdentityStage` then swallows, so the run persists nothing, reconcile has nothing to
reconcile, and no error appears anywhere. It exits non-zero if a run would produce
garbage.

---

## How it flows

```mermaid
flowchart TD
    CFG(["config.yaml / CLI"]) --> A1 & B1

    subgraph LIVE[" LIVE — one thread per camera, running at the same time"]
        direction LR
        subgraph CAM_A["Camera A"]
            direction TB
            A1(["Video frame"]) --> A2["Detect + Track\n(YOLO11n + ByteTrack)"]
            A2 --> A3["Embed crop\n(FastReID R101-IBN, 2048-d)"]
            A3 --> A4["Re-rank + verify\ncandidates -> assign\na reid id"]
        end
        subgraph CAM_B["Camera B"]
            direction TB
            B1(["Video frame"]) --> B2["Detect + Track\n(YOLO11n + ByteTrack)"]
            B2 --> B3["Embed crop\n(FastReID R101-IBN, 2048-d)"]
            B3 --> B4["Re-rank + verify\ncandidates -> assign\na reid id"]
        end
    end

    A4 --> QD
    B4 --> QD

    QD[("Qdrant\nown gallery\n(all cameras write here)")]

    QD --> RECON

    subgraph FINAL["OFFLINE — runs ONCE, after every camera finishes"]
        direction TB
        RECON["Reconcile tracklets\nlink the same person\nacross cameras"]
        RECON --> RENDER["Re-render videos\nusing the FINAL\nreid ids"]
        RECON --> SUMMARY["Print run summary\n(console only)"]
    end

    RENDER --> OUT[(" output_camA.mp4\n output_camB.mp4\nsame person = same REID\nin BOTH videos")]

    style CFG fill:#e8eef7,stroke:#4a6fa5,color:#1a1a1a
    style QD fill:#fdf3d7,stroke:#b8860b,color:#1a1a1a,stroke-width:2px
    style OUT fill:#e3f5e6,stroke:#2e7d32,color:#1a1a1a,stroke-width:2px
    style LIVE fill:#f7f9fc,stroke:#4a6fa5
    style FINAL fill:#f2f8f3,stroke:#2e7d32
    style CAM_A fill:#ffffff,stroke:#9fb3cf
    style CAM_B fill:#ffffff,stroke:#9fb3cf
```

Each camera runs the **top** section live, on its own thread, against ONE
shared Qdrant gallery that this pipeline itself writes to and reads from —
no other service is involved. Only after **every** camera finishes does the
**bottom** section run — once: it links the same person across cameras, then
re-renders the annotated videos with those final IDs, so one person carries
the same `REID n` in every camera's output. `global_id` is still written for
compatibility, but `reid_id` is the label the pipeline presents. Full per-stage detail:
**[ARCHITECTURE.md → section 2, "Data flow"](ARCHITECTURE.md#2-data-flow)**.

---

## Table of contents
- [Quickstart](#quickstart)
1. [Prerequisites](#1-prerequisites)
2. [Install the Python environment](#2-install-the-python-environment)
3. [Model weights](#3-model-weights)
4. [Connect to Qdrant](#4-connect-to-qdrant)
5. [Configure your run](#5-configure-your-run)
6. [Run the pipeline](#6-run-the-pipeline)
7. [Understand the output](#7-understand-the-output)
8. [Troubleshooting](#8-troubleshooting)
9. [Project layout](#9-project-layout)
10. [Known limitations & roadmap](#10-known-limitations--roadmap)
11. [Models & credits](#11-models--credits)
12. [License](#12-license)

---

## 1. Prerequisites

- **Python 3.10** (the project is verified on 3.10.12).
- **Docker** + **Docker Compose** — used to run the Qdrant server this
  pipeline writes its own gallery to. Install Docker Desktop (Windows/macOS)
  or Docker Engine (Linux). Verify:
  ```bash
  docker --version
  docker compose version
  ```
- **One or more video files** (e.g. `.avi`, `.mp4`) **or RTSP/HTTP stream URLs**.
- **CPU** is fine for file inputs, development, and the (synthetic) logic tests.
  **Live RTSP in real time needs a GPU** — the live pipeline was developed and
  validated on an **NVIDIA A6000** server (it auto-detects the device). CPU drops
  too many frames to keep up with live streams or to judge live reid quality;
  never benchmark the live path on CPU.

---

## 2. Install the Python environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the exact verified versions: ultralytics, torch,
torchvision, torchreid, qdrant-client, numpy, opencv-python, PyYAML, plus four
packages that look optional but are **not**:

| Pin | Why it must be there |
|---|---|
| `scipy` | torchreid's PyPI wheel declares **no runtime dependencies at all**, and `import torchreid` walks its whole dataset package — one of which does `from scipy.io import loadmat`. |
| `gdown` | Same reason: another torchreid dataset module does a module-level `import gdown`. |
| `tensorboard` | Same reason: torchreid's engine imports `torch.utils.tensorboard`, which imports the `tensorboard` package at module level. |
| `lap` | ByteTrack's data association imports `lap`. Ultralytics ships it only in its optional `[solutions]` extra, so without the pin ultralytics tries to **pip-install it mid-run** — which fails on an offline/locked-down box (e.g. the GPU server). |

Miss any of the first three and the ReID extractor dies with
`ModuleNotFoundError` before it can embed a single crop. Everything those four
need in turn (`tqdm`, `beautifulsoup4`, `filelock`, …) is declared by them and
installs automatically.

> **This list is verified, not inferred.** It was checked by building a venv
> containing *only* `requirements.txt` and running both paths end to end
> (file-batch and `--mode live`) on real footage: detection, pose-ensemble
> splitting, tracking, ReID embedding, the Qdrant gallery, reconcile, and the
> final re-render. `scipy` was found *only* by that run — static import analysis
> had missed it.

**CPU vs GPU torch**, in three parts:

> **1. One install covers both.** The pinned `torch==2.12.1` default Linux wheel
> is **CUDA-enabled**: it uses an NVIDIA GPU when one is present (live pipeline
> `device: auto`) and falls back to CPU otherwise. Nothing needs changing between
> machines — the *same* `pip install -r requirements.txt` is correct on the CPU
> dev box and on the GPU server.

> **2. On the A6000 server.** Same install, no extra step — the server only needs
> a compatible NVIDIA driver for the bundled CUDA runtime to work. Check it with
> `nvidia-smi`, and confirm torch can see the GPU with
> `python -c "import torch; print(torch.cuda.is_available())"`. If that prints
> `False`, the driver is the problem, not the wheel.

> **3. Pinning a specific CUDA version.** Only needed when your driver or cluster
> requires a particular CUDA build. Install `torch`/`torchvision` from the
> matching [PyTorch wheel index](https://pytorch.org/get-started/locally/)
> *before* running `pip install -r requirements.txt`, so the explicit build is
> already satisfied and the requirements file doesn't pull the default wheel over
> it.

### System libraries (Linux / WSL / Docker)

`opencv-python` links a few shared libraries that headless Linux images (and many
Docker bases / WSL) don't ship. If `import cv2` fails with
`ImportError: libGL.so.1: cannot open shared object file` (or a `libgthread` /
Qt `xcb` error), install them once:

```bash
sudo apt-get update && sudo apt-get install -y \
  libgl1 libglib2.0-0 libsm6 libice6 libxext6 libxrender1
```

Not needed on macOS or Windows. (These are also required for the live OpenCV
windows if you ever set `display.show_window: true`; the default is headless.)
Optional: `python3-tk` — only used to size those windows to your screen; without
it the pipeline assumes 1080p and carries on.

### Verify the install

Run the deterministic logic tests — synthetic embeddings, **no GPU/model/video
needed** — to confirm the identity engine + offline-reconcile wiring:

```bash
for t in tests/live/test_*.py; do PYTHONPATH=src:tests/live python "$t"; done
```

Each script prints `OK` on success.

---

## 3. Model weights

Three model files are used (only the ReID checkpoint is committed; both YOLO
weights download themselves):

| File | How to get it | In a fresh clone? |
|---|---|---|
| `yolo11m.pt` (detector) | **Auto-downloaded** by ultralytics on first run | No (gitignored; fetched automatically) |
| `yolo11n-pose.pt` (pose ensemble) | **Auto-downloaded** by ultralytics — but only on a **file-batch** run (the live path disables the pose ensemble, so live runs never fetch it) | No (gitignored; fetched automatically) |
| `src/reid/weights/msmt_sbs_R101-ibn.pth` (ReID, **the default**) | **Fetch per machine** — 537 MB, see below | **No** |
| `src/reid/weights/osnet_ain_x1_0.pth` (ReID, previous default) | Committed to the repo | Yes — already present |

The default ReID checkpoint is **FastReID SBS ResNet101-IBN, trained on MSMT17**
(`reid.model: fastreid_sbs_R101_ibn`, `reid.weights:
src/reid/weights/msmt_sbs_R101-ibn.pth`). It is **not in git** and must be fetched
on every machine:

```bash
curl -L -o src/reid/weights/msmt_sbs_R101-ibn.pth \
  https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R101-ibn.pth
```

At **537 MB** it is over GitHub's 100 MB per-file hard limit, so a push carrying it
is *rejected*, not merely slow. Move it with `deploy.sh` or `scp` if a box has no
outbound network.

> **Two things that catch people out**, both read off upstream source:
> 1. **Input is 384×128, not 256×128** — `Base-SBS.yml` overrides `INPUT.SIZE_TEST`.
>    Feeding 256×128 runs fine and quietly costs accuracy.
> 2. **There is no feature tap to choose.** FastReID's `EmbeddingHead` returns the
>    post-bnneck feature unconditionally at eval, so `reid.tap: n/a` is required and
>    the backend raises on anything else.
>
> Also note `backends.DEFAULT_BACKEND` is `osnet_ain_x1_0`. That is only the
> fallback used when `reid.model` is absent from the config — it is **not** what
> ships. `config.yaml` selects FastReID.

The embedding is **2048-d**, so a Qdrant collection holding the previous 512-d
vectors **must be rebuilt**; the store's dimension guard reports the mismatch at
startup. Every threshold in `config.yaml` was derived in the old 512-d space and
none has been re-anchored — see [ARCHITECTURE.md → section 6, "Known
limitations"](ARCHITECTURE.md#6-known-limitations-model-not-plumbing).

To compare backbones, point `reid.model` at any registered backend
(`fastreid_sbs_R101_ibn` | `fastreid_sbs_R50_ibn` | `osnet_ain_x1_0` |
`osnet_ibn_x1_0` | `osnet_x1_0`) with a matching `reid.weights`, and rank them with
threshold-free metrics — a cosine bar means nothing across feature spaces:

```bash
python tests/calibration/compare_backbones.py register_file.avi 60 6
```

---

## 4. Connect to Qdrant

The pipeline needs somewhere to store its own gallery of embeddings + global
ids. It both writes to and reads from this store — nothing else does.

### 4a. Start the Qdrant server (recommended)

A `docker-compose.yml` is included. From the project root:

```bash
docker compose up -d          # start Qdrant in the background
docker compose logs -f        # (optional) watch logs; Ctrl-C to stop watching
```

This launches `qdrant/qdrant:latest` and exposes:
- **REST API + dashboard:** http://localhost:6333/dashboard
- **gRPC (optional):** `localhost:6334`

Data persists in `./qdrant_storage/` (gitignored), so it survives restarts.
If you already run a Qdrant server elsewhere, point the pipeline at that endpoint
instead (see 4b).

Verify it's up:
```bash
curl http://localhost:6333/readyz        # -> "all shards are ready"
```

Stop it later with `docker compose down` (your data is kept).

### 4b. Tell the pipeline where Qdrant is

Backend precedence: **`QDRANT_URL` env var → `store.url` in config.yaml →
`store.path`** (a local embedded folder, single-process only).

Out of the box, `config.yaml` already has:
```yaml
store:
  enabled: true
  url: http://localhost:6333
```
so no extra step is needed for local Docker. If you prefer env-based config,
create a `.env` file in the project root:
```
QDRANT_URL=http://localhost:6333
```

For **Qdrant Cloud** instead of local Docker, set both (the API key is a secret —
`.env` is gitignored, never commit it):
```
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-key>
```

### 4c. Embedded mode (no Docker, single process only)

```yaml
store:
  enabled: true
  url:                 # leave blank
  path: qdrant_data    # local folder, created automatically
```
This locks the folder to one process — fine for a quick local test, not for
multiple concurrent runs.

---

## 5. Configure your run

> **You must provide your own video files.** Sample footage is **not** shipped in
> the repo (videos are gitignored). Either drop your files in and point
> `source.videos` in `config.yaml` at them, or pass them on the command line with
> `--videos` / `--videos-dir` (see section 6). The paths in the shipped
> `config.yaml` (`placeholder_video/rtsp__219.avi`, `…224.avi`) are **placeholders
> that do not exist** — a default `python main.py` will fail on them until you
> edit them or override with `--videos`.

Edit [config.yaml](config.yaml). The key sections:

```yaml
source:
  videos:                        # one entry per camera
    - name: cam_219              # shipped paths are PLACEHOLDERS -- replace them
      path: placeholder_video/rtsp__219.avi
    - name: cam_224
      path: placeholder_video/rtsp__224.avi
  max_frames: 0                  # 0 = whole video; N = stop early (quick test)
  resize_width: 0                # 0 = native; e.g. 1280 = faster on CPU

detector:
  model: yolo11n.pt              # auto-downloaded
  confidence_threshold: 0.4

tracker:
  enabled: true
  config: bytetrack.yaml

reid:
  enabled: true
  model: fastreid_sbs_R101_ibn   # which backend in src/reid/backends.py
  weights: src/reid/weights/msmt_sbs_R101-ibn.pth
  tap: n/a                       # FastReID has no selectable tap; see section 3
  device: cpu                    # ReID model only, file-batch path. YOLO
                                 # detection auto-picks the GPU regardless --
                                 # see the note under this snippet.
  interval: 10                   # re-embed a track at most every N frames

identity:
  enabled: true
  threshold: 0.63                # plain-path cosine acceptance (used only
                                  # when verification.enabled is false) --
                                  # STILL calibrated for the OLD osnet_ain_x1_0
                                  # feature space; not yet re-anchored
  min_score_gap: 0.03
  rerank:                        # ADR-002 Upgrade 1: camera-aware re-ranking
    enabled: true
    k1: 8
    cross_camera_k1_boost: 4
    lambda_: 0.7
  verification:                  # ADR-002 Upgrade 2: scored verification layer
    enabled: true
    accept_threshold: 0.55
    min_prob_gap: 0.05
    log_path: logs/verification_decisions.jsonl
  reconcile:
    enabled: true
    threshold: 0.63               # cross-camera merge bar -- independent of the
                                   # verifier above; recalibrate both together
    same_camera_threshold: 0.90
    min_tracklet_observations: 3
    require_reciprocal_best: true

display:
  show_window: false             # true = live OpenCV windows; false = headless
  save_annotated: true           # write output_<camera>.mp4

live:                            # real-time streaming pipeline (--mode live)
  run:
    mode: auto                   # live | auto (live for stream URLs) | batch
    device: auto                 # auto picks GPU when present, else CPU
  reconcile:
    enabled: true                # on stop, run the offline reconcile + re-render
    sample_stride: 1             # store every Nth fresh embedding per track
```

See [ARCHITECTURE.md → section 7, "Key
configuration"](ARCHITECTURE.md#7-key-configuration-configyaml) for what every
knob does.

> **`reid.device` does not pin the whole pipeline to one device.** It sets the
> device for **one** thing: the ReID extractor on the **file-batch** path. Note
> also that `auto` is not a torch device string — it means "let the extractor pick".
> Specifically:
>
> | Component | Device it uses |
> |---|---|
> | YOLO detection + pose (both paths) | **whatever Ultralytics picks — CUDA automatically if a GPU is visible.** `src/detector.py` never passes a `device=`, so this ignores `reid.device` entirely. |
> | ReID extractor, file-batch path | `reid.device` |
> | ReID extractor, live path (`--mode live`) | `live.run.device` (`auto` → GPU if present). `reid.device` is **not** consulted here. |

> `python tools/preflight.py` prints both keys side by side, so "I set it to cuda
> and it still ran on CPU" is answerable in one command.
>
> So on a GPU box with the shipped config, a file run detects on the GPU and
> embeds on the CPU. To force everything onto the CPU, set `reid.device: cpu`
> **and** `live.run.device: cpu`, and hide the GPU from the process
> (`CUDA_VISIBLE_DEVICES=""`) — that last part is what actually stops Ultralytics
> from grabbing it.

---

## 6. Run the pipeline

With the venv active and Qdrant reachable:

```bash
python main.py                       # uses the videos listed in config.yaml
```

### Run on your OWN videos (dynamic input — no config edits)

Point the pipeline at any videos from the command line; this **overrides**
`config.yaml`, so you never have to hardcode paths:

```bash
# one or more explicit files (camera name = the file's base name):
python main.py --videos /path/cam_a.mp4 /path/cam_b.mp4

# every video in a folder (.mp4/.avi/.mov/.mkv/...):
python main.py --videos-dir /path/to/footage

```

Precedence: `--videos` > `--videos-dir` > `config.yaml source.videos`. Missing
files fail fast with a clear message; duplicate names are made unique
automatically.

### Run on LIVE RTSP streams (`--mode live`)

Pass one URL per camera. The raw feed is **never recorded**; each stream is
processed in real time, and **Ctrl-C** triggers the offline reconcile that
settles correct cross-camera ids and writes the final `output_<cam>.mp4`:

```bash
python main.py --mode live \
  --videos \
  "rtsp://USER:PASS@192.168.1.224:554/ch01/0" \
  "rtsp://USER:PASS@192.168.1.219:554/ch01/0"
```

- **N cameras = more URLs.** Camera names auto-derive from the last IP octet
  (`cam_224`, `cam_219`, …). URL-encode credential specials (`@` → `%40`).
- On Ctrl-C the console prints *"Please wait a moment while we render the final
  outputs…"*, reconciles, and re-renders — let it finish. Qdrant must be
  reachable (`store.url`); otherwise it logs and falls back to the live output.
- `run.mode: auto` (the default) routes any stream URL to this same live path,
  so `--mode live` is only needed to force it for a file source.

`--reset` wipes the store's collection first, so a run starts from an empty
gallery. **Destructive** — deletes every stored embedding/identity:

```bash
python main.py --reset --videos-dir /path/to/footage
```

Headless (`display.show_window: false`) runs to completion and prints a summary.
With windows enabled, press `q` in a window to stop early.

### Stopping a run — and why not to spam Ctrl-C

**Press Ctrl-C once.** The most important work happens *after* you stop:

```
Ctrl-C  ->  stop the cameras  ->  offline reconcile  ->  re-render output_<cam>.mp4
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  where the correct cross-camera ids are decided
```

That finalize step takes a while and is what produces the deliverable. A second
Ctrl-C used to land in the middle of it, so the run ended with provisional
per-camera ids and half-written videos. Both paths now guard it:

- Every run prints a `[stop]` line at startup telling you this.
- During finalization extra Ctrl-C presses are **ignored** — you get
  `[guard] Ctrl-C ignored -- still finalizing outputs…` on stderr, and the run
  finishes normally with exit code 0.
- **If you truly must abort: press Ctrl-\ (SIGQUIT)**, which the guard never
  blocks. Don't rely on repeated Ctrl-C for this — standard signals don't queue,
  so several presses in quick succession collapse into a single delivery.
  Aborting means the videos keep provisional per-camera ids.

Ctrl-C is still the stop mechanism (with `q` in a window as the alternative when
`display.show_window: true`); it is now just non-destructive to lean on. The
guard lives in [`src/interrupt_guard.py`](src/interrupt_guard.py) and works by
blocking SIGINT for the finalize phase and consuming it in a `sigwait` watcher
thread — a plain Python signal handler is not enough here, because OpenCV/torch
worker threads block SIGINT and a signal that lands on one of them stays pending
and killed the process at teardown (measured: exit `-2` *after* a completed
run).

---

## 7. Understand the output

A run produces:

| Artifact | Location | What it is |
|---|---|---|
| Annotated videos | `output_<camera>.mp4` | source video with boxes + `REID n  IDk` labels drawn |
| Gallery | Qdrant (`store.url` / `store.path`) | every observation + assigned reid id (plus compatibility `global_id`) — this pipeline's own data |
| Verification decision log | `logs/verification_decisions.jsonl` | one line per accept/reject decision, with its full feature vector — for calibrating or eventually training the verifier |

That's it — **no crop images are written.** The embedder makes its own in-memory
crop, so the ReID path never needs files on disk.

The console prints a **RUN SUMMARY** at the end, e.g.:
```
Store: 516 observations -> 11 distinct people (reid_ids)
Cross-camera people: 1
  REID 1: cam_219 (track 0004 + 0025) + cam_224 (track 0001)
```
- **distinct people** = number of reid IDs after reconciliation.
- **Cross-camera people** = reid IDs seen in more than one camera (the product's
  core result).

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: libGL.so.1` (or `libgthread` / Qt `xcb`) on `import cv2` | Missing OpenCV system libs on Linux/WSL/Docker. Install them (see [section 2, "System libraries"](#system-libraries-linux--wsl--docker)). |
| `Connection refused` | Qdrant isn't running. `docker compose up -d`, then `curl http://localhost:6333/readyz`. |
| Live run uses CPU / is very slow on a GPU box | torch can't see the GPU. Check `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`; fix the NVIDIA driver or install a matching CUDA torch ([section 2](#2-install-the-python-environment)). |
| `ModuleNotFoundError: No module named 'gdown'` (or `'tensorboard'`) on `import torchreid` | You installed from an older `requirements.txt`. torchreid declares no runtime dependencies, so these must be pinned explicitly — re-run `pip install -r requirements.txt` (see [section 2](#2-install-the-python-environment)). |
| Run stalls trying to `pip install lap>=0.5.12` when tracking starts (or fails there with no network) | `lap` missing — ultralytics only ships it in an optional extra. Re-run `pip install -r requirements.txt`; the pin is now explicit. |
| Run stopped with Ctrl-C but `output_<cam>.mp4` has per-camera (unreconciled) ids | Ctrl-C was pressed **repeatedly**, force-quitting the finalize step (3 presses). Press it **once** and wait — see [section 6, "Stopping a run"](#stopping-a-run--and-why-not-to-spam-ctrl-c). |
| `Unexpected checkpoint keys dropped` | Wrong/corrupt ReID weights, or `reid.model` and `reid.weights` disagree. Run `python tools/preflight.py --load-model`, then re-fetch `msmt_sbs_R101-ibn.pth` ([section 3](#3-model-weights)). |
| Run completes but `Store: 0 observations` / all ids provisional | The Qdrant collection is at the wrong width for the current backbone. **This is silent** — the store warns, `IdentityStage` swallows the error. `python tools/preflight.py` reports it; `--fix-store` rebuilds it. |
| One person gets many different reid ids | Thresholds are too strict for the running feature space. They were derived under a previous backbone and have not been re-anchored — sweep them offline on a finished run with `tests/calibration/sweep_reconcile_thresholds.py`, no camera time needed. |
| `ValueError` about vector dimension at startup, or observations silently missing | The Qdrant collection still holds 512-d vectors from the previous backbone. Rebuild it ([section 3](#3-model-weights)). |
| Hangs on first run for a while | `yolo11n.pt` is downloading; subsequent runs are fast. |
| Very slow on CPU | Set `source.resize_width: 1280` and/or `source.max_frames` for tests. |
| "database is locked" | Embedded mode (`store.path`) is single-process. Use the shared Docker server (`store.url`), or stop other runs using the same folder. |
| No windows appear | `display.show_window: false` (headless). Set `true` for live windows. |
| Same person shows up as two different reid ids | Expected on out-of-domain footage when cosine similarity is borderline — see limitations below. Check `logs/verification_decisions.jsonl` for that decision's actual scores. |
| Two different people merge into one reid id | The rarer, worse failure. If it happens often, tighten `identity.verification.accept_threshold` / `min_prob_gap` (or `identity.threshold` / `min_score_gap` if verification is disabled). |

---

## 9. Project layout

```
main.py                     entry point + routing + FILE-BATCH orchestration
config.yaml                 all runtime configuration
requirements.txt            pinned dependencies
docker-compose.yml          Qdrant server
ARCHITECTURE.md             deep-dive: data flow, components, design
src/
  video_source.py           frame decoding (files + streams)
  detector.py                YOLO11 + ByteTrack, Detection, crop_person
  drawing.py                  boxes / HUD overlay
  reid/
    extractor.py              crop -> L2-normalized embedding (backend-invariant contract)
    backends.py                the ReIDBackend interface: architecture + preprocessing + tap
    vendor/fastreid/            5 torch-only files from upstream FastReID (see PROVENANCE.md)
    service.py                 TrackEmbedder: throttle, cache, quality + occlusion gates
    weights/                    ReID model checkpoints
  database/
    store.py                   Qdrant wrapper (PersonVectorStore) -- this pipeline's own gallery
  identity/
    service.py                  global-ID assignment (candidate -> re-rank -> verify -> decide -> commit)
    reranking.py                 ADR-002 Upgrade 1: camera-aware k-reciprocal + Jaccard re-ranking
    verifier.py                  ADR-002 Upgrade 2: scored verification layer + decision logging
    reconcile.py                 offline cross-camera reconciliation (used by BOTH paths)
    decision_log.py              every merge decision, accepted and rejected, with all gates
    DESIGN.md                    why the layers are separated
  quiet.py                    silence a C library's stderr for one expected-to-fail call
  geometry/                   IS a merge physically possible? (a CHECK, not a tracker)
    calibration.py             the floor-frame record + the metric-scale guard
    floor.py                    bbox -> point on a shared floor (owns the homography)
    reachability.py             two recorded points -> possible / impossible
    recorder.py                 the LIVE run's writer -- the ONLY place a position is computed
  live/                       REAL-TIME streaming pipeline (--mode live)
    pipeline.py                orchestrator: wire stages, run, shutdown + offline reconcile
    capture.py / decode_backend.py / capabilities.py   per-camera capture + device probe
    scheduler.py / inference.py                         freshness batching + detect/track/embed
    identity_stage.py / identity_engine.py / priority.py   online ids + fair queue + store persistence
    topology.py                 hand-set min-transit veto -- DISABLED, superseded by geometry/
    render.py / writer.py / queues.py / frame.py        draw/encode + load-shedding primitives
tests/live/                  deterministic logic tests (synthetic; no GPU/models needed)
tests/calibration/           measurements on REAL footage (see its README)
tools/
  preflight.py               verify the configured stack BEFORE a run (run this first)
  fit_floor_frame.py         fit the floor frame from people's own foot points
  backfill_geometry.py       give an already-captured run its floor positions
```

Also at the root: `deploy.sh` — rsyncs the code (no venv/videos/store) to a GPU
box over SSH; set `DEPLOY_TARGET` once in a gitignored `.deploy.env`.

Generated at runtime (gitignored): `output_*.mp4`, `logs/`, `qdrant_storage/`,
`qdrant_data/`, `yolo11n.pt`, `yolo11n-pose.pt`.

---

## 10. Known limitations & roadmap

The pipeline is correct; the embedding model is the ceiling on this domain —
see **[ARCHITECTURE.md → section 6, "Known
limitations"](ARCHITECTURE.md#6-known-limitations-model-not-plumbing)** for the
measured same-vs-different score overlap and why no threshold (or hand-tuned
heuristic on top of it) can perfectly resolve it.

The answer being pursued is **not** a better threshold — four threshold changes have
been reverted for hurting accuracy. It is a **physical** guard: `src/geometry/`
refuses any merge that would require a person to move faster than anyone on that
floor has been observed to move, which is a question cosine similarity cannot
answer. It ships disabled; see `ARCHITECTURE.md` → *geometry* and `CLAUDE.md` §1–2
for the rules it must obey (chiefly: the live run records positions, offline
reconcile only consumes them, and nothing claims metres without a trusted metric
reference).

ADR-002's P0 items (camera-aware re-ranking, a scored verification layer) are
implemented. The following are deferred — documented but not built:
- Prototype confidence/variance with adaptive per-identity thresholds.
- A MetaBIN training pass to reduce the underlying domain gap.
- Training a real classifier from `logs/verification_decisions.jsonl` once
  enough runs accumulate, to replace the current hand-set verifier weights.

Built but **off by default**: the camera transition graph
([`src/live/topology.py`](src/live/topology.py), `live.topology.enabled: false`)
vetoes cross-camera matches that would need faster-than-possible transit between
cameras. It was A6000-tested and disabled: with these four cameras the views are
adjacent/overlapping, so the minimum transit time is ~0 and the veto pruned true
matches instead of false ones. Turn it on only for cameras with a real physical
gap between their fields of view.

---

## 11. Models & credits

This pipeline is assembled from pretrained models and open-source libraries — no
model is trained or fine-tuned here. Each component and its role:

| Model / component | Role in the pipeline | Source / library | Reference |
|---|---|---|---|
| **YOLO11n** (`yolo11n.pt`) | Person **detection** every frame (COCO class 0), the entry point for both paths. | [Ultralytics](https://github.com/ultralytics/ultralytics) (COCO-pretrained) | Ultralytics YOLO11 |
| **ByteTrack** (`bytetrack.yaml`) | Multi-object **tracking** — stable per-camera `track_id`s frame-to-frame; the tracklet is the unit of identity evidence. | Ships with Ultralytics | Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022 |
| **YOLO11n-pose** (`yolo11n-pose.pt`) | Optional **pose ensemble** that splits a tracker box merging two overlapping people into one. File-batch only; off in live (`inference.pose_ensemble: false`) for throughput. | [Ultralytics](https://github.com/ultralytics/ultralytics) | Ultralytics YOLO11-pose |
| **FastReID SBS ResNet101-IBN** (`msmt_sbs_R101-ibn.pth`) — **the default** | **Appearance embedding** — one crop → 2048-d L2-normalized vector at 384×128; the cosine similarity that drives re-ranking, verification, and reconciliation. IBN mixes instance and batch norm for cross-domain generalization; trained on MSMT17. Definition vendored into `src/reid/vendor/fastreid/` (5 torch-only files) because FastReID ships no `setup.py` — see its `PROVENANCE.md`. | [JDAI-CV / fast-reid](https://github.com/JDAI-CV/fast-reid) | He et al., *FastReID: A Pytorch Toolbox for General Instance Re-identification*, 2020; Pan et al., *Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net*, ECCV 2018 |
| **OSNet-AIN x1_0** (`osnet_ain_x1_0.pth`) — previous default | Same role, 512-d at 256×128. Retained because every threshold in `config.yaml` was calibrated in this feature space. Multi-source checkpoint (DukeMTMC-reID + Market1501 + CUHK03, eval MSMT17). | [torchreid / deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) | Zhou et al., *Omni-Scale Feature Learning for Person Re-ID*, ICCV 2019; *Learning Generalisable Omni-Scale Representations…*, IEEE TPAMI 2021 |
| **Qdrant** | **Vector store** (not a model) — this pipeline's own gallery: stores every observation embedding + payload and serves nearest-neighbour search. | [Qdrant](https://github.com/qdrant/qdrant) | — |

**Libraries:** the **shipping** backbone is vendored from
[fast-reid](https://github.com/JDAI-CV/fast-reid) — 5 torch-only files under
`src/reid/vendor/fastreid/`, because FastReID ships no `setup.py` and cannot be
pip-installed (see its `PROVENANCE.md`).
[torchreid](https://github.com/KaiyangZhou/deep-person-reid) (Zhou & Xiang,
*Torchreid: A Library for Deep Learning Person Re-ID in Pytorch*, 2019) remains a
dependency and provides the OSNet backbones and feature utilities; PyTorch; OpenCV;
NumPy; PyYAML.

**Dependency licenses:** Ultralytics YOLO11 is **AGPL-3.0**, torchreid is
**MIT**, Qdrant is **Apache-2.0**. Because this project builds on AGPL-3.0 code
(Ultralytics), the project as a whole is distributed under **AGPL-3.0** (see
[section 12, "License"](#12-license)). Swapping to a non-AGPL detector would be
a code change (not just
`detector.model`), since `src/detector.py` is built on the `ultralytics` API.

---

## 12. License

This project is licensed under the **GNU Affero General Public License v3.0
(AGPL-3.0)** — full text in [LICENSE](LICENSE). AGPL applies because the pipeline
combines with **Ultralytics YOLO11 (AGPL-3.0)**; a combined/derivative work must
carry the same license.

```
Copyright (C) 2026 Seifer Mathias

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. This program is distributed WITHOUT ANY WARRANTY; see the GNU
AGPL v3.0 for details.
```

What this means in practice:
- **Open source + public source = compliant**, and **no Ultralytics Enterprise
  license is needed** for this use.
- Anyone who redistributes this (modified or not), **or offers it to users over a
  network** (AGPL-3.0 clause 13, "Remote Network Interaction"), must make the
  corresponding **source available under AGPL-3.0**.
- A **closed-source / proprietary** use would instead require an Ultralytics
  Enterprise license (or removing the AGPL dependency).

> Dependencies keep their own licenses ([section 11](#11-models--credits));
> AGPL-3.0 covers this project's own
> code.
