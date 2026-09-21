from __future__ import annotations

import ctypes
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from rebuild.deepstream_runtime import DeepStreamRuntime
from rebuild.overlap_guard import merge


class NvDCF:
    """NVIDIA DeepStream NvDCF tracker front-end.

    YOLO supplies person detections. The detections are attached to DeepStream
    metadata before nvtracker. nvtracker then runs NVIDIA's NvDCF low-level
    tracker and returns the tracker-owned IDs.
    """

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.model = YOLO(self.cfg["model"])
        self.conf = float(self.cfg.get("conf", 0.55))
        self.iou = float(self.cfg.get("iou", 0.85))
        self.pose = None
        pose = self.cfg.get("pose", {}) or {}
        if bool(pose.get("enabled", True)):
            self.pose = YOLO(pose.get("model", "yolo11n-pose.pt"))
        self.dsroot, self.library, self.tracker = self._runtime()
        self.width = int(self.cfg.get("width", 0))
        self.height = int(self.cfg.get("height", 0))

    def _runtime(self):
        value = str(self.cfg.get("library", "")).strip().lower()
        configured = self.cfg.get("library")
        if value and value != "auto":
            library = self._find(configured, ())
            if library:
                config = self._find(self.cfg.get("config"), ())
                if config is None:
                    raise RuntimeError(
                        f"NvDCF library is configured at {library}, but the tracker config was not found."
                    )
                dsroot, discovered = DeepStreamRuntime.find()
                if dsroot is None:
                    raise RuntimeError(
                        "NvDCF tracker library was found, but the complete DeepStream core runtime "
                        "was not found (libnvds_meta.so is required)."
                    )
                DeepStreamRuntime.configure(dsroot, library)
                ctypes.CDLL(
                    library,
                    mode=getattr(os, "RTLD_NOW", 2) | getattr(os, "RTLD_GLOBAL", 256),
                )
                return dsroot, library, config
        dsroot, library = DeepStreamRuntime.find()
        if dsroot is None or library is None:
            raise RuntimeError(
                "Installed DeepStream with libnvds_nvmultiobjecttracker.so was not found. "
                "The NvDCF backend will not fall back to another tracker."
            )
        config = self._find(self.cfg.get("config"), ())
        if config is None:
            config = DeepStreamRuntime.config(dsroot)
        if config is None:
            raise RuntimeError(
                f"NvDCF library found at {library}, but no NvDCF tracker config exists under {dsroot}."
            )
        DeepStreamRuntime.configure(dsroot)
        ctypes.CDLL(
            library,
            mode=getattr(os, "RTLD_NOW", 2) | getattr(os, "RTLD_GLOBAL", 256),
        )
        return dsroot, library, config

    @staticmethod
    def _find(value, items):
        if value:
            path = Path(str(value)).expanduser()
            if path.is_file():
                return str(path)
        for item in items:
            if Path(item).is_file():
                return item
        return None

    @staticmethod
    def _deps():
        # Keep DeepStream's Python runtime inside the existing ML venv.
        # No root privileges are required: install_nvdcf_runtime.sh extracts
        # the small Debian Python/GStreamer packages into .nvdcf_runtime and
        # this loader adds them to the current process.
        import ctypes
        import glob
        import sys

        base = Path(os.environ.get("NVDCF_RUNTIME", "")).expanduser()
        if not base:
            base = Path(__file__).resolve().parents[1] / ".nvdcf_runtime"
        dsroot, library, _ = DeepStreamRuntime.require()
        ctypes.CDLL(
            library,
            mode=getattr(os, "RTLD_NOW", 2) | getattr(os, "RTLD_GLOBAL", 256),
        )
        roots = [
            base / "usr/lib/python3/dist-packages",
            base / "usr/lib/python3.12/dist-packages",
            base / "usr/lib/x86_64-linux-gnu/girepository-1.0",
            Path("/usr/lib/python3/dist-packages"),
            Path("/usr/lib/python3.12/dist-packages"),
            dsroot / "lib",
            dsroot / "lib/python",
            dsroot / "sources/deepstream_python_apps/bindings",
            dsroot / "sources/deepstream_python_apps/bindings/build",
        ]
        for path in roots:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

        if base.exists():
            typelib = base / "usr/lib/x86_64-linux-gnu/girepository-1.0"
            libs = base / "usr/lib/x86_64-linux-gnu"
            if typelib.exists():
                current = os.environ.get("GI_TYPELIB_PATH", "")
                os.environ["GI_TYPELIB_PATH"] = (
                    str(typelib) + (os.pathsep + current if current else "")
                )
            if libs.exists():
                current = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = (
                    str(libs) + (os.pathsep + current if current else "")
                )

        dslib = dsroot / "lib"
        dsgst = dslib / "gst-plugins"
        for path in (dslib, dsgst):
            if path.exists():
                current = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = (
                    str(path) + (os.pathsep + current if current else "")
                )

        for pattern in (
            str(dsroot / "lib/pyds*.so"),
            str(dsroot / "lib/python/pyds*.so"),
            str(base / "opt/nvidia/deepstream/deepstream/lib/pyds*.so"),
        ):
            for path in glob.glob(pattern):
                parent = str(Path(path).parent)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
        pyds = DeepStreamRuntime.pyds()
        if pyds is not None and pyds.suffix == ".so":
            parent = str(pyds.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)

        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst, GLib
            import pyds
        except Exception as exc:
            raise RuntimeError(
                "NvDCF DeepStream bindings are unavailable. Run "
                "scripts/install_nvdcf_runtime.sh without sudo; it installs "
                "the GI/GStreamer Python packages into the project venv and "
                "builds/uses PyDS from the installed DeepStream SDK."
            ) from exc
        Gst.init(None)
        return Gst, GLib, pyds

    @staticmethod
    def _uri(path: str) -> str:
        value = Path(path).expanduser()
        if not value.is_file():
            raise RuntimeError(f"Input video not found: {path}")
        return value.resolve().as_uri()

    @staticmethod
    def _pad(src, pad, sink):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None:
            return
        name = caps.get_structure(0).get_name()
        if not name.startswith("video/"):
            return
        try:
            if sink.is_linked():
                return
            pad.link(sink)
        except Exception:
            pass

    @staticmethod
    def _box(value):
        x1, y1, x2, y2 = [float(x) for x in value]
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = max(x1 + 1.0, x2)
        y2 = max(y1 + 1.0, y2)
        return x1, y1, x2, y2

    def _posebox(self, frame):
        if self.pose is None:
            return []
        result = self.pose(
            frame,
            classes=[0],
            conf=0.20,
            iou=0.65,
            verbose=False,
        )
        if not result or result[0].boxes is None:
            return []
        out = []
        for box in result[0].boxes:
            out.append(
                (
                    self._box(box.xyxy[0].tolist()),
                    float(box.conf[0]),
                )
            )
        return out

    @staticmethod
    def _iou(left, right):
        ax1, ay1, ax2, ay2 = left
        bx1, by1, bx2, by2 = right
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        ab = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / max(1.0, aa + ab - inter)

    @classmethod
    def _split(cls, dets, poses):
        if not poses:
            return list(dets)

        out = []
        pose_keep = []

        for det in dets:
            inside = []
            for pose, _conf in poses:
                if cls._iou(pose, det) < 0.15:
                    continue
                px1, py1, px2, py2 = pose
                bx1, by1, bx2, by2 = det
                inter = max(
                    0.0,
                    min(px2, bx2) - max(px1, bx1),
                ) * max(
                    0.0,
                    min(py2, by2) - max(py1, by1),
                )
                area = max(
                    1.0,
                    (px2 - px1) * (py2 - py1),
                )
                if inter / area >= 0.60:
                    inside.append(pose)

            keep = []
            for item in inside:
                if all(cls._iou(item, other) < 0.90 for other in keep):
                    keep.append(item)

            if len(keep) >= 2:
                for item in keep:
                    if all(cls._iou(item, other) < 0.90 for other in pose_keep):
                        pose_keep.append(item)
                continue
            out.append(det)

        # Pose is a secondary person detector. It can recover a person missed
        # by the detector or split a single detector crop containing multiple
        # people. Do not run an aggressive NMS here: high-IoU boxes can be two
        # physically distinct, heavily occluded people.
        for pose, pconf in poses:
            if pconf < 0.25:
                continue
            if any(cls._iou(pose, item) >= 0.45 for item in out):
                continue
            if all(cls._iou(pose, item) < 0.90 for item in pose_keep):
                pose_keep.append(pose)

        out.extend(pose_keep)
        return out

    def _detect(self, frame):
        result = self.model(
            frame,
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        dets = []
        if result and result[0].boxes is not None:
            for box in result[0].boxes:
                conf = float(box.conf[0])
                dets.append((
                    *self._box(box.xyxy[0].tolist()),
                    conf,
                ))

        poses = self._posebox(frame) if self.pose is not None else []
        boxes = self._split([x[:4] for x in dets], poses)

        scored = []
        for box in boxes:
            same = [
                float(item[4])
                for item in dets
                if self._iou(box, item[:4]) >= 0.90
            ]
            pose_conf = [
                float(conf)
                for pbox, conf in poses
                if self._iou(box, pbox) >= 0.90
            ]
            scored.append(
                (
                    *box,
                    max(
                        same + pose_conf + [self.conf],
                    ),
                )
            )
        return scored

    def track(self, camera, path, target):
        Gst, GLib, pyds = self._deps()

        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot inspect input video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        cap.release()

        if not self.library:
            raise RuntimeError(
                "NvDCF low-level library not found. Install DeepStream "
                "with libnvds_nvmultiobjecttracker.so."
            )
        if not self.tracker:
            raise RuntimeError(
                "NvDCF accuracy config not found. Expected the NVIDIA "
                "DeepStream config_tracker_NvDCF_accuracy.yml sample."
            )

        # Keep the source at its original resolution for detection/Re-ID,
        # while bounding only NvDCF's internal tracker surface.
        if width > 1920 or height > 1088:
            ratio = min(1920.0 / width, 1088.0 / height)
            twidth = max(32, int(round(width * ratio)) // 32 * 32)
            theight = max(32, int(round(height * ratio)) // 32 * 32)
        else:
            twidth = max(32, (width + 31) // 32 * 32)
            theight = max(32, (height + 31) // 32 * 32)
        twidth = max(32, twidth)
        theight = max(32, theight)

        # OpenCV/FFmpeg owns file decoding. appsrc injects raw BGR frames;
        # one nvvideoconvert uploads them to NVMM as RGBA; nvstreammux feeds
        # the batched surface directly to NvDCF. This deliberately avoids
        # uridecodebin/nvv4l2decoder and also avoids a second native converter
        # link, which was the site of the observed DeepStream segfault.
        pipeline = Gst.Pipeline.new(f"nvdcf_{camera}")
        source = Gst.ElementFactory.make("appsrc", f"source_{camera}")
        preconv = Gst.ElementFactory.make("nvvideoconvert", f"preconv_{camera}")
        prefilter = Gst.ElementFactory.make("capsfilter", f"prefilter_{camera}")
        mux = Gst.ElementFactory.make("nvstreammux", f"mux_{camera}")
        tracker = Gst.ElementFactory.make("nvtracker", f"tracker_{camera}")
        sink = Gst.ElementFactory.make("fakesink", f"sink_{camera}")
        elems = (source, preconv, prefilter, mux, tracker, sink)
        if any(x is None for x in elems):
            raise RuntimeError("DeepStream elements for NvDCF could not be created")

        source.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={width},height={height},"
                f"framerate={max(1, int(round(fps)))}/1"
            ),
        )
        source.set_property("format", Gst.Format.TIME)
        source.set_property("is-live", False)
        source.set_property("block", True)
        source.set_property("do-timestamp", False)
        if source.find_property("max-buffers") is not None:
            source.set_property("max-buffers", 2)
        if source.find_property("max-bytes") is not None:
            source.set_property("max-bytes", 0)

        if preconv.find_property("nvbuf-memory-type") is not None:
            preconv.set_property("nvbuf-memory-type", 0)
        prefilter.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )

        mux.set_property("batch-size", 1)
        mux.set_property("width", width)
        mux.set_property("height", height)
        mux.set_property("batched-push-timeout", int(1_000_000 / fps))
        mux.set_property("live-source", 0)
        mux.set_property("enable-padding", 0)

        tracker.set_property("ll-lib-file", self.library)
        tracker.set_property("ll-config-file", self.tracker)
        tracker.set_property("tracker-width", twidth)
        tracker.set_property("tracker-height", theight)
        tracker.set_property("gpu-id", 0)
        tracker.set_property("display-tracking-id", 0)
        names = [item.name for item in tracker.list_properties()]
        if "enable-past-frame" in names:
            tracker.set_property("enable-past-frame", 1)

        pipeline.add(*elems)

        if not source.link(preconv):
            raise RuntimeError("Could not link NvDCF appsrc to nvvideoconvert")
        if not preconv.link(prefilter):
            raise RuntimeError("Could not link NvDCF converter to NVMM RGBA caps")

        mux_sink = mux.get_request_pad("sink_0")
        prefilter_src = prefilter.get_static_pad("src")
        if mux_sink is None or prefilter_src is None:
            raise RuntimeError("Could not acquire NvDCF nvstreammux sink_0")
        if prefilter_src.link(mux_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Could not link NvDCF NVMM frames to nvstreammux")

        if not mux.link(tracker):
            raise RuntimeError("Could not link NvDCF nvstreammux to tracker")
        if not tracker.link(sink):
            raise RuntimeError("Could not link NvDCF tracker to sink")

        state = {"error": None}
        handle = target.open("w", encoding="utf-8")
        detections_target = Path(target).with_suffix(".detections.jsonl")
        det_handle = detections_target.open("w", encoding="utf-8")
        detection_cache = {}
        detection_lock = threading.Lock()

        def preprobe(_pad, info, _data):
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            if meta is None:
                state["error"] = "NvDCF received a buffer without NvDsBatchMeta"
                return Gst.PadProbeReturn.OK

            frames = meta.frame_meta_list
            while frames is not None:
                try:
                    fm = pyds.NvDsFrameMeta.cast(frames.data)
                except StopIteration:
                    break
                frame_number = int(fm.frame_num) + 1
                with detection_lock:
                    dets = detection_cache.pop(frame_number, None)

                if dets is None:
                    state["error"] = (
                        f"NvDCF detection metadata missing for frame {frame_number}"
                    )
                    try:
                        frames = frames.next
                    except StopIteration:
                        break
                    continue

                cached = []
                for index, (x1, y1, x2, y2, conf) in enumerate(dets):
                    item = {
                        "camera": str(camera),
                        "frame": frame_number,
                        "timestamp": frame_number / fps,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "detection_score": float(conf),
                        "index": int(index),
                    }
                    cached.append(item)
                    det_handle.write(
                        json.dumps(item, separators=(",", ":")) + "\n"
                    )
                    obj = pyds.nvds_acquire_obj_meta_from_pool(meta)
                    if obj is None:
                        raise RuntimeError("NvDCF could not allocate NvDsObjectMeta")
                    obj.unique_component_id = 1
                    obj.class_id = 0
                    obj.confidence = float(conf)
                    obj.rect_params.left = float(x1)
                    obj.rect_params.top = float(y1)
                    obj.rect_params.width = float(x2 - x1)
                    obj.rect_params.height = float(y2 - y1)
                    pyds.nvds_add_obj_meta_to_frame(fm, obj, None)

                detection_cache[frame_number] = cached
                try:
                    frames = frames.next
                except StopIteration:
                    break
            return Gst.PadProbeReturn.OK

        def postprobe(_pad, info, _data):
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            if meta is None:
                return Gst.PadProbeReturn.OK
            frames = meta.frame_meta_list
            while frames is not None:
                try:
                    fm = pyds.NvDsFrameMeta.cast(frames.data)
                except StopIteration:
                    break
                frame = int(fm.frame_num) + 1
                tracked = []
                objs = fm.obj_meta_list
                while objs is not None:
                    try:
                        obj = pyds.NvDsObjectMeta.cast(objs.data)
                    except StopIteration:
                        break
                    if int(obj.class_id) == 0:
                        left = float(obj.rect_params.left)
                        top = float(obj.rect_params.top)
                        tracked.append(
                            {
                                "camera": str(camera),
                                "frame": frame,
                                "timestamp": frame / fps,
                                "track_id": int(obj.object_id),
                                "bbox": [
                                    left,
                                    top,
                                    left + float(obj.rect_params.width),
                                    top + float(obj.rect_params.height),
                                ],
                                "detection_score": float(getattr(obj, "confidence", 0.0)),
                                "tracker_confidence": float(getattr(obj, "tracker_confidence", 0.0)),
                            }
                        )
                    try:
                        objs = objs.next
                    except StopIteration:
                        break

                detections = detection_cache.pop(frame, [])
                merged = merge(tracked, detections, frame, minimum=0.20)
                for item in merged:
                    handle.write(json.dumps(item, separators=(",", ":")) + "\n")

                try:
                    frames = frames.next
                except StopIteration:
                    break
            return Gst.PadProbeReturn.OK

        # Metadata must be attached after nvstreammux has created NvDsBatchMeta
        # but before nvtracker consumes the batch.
        mux.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, preprobe, None
        )
        tracker.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, postprobe, None
        )

        loop = GLib.MainLoop()

        def message(_bus, msg):
            if msg.type == Gst.MessageType.ERROR:
                error, detail = msg.parse_error()
                state["error"] = f"{error}: {detail}"
                loop.quit()
            elif msg.type == Gst.MessageType.EOS:
                loop.quit()

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", message)

        def feed():
            feed_cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            if not feed_cap.isOpened():
                state["error"] = f"OpenCV/FFmpeg could not open input video: {path}"
                try:
                    source.emit("end-of-stream")
                except Exception:
                    pass
                return

            duration = max(1, int(round(Gst.SECOND / fps)))
            index = 0
            try:
                while True:
                    ok, frame = feed_cap.read()
                    if not ok:
                        break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

                    # Detect on the original CPU frame before handing the
                    # buffer to DeepStream. This removes all dependence on
                    # pyds surface mapping and keeps detection/Re-ID coordinates
                    # in the original 2560x1440 source space.
                    dets = self._detect(frame)
                    with detection_lock:
                        detection_cache[index + 1] = dets
                        if len(detection_cache) > 24:
                            oldest = sorted(detection_cache)[:-24]
                            for key in oldest:
                                detection_cache.pop(key, None)

                    rgba = cv2.cvtColor(
                        np.ascontiguousarray(frame),
                        cv2.COLOR_BGR2RGBA,
                    )
                    payload = np.ascontiguousarray(rgba)
                    gst_buffer = Gst.Buffer.new_allocate(
                        None, int(payload.nbytes), None
                    )
                    gst_buffer.fill(0, payload.tobytes())
                    gst_buffer.pts = index * duration
                    gst_buffer.dts = gst_buffer.pts
                    gst_buffer.duration = duration
                    gst_buffer.offset = index
                    gst_buffer.offset_end = index + 1

                    result = source.emit("push-buffer", gst_buffer)
                    if result != Gst.FlowReturn.OK:
                        if state["error"] is None and result not in (
                            Gst.FlowReturn.FLUSHING,
                            Gst.FlowReturn.EOS,
                        ):
                            state["error"] = (
                                f"NvDCF appsrc stopped with flow result {result}"
                            )
                        break
                    index += 1
            except Exception as exc:
                state["error"] = f"NvDCF OpenCV feed failed: {exc}"
            finally:
                feed_cap.release()
                try:
                    source.emit("end-of-stream")
                except Exception:
                    pass

        feeder = threading.Thread(
            target=feed,
            name=f"nvdcf-feed-{camera}",
            daemon=True,
        )
        pipeline.set_state(Gst.State.PLAYING)
        feeder.start()
        try:
            loop.run()
        finally:
            feeder.join(timeout=15.0)
            pipeline.set_state(Gst.State.NULL)
            handle.close()
            det_handle.close()
            if mux_sink is not None:
                try:
                    mux.release_request_pad(mux_sink)
                except Exception:
                    pass
            with detection_lock:
                detection_cache.clear()

        if state["error"]:
            raise RuntimeError(f"NvDCF failed for {camera}: {state['error']}")
        return fps, width, height

