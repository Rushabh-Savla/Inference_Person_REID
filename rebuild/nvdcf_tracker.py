from __future__ import annotations

import ctypes
import json
import os
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

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot inspect input video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        cap.release()

        twidth = self.width or width
        theight = self.height or height
        twidth = max(32, (twidth + 31) // 32 * 32)
        theight = max(32, (theight + 31) // 32 * 32)

        # DeepStream's hardware decoder path on this GPU rejects frames
        # larger than 2048x2048. Force software decode for such sources,
        # then continue through the same NvDCF pipeline after upload to NVMM.
        software = max(width, height) > 2048
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

        pipeline = Gst.Pipeline.new(f"nvdcf_{camera}")
        feature = None
        feature_rank = None
        if software:
            feature = Gst.Registry.get().lookup_feature("nvv4l2decoder")
            if feature is not None:
                feature_rank = feature.get_rank()
                feature.set_rank(Gst.Rank.NONE)

        source = Gst.ElementFactory.make("uridecodebin", f"source_{camera}")
        preconv = Gst.ElementFactory.make("nvvideoconvert", f"preconv_{camera}")
        prefilter = Gst.ElementFactory.make("capsfilter", f"prefilter_{camera}")
        mux = Gst.ElementFactory.make("nvstreammux", f"mux_{camera}")
        conv = Gst.ElementFactory.make("nvvideoconvert", f"conv_{camera}")
        filt = Gst.ElementFactory.make("capsfilter", f"caps_{camera}")
        tracker = Gst.ElementFactory.make("nvtracker", f"tracker_{camera}")
        sink = Gst.ElementFactory.make("fakesink", f"sink_{camera}")
        elems = (source, preconv, prefilter, mux, conv, filt, tracker, sink)
        if any(x is None for x in elems):
            raise RuntimeError("DeepStream elements for NvDCF could not be created")

        source.set_property("uri", self._uri(path))
        mux.set_property("batch-size", 1)
        mux.set_property("width", width)
        mux.set_property("height", height)
        mux.set_property("batched-push-timeout", int(1_000_000 / fps))
        mux.set_property("live-source", 0)
        mux.set_property("enable-padding", 0)

        prefilter.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"),
        )
        filt.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )

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
        sinkpad = mux.get_request_pad("sink_0")
        if sinkpad is None:
            raise RuntimeError("NvDCF could not request nvstreammux sink_0")
        source_sink = preconv.get_static_pad("sink")
        if source_sink is None:
            raise RuntimeError("NvDCF could not access decode upload sink")
        source.connect("pad-added", self._pad, source_sink)
        if (
            not preconv.link(prefilter)
            or not prefilter.link(mux)
            or not mux.link(conv)
            or not conv.link(filt)
            or not filt.link(tracker)
            or not tracker.link(sink)
        ):
            raise RuntimeError("Could not link NvDCF DeepStream pipeline")

        state = {"error": None}
        handle = target.open("w", encoding="utf-8")
        detections_target = Path(target).with_suffix(".detections.jsonl")
        det_handle = detections_target.open("w", encoding="utf-8")
        detection_cache = {}

        def preprobe(_pad, info, _data):
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            frames = meta.frame_meta_list
            while frames is not None:
                try:
                    fm = pyds.NvDsFrameMeta.cast(frames.data)
                except StopIteration:
                    break
                surface = pyds.get_nvds_buf_surface(hash(buf), fm.batch_id)
                image = cv2.cvtColor(
                    np.array(surface, copy=True, order="C"),
                    cv2.COLOR_RGBA2BGR,
                )
                dets = self._detect(image)
                frame_number = int(fm.frame_num) + 1
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

                # Never let a single NvDCF box erase multiple detector
                # hypotheses. When one tracker box ambiguously covers two or
                # more person detections, emit one-to-one shadow rows so the
                # feature matcher sees every person independently.
                detections = detection_cache.pop(frame, [])
                merged = merge(tracked, detections, frame, minimum=0.20)
                for item in merged:
                    handle.write(json.dumps(item, separators=(",", ":")) + "\n")

                try:
                    frames = frames.next
                except StopIteration:
                    break
            return Gst.PadProbeReturn.OK

        filt.get_static_pad("src").add_probe(
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

        pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL)
            handle.close()
            det_handle.close()
            if feature is not None and feature_rank is not None:
                feature.set_rank(feature_rank)

        if state["error"]:
            raise RuntimeError(f"NvDCF failed for {camera}: {state['error']}")
        return fps, width, height
''