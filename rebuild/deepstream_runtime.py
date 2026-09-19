from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class DeepStreamRuntime:
    """Discover and validate the installed DeepStream/NvDCF runtime."""

    @staticmethod
    def roots():
        values = []
        env = os.environ.get("NVDCF_DEEPSTREAM_ROOT") or os.environ.get("DEEPSTREAM_ROOT")
        if env:
            values.append(Path(env).expanduser())

        values.extend(
            [
                Path("/opt/nvidia/deepstream/deepstream"),
                Path("/usr/local/nvidia/deepstream/deepstream"),
                Path("/usr/local/deepstream"),
            ]
        )
        values.extend(Path("/opt/nvidia/deepstream").glob("deepstream*"))
        values.extend(Path("/usr/local/nvidia/deepstream").glob("deepstream*"))
        values.extend(Path("/usr/local").glob("deepstream*"))
        values.extend(Path.home().glob("deepstream*"))
        values.extend(Path.home().glob(".local/share/deepstream*"))

        unique = []
        seen = set()
        for root in values:
            try:
                key = str(root.resolve())
            except OSError:
                key = str(root)
            if key not in seen:
                seen.add(key)
                unique.append(Path(key))

        # A non-standard installation may have the library without a standard
        # directory name. Search only common NVIDIA roots to avoid an expensive
        # unrestricted filesystem crawl.
        library = (
            "libnvds_nvmultiobjecttracker.so",
            "libnvds_nvmultiobjecttracker.so.*",
        )
        for base in (
            Path("/opt/nvidia"),
            Path("/usr/local/nvidia"),
            Path("/usr/local"),
            Path.home() / "face_recognition_system",
            Path.home() / "deepstream",
            Path.home() / ".local/share",
            Path(os.environ.get("VIRTUAL_ENV", "")),
        ):
            if not base.exists():
                continue
            for name in library:
                try:
                    for item in base.rglob(name):
                        parent = item.parent
                        if parent.name == "gst-plugins":
                            root = parent.parent.parent
                        elif parent.name == "lib":
                            root = parent.parent
                        else:
                            root = parent
                        unique.append(root)
                except (OSError, PermissionError):
                    continue

        out = []
        seen = set()
        for root in unique:
            try:
                key = str(root.resolve())
            except OSError:
                key = str(root)
            if key in seen:
                continue
            seen.add(key)
            out.append(Path(key))
        return out

    @staticmethod
    def core_library(root):
        if root is None:
            return None
        for path in (
            root / "lib/libnvds_meta.so",
            root / "lib/libnvds_meta.so.*",
            root / "lib64/libnvds_meta.so",
            root / "lib64/libnvds_meta.so.*",
        ):
            if path.is_file():
                return str(path)
        return None

    @classmethod
    def core_root(cls):
        candidates = []
        env = os.environ.get("NVDCF_DEEPSTREAM_ROOT") or os.environ.get("DEEPSTREAM_ROOT")
        if env:
            start = Path(env).expanduser()
            candidates.extend([start, *start.parents])
        candidates.extend([
            Path("/opt/nvidia/deepstream/deepstream"),
            Path("/usr/local/nvidia/deepstream/deepstream"),
            Path("/usr/local/deepstream"),
        ])
        candidates.extend(Path("/opt/nvidia/deepstream").glob("deepstream*"))
        candidates.extend(Path("/usr/local/nvidia/deepstream").glob("deepstream*"))
        candidates.extend(Path("/usr/local").glob("deepstream*"))
        candidates.extend(Path.home().glob("deepstream*"))
        candidates.extend(Path.home().glob(".local/share/deepstream*"))
        seen = set()
        for root in candidates:
            try:
                key = str(root.resolve())
            except OSError:
                key = str(root)
            if key in seen:
                continue
            seen.add(key)
            value = Path(key)
            if cls.core_library(value):
                return value

        for base in (
            Path("/opt/nvidia"),
            Path("/usr/local/nvidia"),
            Path("/usr/local"),
            Path.home() / "face_recognition_system",
            Path.home() / "deepstream",
            Path.home() / ".local/share",
        ):
            if not base.exists():
                continue
            try:
                for item in base.rglob("libnvds_meta.so"):
                    if not item.is_file():
                        continue
                    if item.parent.name == "lib":
                        return item.parent.parent
                    return item.parent
            except (OSError, PermissionError):
                continue
        return None

    @classmethod
    def find(cls):
        tracker = None
        configured = os.environ.get("NVDCF_TRACKER_LIBRARY")
        if configured and Path(configured).expanduser().is_file():
            tracker = str(Path(configured).expanduser().resolve())

        if tracker is None:
            for root in cls.roots():
                value = cls.library(root)
                if value:
                    tracker = value
                    break

        if tracker is None:
            for base in (Path("/opt"), Path("/usr/local"), Path.home()):
                if not base.exists():
                    continue
                for name in (
                    "libnvds_nvmultiobjecttracker.so",
                    "libnvds_nvmultiobjecttracker.so.*",
                ):
                    try:
                        for item in base.rglob(name):
                            if item.is_file():
                                tracker = str(item.resolve())
                                break
                    except (OSError, PermissionError):
                        continue
                    if tracker:
                        break
                if tracker:
                    break

        root = cls.core_root()
        if root is None and tracker is not None:
            root = cls.sdk_root(tracker)
        return root, tracker

    @staticmethod
    def library(root):
        if root is None:
            return None
        patterns = (
            root / "lib/libnvds_nvmultiobjecttracker.so",
            root / "lib/gst-plugins/libnvds_nvmultiobjecttracker.so",
            root / "lib/libnvds_nvmultiobjecttracker.so.*",
            root / "lib/gst-plugins/libnvds_nvmultiobjecttracker.so.*",
        )
        for path in patterns:
            if path.is_file():
                return str(path)
        return None

    @staticmethod
    def config(root):
        if root is None:
            return None
        patterns = (
            root / "samples/configs/deepstream-app/config_tracker_NvDCF_accuracy.yml",
            root / "samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
        )
        for path in patterns:
            if path.is_file():
                return str(path)
        matches = glob.glob(
            str(root / "**/config_tracker_NvDCF_accuracy.yml"),
            recursive=True,
        )
        if matches:
            return matches[0]
        repo_cfg = REPO / "trackers/nvdcf_accuracy.yml"
        if repo_cfg.is_file():
            return str(repo_cfg)
        return None

    @staticmethod
    def library_dir(library):
        if not library:
            return None
        try:
            return Path(library).resolve().parent
        except OSError:
            return Path(library).parent

    @classmethod
    def sdk_root(cls, library):
        if not library:
            return None
        current = cls.library_dir(library)
        if current is None:
            return None
        for candidate in [current] + list(current.parents):
            if (
                (candidate / "sources/includes/nvds_version.h").is_file()
                and (candidate / "sources/includes").is_dir()
            ):
                return candidate
        return None

    @classmethod
    def dependency_lib_dirs(cls, library):
        if not library:
            return []
        try:
            result = subprocess.run(
                ["ldd", str(library)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        found = []
        seen = set()
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            values = [part for part in parts if part.startswith("/")]
            if not values:
                continue
            item = Path(values[0]).resolve()
            if not item.is_file():
                continue
            directory = item.parent
            key = str(directory)
            if key not in seen:
                seen.add(key)
                found.append(directory)
        return found
    @classmethod
    def runtime_lib_dirs(cls):
        names = (
            "libnvds_meta.so",
            "libnvds_infer.so",
            "libnvbufsurface.so",
            "libnvdsgst_meta.so",
            "libnvdsgst_helper.so",
            "libnvdsgst_customhelper.so",
        )
        bases = (
            Path("/opt/nvidia"),
            Path("/usr/local/nvidia"),
            Path("/usr/local"),
            Path.home() / "face_recognition_system",
            Path.home() / "deepstream",
            Path.home() / ".local/share",
        )
        found = []
        seen = set()
        for base in bases:
            if not base.exists():
                continue
            for name in names:
                try:
                    for item in base.rglob(name):
                        if not item.is_file():
                            continue
                        key = str(item.parent.resolve())
                        if key not in seen:
                            seen.add(key)
                            found.append(Path(key))
                except (OSError, PermissionError):
                    continue
        return found
    @classmethod
    def tracker_plugin(cls):
        names = ("libnvdsgst_tracker.so", "libnvdsgst_tracker.so.*")
        bases = (
            Path("/opt/nvidia"),
            Path("/usr/local/nvidia"),
            Path("/usr/local"),
            Path.home() / "deepstream",
            Path.home() / ".local/share",
            Path.home() / "face_recognition_system",
        )
        for base in bases:
            if not base.exists():
                continue
            for name in names:
                try:
                    for item in base.rglob(name):
                        if item.is_file():
                            return item
                except (OSError, PermissionError):
                    continue
        return None

    @classmethod
    def pyds(cls):
        names = ("pyds*.so", "pyds*.whl")
        bases = (
            Path("/opt/nvidia"),
            Path("/usr/local/nvidia"),
            Path("/usr/local"),
            Path.home() / "deepstream",
            Path.home() / ".local/share",
            Path.home() / "face_recognition_system",
        )
        for base in bases:
            if not base.exists():
                continue
            for name in names:
                try:
                    for item in base.rglob(name):
                        if item.is_file():
                            return item
                except (OSError, PermissionError):
                    continue
        return None

    @classmethod
    def version(cls, root, library=None):
        if root is not None:
            header = root / "sources/includes/nvds_version.h"
            values = {}
            if header.is_file():
                text = header.read_text(encoding="utf-8", errors="ignore")
                for key in ("MAJOR", "MINOR", "MICRO"):
                    match = re.search(rf"NVDS_VERSION_{key}\s+([0-9]+)", text)
                    if match:
                        values[key] = match.group(1)
            if len(values) == 3:
                return f'{values["MAJOR"]}.{values["MINOR"]}.{values["MICRO"]}'
            match = re.search(r"deepstream[-_]?([0-9]+(?:\.[0-9]+){1,2})", str(root).lower())
            if match:
                return match.group(1)
        if library:
            paths = [str(library)]
            try:
                result = subprocess.run(
                    ["ldd", str(library)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=15,
                    check=False,
                )
                paths.extend(result.stdout.splitlines())
            except (OSError, subprocess.SubprocessError):
                pass
            for value in paths:
                match = re.search(r"deepstream[-_]?([0-9]+(?:\.[0-9]+){1,2})", value.lower())
                if match:
                    return match.group(1)
            try:
                result = subprocess.run(
                    ["strings", str(library)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=15,
                    check=False,
                )
                match = re.search(r"deepstream[^0-9]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", result.stdout.lower())
                if match:
                    return match.group(1)
            except (OSError, subprocess.SubprocessError):
                pass
        return "unknown"
    @staticmethod
    def command(name):
        try:
            value = subprocess.run(
                name,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return value.stdout.strip()

    @classmethod
    def diagnostics(cls):
        root, library = cls.find()
        result = {
            "root": str(root) if root else None,
            "core_library": cls.core_library(root),
            "version": cls.version(root, library),
            "nvdcf_library": library,
            "nvdcf_config": cls.config(root),
            "gpu": cls.command(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]),
            "cuda": cls.command(["nvcc", "--version"]),
            "tensorrt": cls.command(["trtexec", "--version"]),
        }
        return result

    @staticmethod
    def configure(root, library=None):
        if root is None:
            return
        paths = [
            root / "lib",
            root / "lib/gst-plugins",
        ]
        if library:
            libdir = DeepStreamRuntime.library_dir(library)
            if libdir is not None:
                paths.append(libdir)
        for path in paths:
            if path.is_dir():
                current = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = (
                    str(path) + (os.pathsep + current if current else "")
                )

        plugins = root / "lib/gst-plugins"
        if plugins.is_dir():
            current = os.environ.get("GST_PLUGIN_PATH", "")
            os.environ["GST_PLUGIN_PATH"] = (
                str(plugins) + (os.pathsep + current if current else "")
            )
        for libdir in DeepStreamRuntime.dependency_lib_dirs(library):
            current = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = (
                str(libdir) + (os.pathsep + current if current else "")
            )

        for libdir in DeepStreamRuntime.runtime_lib_dirs():
            current = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = (
                str(libdir) + (os.pathsep + current if current else "")
            )

        plugin = DeepStreamRuntime.tracker_plugin()
        if plugin is not None:
            plugin_dir = plugin.parent
            current = os.environ.get("GST_PLUGIN_PATH", "")
            os.environ["GST_PLUGIN_PATH"] = (
                str(plugin_dir) + (os.pathsep + current if current else "")
            )

        typelibs = root / "lib/girepository-1.0"
        if not typelibs.is_dir():
            typelibs = root / "lib/x86_64-linux-gnu/girepository-1.0"
        if typelibs.is_dir():
            current = os.environ.get("GI_TYPELIB_PATH", "")
            os.environ["GI_TYPELIB_PATH"] = (
                str(typelibs) + (os.pathsep + current if current else "")
            )

    @classmethod
    def require(cls):
        root, library = cls.find()
        if library is None:
            raise RuntimeError(
                "NVIDIA NvDCF tracker library libnvds_nvmultiobjecttracker.so was not found."
            )
        if root is None:
            raise RuntimeError(
                "NvDCF tracker was found, but the DeepStream core runtime was not found."
            )
        core = cls.core_library(root)
        if core is None:
            raise RuntimeError(
                f"DeepStream root {root} does not contain libnvds_meta.so."
            )
        config = cls.config(root)
        if config is None:
            raise RuntimeError(
                f"No NVIDIA NvDCF tracker config was found for DeepStream root {root}."
            )
        cls.configure(root, library)
        try:
            import ctypes
            mode = getattr(os, "RTLD_NOW", 2) | getattr(os, "RTLD_GLOBAL", 256)
            ctypes.CDLL(core, mode=mode)
            ctypes.CDLL(library, mode=mode)
        except OSError as exc:
            raise RuntimeError(
                f"DeepStream/NvDCF native library could not be loaded: {exc}"
            ) from exc
        return root, library, config
