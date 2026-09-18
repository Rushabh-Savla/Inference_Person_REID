from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path


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
            Path.home() / "deepstream",
            Path.home() / ".local/share",
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

    @classmethod
    def find(cls):
        for root in cls.roots():
            library = cls.library(root)
            if library:
                return root, library
        return None, None

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
        return None

    @staticmethod
    def version(root):
        if root is None:
            return "unknown"
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
            "version": cls.version(root),
            "nvdcf_library": library,
            "nvdcf_config": cls.config(root),
            "gpu": cls.command(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]),
            "cuda": cls.command(["nvcc", "--version"]),
            "tensorrt": cls.command(["trtexec", "--version"]),
        }
        return result

    @staticmethod
    def configure(root):
        if root is None:
            return
        paths = [
            root / "lib",
            root / "lib/gst-plugins",
        ]
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
        if root is None or library is None:
            raise RuntimeError(
                "Installed DeepStream with libnvds_nvmultiobjecttracker.so was not found. "
                "Set NVDCF_DEEPSTREAM_ROOT to the actual DeepStream root if it is outside "
                "the standard NVIDIA install locations."
            )
        config = cls.config(root)
        if config is None:
            raise RuntimeError(
                f"NvDCF library found at {library}, but no NVIDIA NvDCF tracker config "
                f"was found below {root}."
            )
        cls.configure(root)
        return root, library, config
