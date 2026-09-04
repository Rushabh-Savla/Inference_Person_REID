from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "rebuild" / "live_reid_final.py",
    ROOT / "rebuild" / "config_live_final.yaml",
]
WEIGHTS = [
    ROOT / "weights" / "yolo11m.pt",
    ROOT / "weights" / "reid" / "resnet50_market1501_aicity156.onnx",
    ROOT / "weights" / "reid" / "nvidia_swin_base_1024" / "export_55" / "swin_base_market1501_aicity156_featuredim1024.onnx",
    ROOT / "weights" / "solider_swin_base_msmt17.onnx",
]


def source_check(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    bad = sorted(x for x in imports if "seif" in x.lower() or "inference_personreid" in x.lower())
    if bad:
        raise RuntimeError(f"forbidden external ReID dependency in {path}: {bad}")
    print(f"[OK] syntax/dependency scan: {path.relative_to(ROOT)}")


def main() -> int:
    print("=" * 72)
    print("FINAL INDEPENDENT LIVE REID PREFLIGHT")
    print("=" * 72)
    for path in FILES:
        source_check(path)
    missing = [str(p.relative_to(ROOT)) for p in WEIGHTS if not p.is_file()]
    if missing:
        print("[FAIL] missing weights:")
        for path in missing:
            print(f"  {path}")
        return 2
    for path in WEIGHTS:
        print(f"[OK] weight: {path.relative_to(ROOT)} ({path.stat().st_size / 1048576:.1f} MiB)")

    if importlib.util.find_spec("onnxruntime") is None:
        print("[FAIL] onnxruntime is not installed in this Python environment")
        return 3
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"[OK] onnxruntime providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print("[FAIL] CUDAExecutionProvider is unavailable")
        return 4
    print("[OK] CUDAExecutionProvider available")
    print("[OK] independent live pipeline is ready for a runtime smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
