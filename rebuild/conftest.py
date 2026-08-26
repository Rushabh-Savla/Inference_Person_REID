from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
REBUILD_PATH = ROOT / "rebuild"

# Pytest can put rebuild/ ahead of src/, allowing rebuild/live.py to shadow
# the real src/live package. Fix both sys.path and any already-imported module.
src_text = str(SRC_PATH)
rebuild_text = str(REBUILD_PATH)
while src_text in sys.path:
    sys.path.remove(src_text)
sys.path.insert(0, src_text)
while rebuild_text in sys.path:
    sys.path.remove(rebuild_text)

live_init = SRC_PATH / "live" / "__init__.py"
if live_init.exists():
    sys.modules.pop("live", None)
    spec = importlib.util.spec_from_file_location(
        "live",
        live_init,
        submodule_search_locations=[str(live_init.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source live package: {live_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["live"] = module
    spec.loader.exec_module(module)
