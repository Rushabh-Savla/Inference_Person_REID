from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
REBUILD = str(ROOT / "rebuild")

# Pytest prepends the test directory/root to sys.path. This repository also
# contains rebuild/live.py, which can shadow the real src/live package.
# Force src/ to the front before any rebuild tests import live.* modules.
while SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)

# Avoid leaving the rebuild directory itself ahead of src/ after pytest's
# collection adjustments.
while REBUILD in sys.path:
    sys.path.remove(REBUILD)
