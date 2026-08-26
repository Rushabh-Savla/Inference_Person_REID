from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
while SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)
