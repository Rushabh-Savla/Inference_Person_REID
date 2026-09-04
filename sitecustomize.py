from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SRC_TEXT = str(SRC)

while SRC_TEXT in sys.path:
    sys.path.remove(SRC_TEXT)
sys.path.insert(0, SRC_TEXT)
