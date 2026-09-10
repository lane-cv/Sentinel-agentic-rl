from __future__ import annotations

import sys
from pathlib import Path

COLD_START_DIR = Path(__file__).resolve().parent.parent
if str(COLD_START_DIR) not in sys.path:
    sys.path.insert(0, str(COLD_START_DIR))
