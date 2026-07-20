from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "wfs_merge_native_train"

for path in (PROJECT_ROOT, TRAIN_DIR):
    text = str(path)
    if path.exists() and text not in sys.path:
        sys.path.insert(0, text)
