# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import os
import sys

from config import CONFIG

directory = CONFIG.nyu_gt_train_dir

file_count = 0

for root, dirs, files in os.walk(directory):
    file_count += len(files)

print(f"Number of files: {file_count}")
print(f"Current indecies: {file_count/2}")
