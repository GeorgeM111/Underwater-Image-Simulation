"""Shared experiment registry + NFS file locations for the split bot.

The node is offline, so the two bot halves talk through files under a shared directory
(``$HOME``/``$PROJECT_OUT`` are NFS-shared on Grid'5000):

    <BOT_DIR>/commands/*.json   frontend writes, node consumes   (run/train/stop/skip)
    <BOT_DIR>/results/*.json    node writes, frontend sends       (one per finished experiment)
    <BOT_DIR>/status.json       node writes, frontend reads       (what is running + the queue)
    <BOT_DIR>/logs/*.log        raw train/test stdout

BOT_DIR defaults to <PROJECT_OUT>/bot; override with the BOT_DIR env var if needed.
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from config import CONFIG  # noqa: E402

PY = sys.executable or "python"
BOT_DIR = os.environ.get("BOT_DIR") or os.path.join(os.path.dirname(CONFIG.runs_dir), "bot")
CMD_DIR = os.path.join(BOT_DIR, "commands")
RES_DIR = os.path.join(BOT_DIR, "results")
LOG_DIR = os.path.join(BOT_DIR, "logs")
STATUS = os.path.join(BOT_DIR, "status.json")
EPOCH_RE = re.compile(r"epoch\s+(\d+)", re.IGNORECASE)


def ensure_dirs():
    for d in (BOT_DIR, CMD_DIR, RES_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


# baseline test.py needs --resume <ckpt>; techniques' test.py auto-loads its own.
_CKPT = {
    "Encoder_Decoder_Direct": {"NYU": "Models.ckpt", "Make3D": "Models_Make3D.ckpt", "KITTI": "Models_KITTI.ckpt"},
    "Pix2Pix_GAN":            {"NYU": "Pix2Pix_NYU.ckpt", "Make3D": "Pix2Pix_Make3D.ckpt", "KITTI": "Pix2Pix_KITTI.ckpt"},
    "CycleGAN":               {"NYU": "CycleGAN_NYU.ckpt", "Make3D": "CycleGAN_Make3D.ckpt", "KITTI": "CycleGAN_KITTI.ckpt"},
}


def _ckpt(*parts):
    return os.path.join(CONFIG.checkpoint_dir, *parts)


def make_entry(name):
    """Run entry for ANY experiment dir with train.py+test.py (e.g. 'Technique_1/NYU/base',
    'CycleGAN/NYU'). Techniques auto-load their checkpoint; baselines get --resume. None if no train.py."""
    p = re.sub(r"/train\.py$", "", name.strip().strip("/").replace("\\", "/"))
    if not os.path.isfile(os.path.join(REPO_ROOT, p, "train.py")):
        return None
    entry = {"name": p,
             "train": [PY, p + "/train.py", "--config", "config.yaml"],
             "test":  [PY, p + "/test.py", "--config", "config.yaml"]}
    bits = (p.split("/") + ["", ""])
    fam, ds = bits[0], bits[1]
    if fam in _CKPT and ds in _CKPT[fam]:
        entry["test"] += ["--resume", _ckpt(fam, _CKPT[fam][ds])]
    entry["group"] = "kitti" if "KITTI" in p else ("nyu" if "NYU" in p else "other")
    return entry


def build_registry():
    """Experiments the /run groups sweep: all KITTI (techniques + baselines) + NYU baselines."""
    names = ["Technique_%d/KITTI/%s" % (t, v) for t in (1, 2, 3, 4) for v in ("base", "var1", "var2")]
    for fam in ("Encoder_Decoder_Direct", "Pix2Pix_GAN", "CycleGAN"):
        names += ["%s/KITTI" % fam, "%s/NYU" % fam]
    return {n: e for n in names for e in [make_entry(n)] if e}


REGISTRY = build_registry()


def get_entry(name):
    return REGISTRY.get(name) or make_entry(name)


def group_names(group):
    g = (group or "").lower()
    if g == "all":
        return [k for k in REGISTRY if REGISTRY[k]["group"] == "kitti"] + \
               [k for k in REGISTRY if REGISTRY[k]["group"] == "nyu"]
    return sorted(k for k in REGISTRY if REGISTRY[k]["group"] == g)


def resolve(path):
    """User string -> a runnable experiment name: any dir with a train.py, or a unique substring."""
    p = re.sub(r"/train\.py$", "", path.strip().strip("/").replace("\\", "/"))
    if get_entry(p):
        return p
    hits = [k for k in REGISTRY if p.lower() in k.lower()]
    return hits[0] if len(hits) == 1 else None
