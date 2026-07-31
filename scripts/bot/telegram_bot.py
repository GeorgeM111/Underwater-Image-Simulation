"""Telegram experiment bot for the underwater-simulation project (Grid'5000).

Runs on your RESERVED NODE (in tmux). It launches trainings sequentially, and after each
one finishes it reports to you on Telegram: the number of epochs, the final training and
validation loss (so you can see overfitting), and the test metrics. You can also start a
specific training by sending a message.

Design notes
------------
* Stdlib only (urllib) -> NO pip install needed on the node.
* Reads the token/chat from the environment (never hard-coded / committed):
      export TELEGRAM_BOT_TOKEN=...        # from BotFather
      export TELEGRAM_CHAT_ID=8509387068  # your chat id (only this chat is obeyed)
* Grid'5000 nodes reach the internet through a proxy. urllib honours the proxy env vars:
      export https_proxy=http://proxy:3128    # (set the value your site documents)
      export http_proxy=$https_proxy
  On start the bot calls getMe; if that fails it prints a clear error (usually the proxy).
* Source scripts/g5k/env.sh BEFORE launching, so the trainings inherit PROJECT_DATA etc.

Commands (only from your chat id)
    /help                list commands
    /list                show the known experiments and the queue
    /run kitti           queue every KITTI experiment (techniques + baselines)
    /run nyu             queue the NYU state-of-the-art baselines
    /run all             queue kitti then nyu
    /train <path>        queue one experiment, e.g. /train Technique_1/NYU/base
                                                  or /train CycleGAN/NYU
    /status              what is running now + the latest epoch line + queue length
    /queue               list what is queued
    /skip                kill the current experiment and move to the next
    /stop                stop after the current experiment and clear the queue
"""

import os
import re
import sys
import json
import time
import queue
import threading
import subprocess
import urllib.parse
import urllib.request

# --- repo root on sys.path (scripts/bot/ -> repo root) ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from config import CONFIG  # noqa: E402

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
PY = sys.executable or "python"
API = "https://api.telegram.org/bot%s/%s"
LOG_DIR = os.path.join(getattr(CONFIG, "runs_dir", os.path.join(REPO_ROOT, "runs")), "..", "bot_logs")
LOG_DIR = os.path.abspath(LOG_DIR)

_send_lock = threading.Lock()
_jobs = queue.Queue()            # queue of experiment names
_state = {"current": None, "epoch_line": "", "queued": [], "stop": False, "proc": None}


# --------------------------------------------------------------------------- Telegram I/O
def tg(method, **params):
    data = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}).encode()
    req = urllib.request.Request(API % (TOKEN, method), data=data)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode())


def send(text):
    # Telegram caps messages at 4096 chars.
    text = text if len(text) <= 4000 else (text[:2000] + "\n...\n" + text[-1900:])
    with _send_lock:
        try:
            tg("sendMessage", chat_id=CHAT_ID, text=text)
        except Exception as e:                       # never let a send crash the bot
            print("send failed:", e)


# --------------------------------------------------------------------------- experiments
def _ckpt(*parts):
    return os.path.join(CONFIG.checkpoint_dir, *parts)


# baseline test.py needs --resume <ckpt>; techniques' test.py auto-loads its own.
_CKPT = {
    "Encoder_Decoder_Direct": {"NYU": "Models.ckpt", "Make3D": "Models_Make3D.ckpt", "KITTI": "Models_KITTI.ckpt"},
    "Pix2Pix_GAN":            {"NYU": "Pix2Pix_NYU.ckpt", "Make3D": "Pix2Pix_Make3D.ckpt", "KITTI": "Pix2Pix_KITTI.ckpt"},
    "CycleGAN":               {"NYU": "CycleGAN_NYU.ckpt", "Make3D": "CycleGAN_Make3D.ckpt", "KITTI": "CycleGAN_KITTI.ckpt"},
}


def make_entry(name):
    """Build a run entry for ANY experiment dir that has train.py + test.py (e.g.
    'Technique_1/NYU/base', 'CycleGAN/NYU', 'Technique_3/Make3D/var2'). Returns None if the
    directory has no train.py."""
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


def resolve(path):
    """User string -> a runnable experiment name: any dir with a train.py, or a unique substring."""
    p = re.sub(r"/train\.py$", "", path.strip().strip("/").replace("\\", "/"))
    if get_entry(p):                                   # valid dir (in the sweep or ad-hoc)
        return p
    hits = [k for k in REGISTRY if p.lower() in k.lower()]
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------- runner
_EPOCH_RE = re.compile(r"epoch\s+(\d+)", re.IGNORECASE)


def _run(argv, log_path, track_epoch=False):
    """Run a subprocess from REPO_ROOT, tee stdout to log_path, return (rc, tail, last_epoch_line)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    last_epoch, tail = "", []
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        _state["proc"] = proc
        for line in proc.stdout:
            lf.write(line); lf.flush()
            tail.append(line.rstrip("\n"))
            tail = tail[-40:]
            if track_epoch and _EPOCH_RE.search(line):
                last_epoch = line.rstrip("\n")
                _state["epoch_line"] = last_epoch
        proc.wait()
    _state["proc"] = None
    return proc.returncode, "\n".join(tail), last_epoch


def run_experiment(name):
    entry = get_entry(name)
    if entry is None:
        send("❌ %s: no train.py found; skipping." % name); return
    _state["current"] = name
    _state["epoch_line"] = ""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tlog = os.path.join(LOG_DIR, name.replace("/", "_") + "_%s_train.log" % stamp)
    elog = os.path.join(LOG_DIR, name.replace("/", "_") + "_%s_test.log" % stamp)

    send("▶️ START  %s\n(logs: %s)" % (name, os.path.dirname(tlog)))
    rc, tail, epoch_line = _run(entry["train"], tlog, track_epoch=True)
    if rc != 0:
        send("❌ %s — TRAINING FAILED (exit %d)\n--- last lines ---\n%s" % (name, rc, tail))
        _state["current"] = None
        return

    n_epochs = "?"
    m = _EPOCH_RE.search(epoch_line)
    if m:
        n_epochs = str(int(m.group(1)) + 1)          # epochs are 0-indexed in the logs

    trc, ttail, _ = _run(entry["test"], elog, track_epoch=False)
    test_block = ttail if trc == 0 else "TEST FAILED (exit %d)\n%s" % (trc, ttail)

    send("✅ DONE  %s\n"
         "epochs: %s\n"
         "📉 final (train/val):\n%s\n"
         "🧪 test metrics:\n%s" % (name, n_epochs, epoch_line or "(no epoch line captured)", test_block))
    _state["current"] = None


def worker():
    while True:
        name = _jobs.get()
        if name is None:
            return
        if _state["stop"]:
            _jobs.task_done(); continue
        try:
            run_experiment(name)
        except Exception as e:
            send("❌ %s crashed the runner: %s" % (name, e))
        finally:
            _jobs.task_done()
            if name in _state["queued"]:
                _state["queued"].remove(name)


# --------------------------------------------------------------------------- commands
def enqueue(names):
    added = []
    for n in names:
        _jobs.put(n); _state["queued"].append(n); added.append(n)
    return added


def handle(text):
    parts = text.strip().split()
    if not parts:
        return
    cmd, args = parts[0].lower(), parts[1:]

    if cmd in ("/help", "/start"):
        send(__doc__.split("Commands")[1])
    elif cmd == "/list":
        groups = {}
        for k, v in REGISTRY.items():
            groups.setdefault(v["group"], []).append(k)
        msg = ["Known experiments:"]
        for g in ("kitti", "nyu"):
            msg.append("\n[%s]" % g)
            msg += ["  " + k for k in sorted(groups.get(g, []))]
        msg.append("\nQueued: %d  Running: %s" % (len(_state["queued"]), _state["current"]))
        send("\n".join(msg))
    elif cmd == "/run":
        g = (args[0].lower() if args else "")
        if g == "all":
            names = [k for k in REGISTRY if REGISTRY[k]["group"] == "kitti"] + \
                    [k for k in REGISTRY if REGISTRY[k]["group"] == "nyu"]
        elif g in ("kitti", "nyu"):
            names = [k for k in REGISTRY if REGISTRY[k]["group"] == g]
        else:
            send("usage: /run kitti | /run nyu | /run all"); return
        _state["stop"] = False
        enqueue(sorted(names))
        send("queued %d experiments (%s). Running sequentially." % (len(names), g))
    elif cmd == "/train":
        if not args:
            send("usage: /train Technique_1/NYU/base   (or a family like CycleGAN/NYU)"); return
        name = resolve(args[0])
        if not name:
            send("could not resolve '%s' to a single experiment. Try /list." % args[0]); return
        _state["stop"] = False
        enqueue([name])
        send("queued: %s" % name)
    elif cmd == "/status":
        send("running: %s\nlatest: %s\nqueued: %d\n%s" % (
            _state["current"], _state["epoch_line"] or "(waiting)", len(_state["queued"]),
            "\n".join(_state["queued"][:20])))
    elif cmd == "/queue":
        send("queued (%d):\n%s" % (len(_state["queued"]), "\n".join(_state["queued"]) or "(empty)"))
    elif cmd == "/skip":
        p = _state["proc"]
        if p:
            p.terminate(); send("skipping current experiment: %s" % _state["current"])
        else:
            send("nothing running to skip")
    elif cmd == "/stop":
        _state["stop"] = True
        while not _jobs.empty():
            try: _jobs.get_nowait(); _jobs.task_done()
            except queue.Empty: break
        _state["queued"].clear()
        send("will stop after the current experiment; queue cleared.")
    else:
        send("unknown command. /help")


# --------------------------------------------------------------------------- main loop
def main():
    if not TOKEN or not CHAT_ID:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment."); sys.exit(2)
    try:
        me = tg("getMe")
        assert me.get("ok"), me
    except Exception as e:
        print("Cannot reach Telegram (%s). On Grid'5000 set https_proxy/http_proxy." % e); sys.exit(2)
    print("bot @%s online; repo=%s" % (me["result"]["username"], REPO_ROOT))

    threading.Thread(target=worker, daemon=True).start()
    send("🤖 Bot online on the node. %d experiments known. Send /help." % len(REGISTRY))

    offset = None
    while True:
        try:
            resp = tg("getUpdates", offset=offset, timeout=25)
        except Exception as e:
            print("getUpdates error:", e); time.sleep(5); continue
        for upd in resp.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("edited_message") or {}
            if str(msg.get("chat", {}).get("id")) != str(CHAT_ID):
                continue                              # only obey the whitelisted chat
            text = msg.get("text", "")
            if text:
                try:
                    handle(text)
                except Exception as e:
                    send("command error: %s" % e)


if __name__ == "__main__":
    main()
