"""Node side of the split bot -- runs on your RESERVED NODE (in tmux), fully OFFLINE.

It executes experiments sequentially and writes a result file per experiment to the shared
NFS directory; the frontend bot (which has internet) turns those into Telegram messages. It
also reads command files the frontend drops, so /run, /train, /stop, /skip work end to end.

Run it (node, project env active, env.sh sourced):
    python scripts/bot/node_runner.py
"""

import os
import re
import glob
import json
import time
import subprocess

from experiments import (REGISTRY, get_entry, resolve, group_names, REPO_ROOT, PY,  # noqa: F401
                         CMD_DIR, RES_DIR, LOG_DIR, STATUS, EPOCH_RE, ensure_dirs)

_state = {"queue": [], "current": None, "epoch_line": "", "stop": False, "skip": False, "proc": None}


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)                                 # atomic for the frontend reader


def write_status():
    _write_json(STATUS, {"current": _state["current"], "epoch_line": _state["epoch_line"],
                         "queued": list(_state["queue"]), "stopped": _state["stop"],
                         "ts": int(time.time())})


def process_commands():
    for path in sorted(glob.glob(os.path.join(CMD_DIR, "*.json"))):
        try:
            with open(path) as f:
                cmd = json.load(f)
        except Exception:
            os.remove(path); continue
        os.remove(path)
        act, arg = cmd.get("action"), cmd.get("arg", "")
        if act == "run":
            _state["stop"] = False
            _state["queue"] += group_names(arg)
        elif act == "train":
            _state["stop"] = False
            n = resolve(arg)
            if n:
                _state["queue"].append(n)
        elif act == "stop":
            _state["stop"] = True
            _state["queue"].clear()
        elif act == "skip":
            _state["skip"] = True
            if _state["proc"]:
                _state["proc"].terminate()
        elif act == "resume":
            _state["stop"] = False


def _run(argv, log_path, track_epoch=False):
    """Run a subprocess from REPO_ROOT, tee stdout to log_path, poll commands while it runs."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    last_epoch, tail, last_poll = "", [], 0.0
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        _state["proc"] = proc
        for line in proc.stdout:
            lf.write(line); lf.flush()
            tail.append(line.rstrip("\n")); tail = tail[-40:]
            if track_epoch and EPOCH_RE.search(line):
                last_epoch = line.rstrip("\n")
                _state["epoch_line"] = last_epoch
                write_status()
            if time.time() - last_poll > 3:              # stay responsive to /skip and /stop
                process_commands(); last_poll = time.time()
        proc.wait()
    _state["proc"] = None
    return proc.returncode, "\n".join(tail), last_epoch


def run_experiment(name):
    entry = get_entry(name)
    if entry is None:
        _write_json(os.path.join(RES_DIR, "%s.json" % name.replace("/", "_")),
                    {"name": name, "status": "error", "detail": "no train.py"})
        return
    _state["current"] = name
    _state["epoch_line"] = ""
    _state["skip"] = False
    write_status()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = name.replace("/", "_")
    tlog = os.path.join(LOG_DIR, "%s_%s_train.log" % (tag, stamp))
    elog = os.path.join(LOG_DIR, "%s_%s_test.log" % (tag, stamp))

    rc, tail, epoch_line = _run(entry["train"], tlog, track_epoch=True)

    if _state["skip"]:
        result = {"name": name, "status": "skipped", "epoch_line": epoch_line}
    elif rc != 0:
        result = {"name": name, "status": "train_failed", "rc": rc, "tail": tail}
    else:
        n_epochs = None
        m = EPOCH_RE.search(epoch_line)
        if m:
            n_epochs = int(m.group(1)) + 1               # logs are 0-indexed
        trc, ttail, _ = _run(entry["test"], elog, track_epoch=False)
        result = {"name": name, "status": "done", "epochs": n_epochs,
                  "epoch_line": epoch_line,
                  "test": ttail if trc == 0 else "TEST FAILED (rc %d)\n%s" % (trc, ttail)}

    _write_json(os.path.join(RES_DIR, "%s_%s.json" % (tag, stamp)), result)
    _state["current"] = None
    _state["epoch_line"] = ""
    write_status()


def main():
    ensure_dirs()
    print("node_runner online; BOT_DIR shared with the frontend. repo=%s" % REPO_ROOT)
    write_status()
    while True:
        process_commands()
        if not _state["stop"] and _state["queue"]:
            run_experiment(_state["queue"].pop(0))
        else:
            write_status()
            time.sleep(3)


if __name__ == "__main__":
    main()
