"""Node side of the split bot -- runs on your RESERVED NODE (in tmux), fully OFFLINE.

Executes experiments across ALL GPUs of the node: one worker per GPU, each job pinned with
CUDA_VISIBLE_DEVICES (so N experiments run concurrently, like scripts/g5k/train_sweep.sh). After
each experiment it writes a result file to the shared NFS dir; the frontend bot turns those into
Telegram messages. It also reads command files the frontend drops (/run, /train, /stop, /skip).

GPU count: NGPU env var if set, else auto-detected from `nvidia-smi -L`, else 1.

Run it (node, project env active, env.sh sourced):
    python scripts/bot/node_runner.py            # uses every GPU
    NGPU=2 python scripts/bot/node_runner.py      # cap at 2
"""

import os
import glob
import json
import time
import threading
import subprocess

from experiments import (REGISTRY, get_entry, resolve, group_names, REPO_ROOT,  # noqa: F401
                         CMD_DIR, RES_DIR, LOG_DIR, STATUS, EPOCH_RE, ensure_dirs)

_lock = threading.Lock()
_queue = []            # experiment names waiting to run
_running = {}          # gpu -> name
_epoch = {}            # gpu -> latest epoch line
_procs = {}            # gpu -> Popen
_stop = {"v": False}


def n_gpus():
    if os.environ.get("NGPU"):
        return max(int(os.environ["NGPU"]), 1)
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        n = sum(1 for ln in out.splitlines() if ln.strip().startswith("GPU"))
        return max(n, 1)
    except Exception:
        return 1


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def write_status():
    with _lock:
        st = {"running": dict(_running), "epoch": dict(_epoch),
              "queued": list(_queue), "stopped": _stop["v"], "ts": int(time.time())}
    _write_json(STATUS, st)


def process_commands():
    for path in sorted(glob.glob(os.path.join(CMD_DIR, "*.json"))):
        try:
            with open(path) as f:
                cmd = json.load(f)
        except Exception:
            os.remove(path); continue
        os.remove(path)
        act, arg = cmd.get("action"), cmd.get("arg", "")
        with _lock:
            if act == "run":
                _stop["v"] = False
                _queue.extend(group_names(arg))
            elif act == "train":
                _stop["v"] = False
                n = resolve(arg)
                if n:
                    _queue.append(n)
            elif act == "stop":
                _stop["v"] = True
                _queue.clear()
            elif act == "resume":
                _stop["v"] = False
            elif act == "skip":
                for p in list(_procs.values()):          # kill everything currently running
                    try:
                        p.terminate()
                    except Exception:
                        pass


def _next():
    with _lock:
        if _stop["v"] or not _queue:
            return None
        return _queue.pop(0)


def _run(argv, log_path, env, gpu, track_epoch=False):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    last_epoch, tail = "", []
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        with _lock:
            _procs[gpu] = proc
        for line in proc.stdout:
            lf.write(line); lf.flush()
            tail.append(line.rstrip("\n")); tail = tail[-40:]
            if track_epoch and EPOCH_RE.search(line):
                last_epoch = line.rstrip("\n")
                with _lock:
                    _epoch[gpu] = last_epoch
        proc.wait()
    with _lock:
        _procs.pop(gpu, None)
    return proc.returncode, "\n".join(tail), last_epoch


def run_experiment(name, gpu):
    entry = get_entry(name)
    tag = name.replace("/", "_")
    if entry is None:
        _write_json(os.path.join(RES_DIR, "%s.json" % tag), {"name": name, "status": "error",
                                                             "detail": "no train.py"})
        return
    with _lock:
        _running[gpu] = name
        _epoch[gpu] = ""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)                # pin this job to its GPU
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tlog = os.path.join(LOG_DIR, "%s_%s_g%d_train.log" % (tag, stamp, gpu))
    elog = os.path.join(LOG_DIR, "%s_%s_g%d_test.log" % (tag, stamp, gpu))

    rc, tail, epoch_line = _run(entry["train"], tlog, env, gpu, track_epoch=True)
    killed = rc is not None and rc < 0                    # terminate() -> negative rc

    if killed:
        result = {"name": name, "status": "skipped", "epoch_line": epoch_line}
    elif rc != 0:
        result = {"name": name, "status": "train_failed", "rc": rc, "tail": tail}
    else:
        n_epochs = None
        m = EPOCH_RE.search(epoch_line)
        if m:
            n_epochs = int(m.group(1)) + 1
        trc, ttail, _ = _run(entry["test"], elog, env, gpu, track_epoch=False)
        result = {"name": name, "status": "done", "epochs": n_epochs, "gpu": gpu,
                  "epoch_line": epoch_line,
                  "test": ttail if trc == 0 else "TEST FAILED (rc %d)\n%s" % (trc, ttail)}

    _write_json(os.path.join(RES_DIR, "%s_%s_g%d.json" % (tag, stamp, gpu)), result)
    with _lock:
        _running.pop(gpu, None)
        _epoch.pop(gpu, None)


def worker(gpu):
    while True:
        name = _next()
        if name is None:
            time.sleep(2); continue
        try:
            run_experiment(name, gpu)
        except Exception as e:
            _write_json(os.path.join(RES_DIR, "err_%d_%d.json" % (gpu, int(time.time()))),
                        {"name": name, "status": "error", "detail": str(e)})
            with _lock:
                _running.pop(gpu, None)


def main():
    ensure_dirs()
    n = n_gpus()
    print("node_runner online: %d GPU worker(s). repo=%s" % (n, REPO_ROOT))
    write_status()
    for g in range(n):
        threading.Thread(target=worker, args=(g,), daemon=True).start()
    while True:
        process_commands()
        write_status()
        time.sleep(3)


if __name__ == "__main__":
    main()
