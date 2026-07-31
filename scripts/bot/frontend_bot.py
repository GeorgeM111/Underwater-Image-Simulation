"""Frontend side of the split bot -- runs on the Grid'5000 FRONTEND (has internet), in tmux.

It is the Telegram end: it turns your messages into command files the node runner consumes,
and it turns the node's result files into Telegram messages. The node stays fully offline; the
two halves only share files under BOT_DIR (NFS).

Env (never committed):
    export TELEGRAM_BOT_TOKEN=...           # from BotFather
    export TELEGRAM_CHAT_ID=8509387068      # only this chat is obeyed
Run it (frontend, project env active, env.sh sourced):
    python scripts/bot/frontend_bot.py
"""

import os
import time
import json
import glob
import threading
import urllib.parse
import urllib.request

from experiments import (REGISTRY, resolve, group_names, CMD_DIR, RES_DIR, STATUS, ensure_dirs)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
API = "https://api.telegram.org/bot%s/%s"
_send_lock = threading.Lock()
_counter = [0]


# ------------------------------------------------------------------- Telegram
def tg(method, **params):
    data = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}).encode()
    with urllib.request.urlopen(urllib.request.Request(API % (TOKEN, method), data=data), timeout=40) as r:
        return json.loads(r.read().decode())


def send(text):
    text = text if len(text) <= 4000 else (text[:2000] + "\n...\n" + text[-1900:])
    with _send_lock:
        try:
            tg("sendMessage", chat_id=CHAT_ID, text=text)
        except Exception as e:
            print("send failed:", e)


# ------------------------------------------------------------------- commands out
def write_command(action, arg=""):
    _counter[0] += 1
    name = "%d_%d.json" % (int(time.time() * 1000), _counter[0])
    tmp = os.path.join(CMD_DIR, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"action": action, "arg": arg}, f)
    os.replace(tmp, os.path.join(CMD_DIR, name))


def read_status():
    try:
        with open(STATUS) as f:
            return json.load(f)
    except Exception:
        return {}


# ------------------------------------------------------------------- results -> messages
def _format(res):
    n = res.get("name", "?")
    st = res.get("status")
    if st == "done":
        return ("✅ DONE  %s\nepochs: %s\n📉 final (train/val):\n%s\n🧪 test metrics:\n%s"
                % (n, res.get("epochs", "?"), res.get("epoch_line", "(none)"), res.get("test", "")))
    if st == "train_failed":
        return "❌ %s — TRAINING FAILED (rc %s)\n%s" % (n, res.get("rc"), res.get("tail", ""))
    if st == "skipped":
        return "⏭ %s — SKIPPED" % n
    return "❌ %s — %s" % (n, res.get("detail", st))


def poll_results():
    for path in sorted(glob.glob(os.path.join(RES_DIR, "*.json"))):
        try:
            with open(path) as f:
                res = json.load(f)
        except Exception:
            continue
        send(_format(res))
        os.replace(path, path + ".sent")                 # don't re-send across restarts


# ------------------------------------------------------------------- Telegram commands
def handle(text):
    parts = text.strip().split()
    if not parts:
        return
    cmd, args = parts[0].lower(), parts[1:]

    if cmd in ("/help", "/start"):
        send("/list  /run kitti|nyu|all  /train <path>  /status  /queue  /skip  /stop\n"
             "e.g. /train Technique_1/NYU/base   or   /train CycleGAN/NYU")
    elif cmd == "/list":
        groups = {}
        for k, v in REGISTRY.items():
            groups.setdefault(v["group"], []).append(k)
        msg = ["Sweep experiments:"]
        for g in ("kitti", "nyu"):
            msg.append("\n[%s]" % g)
            msg += ["  " + k for k in sorted(groups.get(g, []))]
        s = read_status()
        msg.append("\nrunning: %s   queued: %d" % (s.get("current"), len(s.get("queued", []))))
        send("\n".join(msg))
    elif cmd == "/run":
        g = (args[0].lower() if args else "")
        if g not in ("kitti", "nyu", "all"):
            send("usage: /run kitti | /run nyu | /run all"); return
        write_command("run", g)
        send("queued group '%s' (%d experiments)." % (g, len(group_names(g))))
    elif cmd == "/train":
        if not args:
            send("usage: /train Technique_1/NYU/base"); return
        n = resolve(args[0])
        if not n:
            send("could not resolve '%s' to one experiment. Try /list." % args[0]); return
        write_command("train", n)
        send("queued: %s" % n)
    elif cmd == "/status":
        s = read_status()
        send("running: %s\nlatest: %s\nqueued: %d\n%s" % (
            s.get("current"), s.get("epoch_line") or "(waiting)",
            len(s.get("queued", [])), "\n".join(s.get("queued", [])[:20])))
    elif cmd == "/queue":
        s = read_status()
        send("queued (%d):\n%s" % (len(s.get("queued", [])), "\n".join(s.get("queued", [])) or "(empty)"))
    elif cmd == "/skip":
        write_command("skip"); send("skip requested.")
    elif cmd == "/stop":
        write_command("stop"); send("stop requested (finishes current, clears queue).")
    else:
        send("unknown command. /help")


# ------------------------------------------------------------------- main
def main():
    if not TOKEN or not CHAT_ID:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment.")
    ensure_dirs()
    try:
        me = tg("getMe"); assert me.get("ok"), me
    except Exception as e:
        raise SystemExit("Cannot reach Telegram (%s). The FRONTEND has internet; run me there." % e)
    print("frontend bot @%s online" % me["result"]["username"])
    send("🤖 Frontend bot online. Node runner talks to me via %s. Send /help." % os.path.dirname(RES_DIR))

    offset = None
    while True:
        try:
            resp = tg("getUpdates", offset=offset, timeout=10)     # short poll so we also check results
        except Exception as e:
            print("getUpdates error:", e); time.sleep(5); resp = {}
        for upd in resp.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("edited_message") or {}
            if str(msg.get("chat", {}).get("id")) != str(CHAT_ID):
                continue
            text = msg.get("text", "")
            if text:
                try:
                    handle(text)
                except Exception as e:
                    send("command error: %s" % e)
        poll_results()


if __name__ == "__main__":
    main()
