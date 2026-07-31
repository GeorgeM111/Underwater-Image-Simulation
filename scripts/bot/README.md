# Split experiment bot (offline node + online frontend)

Your reserved node has **no internet**, so the bot is split in two halves that share files over
NFS (`$HOME`/`$PROJECT_OUT`):

- **`node_runner.py`** — runs on the **reserved node** (tmux, offline). Executes experiments
  sequentially on the GPU and, after each one, writes a result file. Reads command files.
- **`frontend_bot.py`** — runs on the **frontend** (tmux, internet). The Telegram end: turns your
  messages into command files, and the node's result files into Telegram messages.

Both import `experiments.py` (the shared registry + the shared `BOT_DIR` under `$PROJECT_OUT/bot`).
Nothing is sent from the node; the frontend never touches the GPU.

```
you ── Telegram ──► frontend_bot ──(BOT_DIR/commands/*.json)──► node_runner ──► GPU
you ◄─ Telegram ──  frontend_bot ◄──(BOT_DIR/results/*.json)──  node_runner ◄── train+test
```

## Security
Token/chat are read from the environment only; nothing secret is committed. If the token leaks,
`/revoke` in BotFather and re-export.

## Launch

### A) Node runner (on the reserved node, project env active)
```bash
oarsub -C gmoussa
cd ~/Underwater-Image-Simulation
source scripts/g5k/env.sh          # sets PROJECT_DATA/PROJECT_OUT + activates the project venv
tmux new -s runner
python scripts/bot/node_runner.py  # Ctrl-b d to detach
```

### B) Frontend bot (on the frontend, which has internet)
```bash
# same repo, same env so BOT_DIR resolves to the same NFS path
cd ~/Underwater-Image-Simulation
source scripts/g5k/env.sh
export TELEGRAM_BOT_TOKEN='<token>' TELEGRAM_CHAT_ID='8509387068'
tmux new -s bot
python scripts/bot/frontend_bot.py # Ctrl-b d to detach
```
`source scripts/g5k/env.sh` on **both** sides is what makes `BOT_DIR` (`$PROJECT_OUT/bot`) the same
directory. If your `PROJECT_OUT` isn't NFS-shared, set the same `export BOT_DIR=$HOME/underwater_bot`
on both sides instead.

## Commands (Telegram)
| command | effect |
|---|---|
| `/help` | list commands |
| `/list` | sweep experiments + what's running/queued |
| `/run kitti` | queue all KITTI (Technique_1..4 × {base,var1,var2} + EncDec/Pix2Pix/CycleGAN) |
| `/run nyu` | queue the NYU state-of-the-art baselines |
| `/run all` | kitti then nyu |
| `/train <path>` | queue one, e.g. `/train Technique_1/NYU/base` or `/train CycleGAN/NYU` (any dir with a train.py) |
| `/status` | running experiment + latest epoch line + queue |
| `/queue` | list the queue |
| `/skip` | kill the current experiment, move to the next |
| `/stop` | stop after the current experiment, clear the queue |

## A completion message
```
✅ DONE  Technique_1/KITTI/base
epochs: 38
📉 final (train/val):
[T1_KITTI_base] epoch 37/49  train=0.0421  val=0.0455  obj=...
🧪 test metrics:
        a1,        a2,        a3,       rel,       rms,    log_10
    0.9159,    0.9721,    0.9858,    0.0942,    0.0505,    0.0366
```
Raw logs live under `$PROJECT_OUT/bot/logs/`.

## Notes
* **Multi-GPU:** the node runner starts **one worker per GPU** (auto-detected from `nvidia-smi -L`)
  and pins each job with `CUDA_VISIBLE_DEVICES`, so on a 4-GPU node **4 experiments run at once**.
  Cap it with `NGPU=2 python scripts/bot/node_runner.py`. `/status` lists what each GPU is running.
  (Heads-up: 4 concurrent NYU jobs each load the ~4 GB NYU zip into RAM — fine on a big node, but
  watch memory; KITTI streams frames so it's lighter.)
* Reservation lasts ~60 h; when it ends the **node runner** stops. Re-launch it (step A) on your next
  reservation — the **frontend bot** can keep running and will report as soon as the runner is back.
* Smoke-test one GAN on KITTI first (`/train CycleGAN/KITTI`) — the GAN/KITTI code hasn't had a real GPU pass.
