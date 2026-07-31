# Telegram experiment bot (runs on your reserved Grid'5000 node)

Launches trainings sequentially and, after each finishes, messages you the **number of
epochs**, the **final training and validation loss** (to spot overfitting), and the **test
metrics**. You can also start any specific training by message.

It is **stdlib-only** (no `pip install`) and runs **on the reserved node**, where it launches
the trainings as subprocesses on the GPU — no `oarsub` from the bot, since you are already on
the node.

## Security
The bot token is a secret: it is read from the environment and is **never** stored in the repo.
Only your chat id is obeyed. If the token is ever exposed, get a new one from BotFather
(`/revoke`) and re-export it.

## Setup on the node (once per reservation)
```bash
# 1) connect to your reserved node
oarsub -C gmoussa

# 2) go to the repo and load the project env (sets PROJECT_DATA etc.)
cd <repo>
source scripts/g5k/env.sh

# 3) Grid'5000 nodes reach the internet via a proxy -- export it so the bot can call Telegram.
#    (Use the proxy host/port your site documents; commonly:)
export https_proxy=http://proxy:3128
export http_proxy=$https_proxy

# 4) the bot credentials (do NOT commit these)
export TELEGRAM_BOT_TOKEN='<token from BotFather>'
export TELEGRAM_CHAT_ID='8509387068'

# 5) launch in tmux so it survives your SSH disconnect (reservation lasts ~60 h)
tmux new -s bot
python scripts/bot/telegram_bot.py
#   detach with  Ctrl-b d   ;  reattach later with  tmux attach -t bot
```
On start it calls Telegram `getMe`; if that fails it prints a proxy hint. When the reservation
ends the bot stops with the node — just repeat steps 1–5 on your next reservation.

## Commands (send these to the bot on Telegram)
| command | effect |
|---|---|
| `/help` | list commands |
| `/list` | the experiments the sweep groups know + what is queued |
| `/run kitti` | queue **all** KITTI experiments: Technique_1..4 × {base,var1,var2} + EncDec/Pix2Pix/CycleGAN |
| `/run nyu` | queue the NYU state-of-the-art baselines (EncDec/Pix2Pix/CycleGAN) |
| `/run all` | `kitti` then `nyu` |
| `/train <path>` | queue one experiment, e.g. `/train Technique_1/NYU/base` or `/train CycleGAN/NYU` (any dir with a train.py works) |
| `/status` | what is running, its latest epoch line, and the queue length |
| `/queue` | list the queue |
| `/skip` | kill the current experiment, move to the next |
| `/stop` | stop after the current experiment and clear the queue |

## What a completion message looks like
```
✅ DONE  Technique_1/KITTI/base
epochs: 38
📉 final (train/val):
[T1_KITTI_base] epoch 37/49  train=0.0421  val=0.0455  obj=...
🧪 test metrics:
        a1,        a2,        a3,       rel,       rms,    log_10
    0.9159,    0.9721,    0.9858,    0.0942,    0.0505,    0.0366
```
Full logs are kept under `$PROJECT_OUT/bot_logs/` for inspection.

## Notes
* Experiments run **one at a time** (they each use the GPU). Send `/run kitti` and leave it.
* The GAN baselines have not been run on real KITTI yet — **smoke-test one** (`/train CycleGAN/KITTI`)
  and watch the first epoch before launching the whole sweep.
