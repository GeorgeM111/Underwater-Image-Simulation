"""Training-acceleration helpers: perf flags, AMP autocast, and DDP utilities.

Everything here degrades gracefully so existing single-GPU launches
(``python Technique_X/.../train.py``) keep working unchanged:

* :func:`setup_perf`  — enables TF32 + cuDNN autotuning. Numerics-neutral for
  this project (it only affects conv/matmul kernel selection), safe to call
  unconditionally at the top of ``main()``.
* :func:`autocast`    — returns a bf16 autocast context for the *model forward*
  when ``cfg.amp`` is on and CUDA is available, else a null context (the exact
  fp32 path). bf16 needs no GradScaler, so the loop stays simple.
* DDP helpers (:func:`ddp_setup`, :func:`is_main`, :func:`reduce_avg`,
  :func:`ddp_cleanup`) are no-ops unless the process was started by ``torchrun``
  with ``WORLD_SIZE > 1``. This lets the same file run single-process (for a
  cheap smoke test / CPU) or under multi-GPU DDP with no code change.
"""

import contextlib
import os

import torch

try:
    import torch.distributed as dist
    _HAS_DIST = True
except Exception:  # pragma: no cover - torch always ships distributed
    dist = None
    _HAS_DIST = False


# ---------------------------------------------------------------------------
# Perf flags + mixed precision
# ---------------------------------------------------------------------------
def setup_perf(cfg=None):
    """Enable TF32 matmul + cuDNN autotuning (safe, numerics-neutral here)."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True           # inputs are fixed-size -> autotune kernels
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')  # TF32 for fp32 matmuls
    except Exception:
        pass


def amp_enabled(cfg, device):
    return (bool(getattr(cfg, 'amp', True))
            and torch.cuda.is_available()
            and getattr(device, 'type', str(device)) == 'cuda')


def _amp_dtype(cfg):
    name = str(getattr(cfg, 'amp_dtype', 'bfloat16')).lower()
    return torch.float16 if name in ('fp16', 'float16', 'half') else torch.bfloat16


def autocast(cfg, device):
    """Autocast context for the model forward pass.

    Wrap ONLY the DenseDepth model forwards in this — that is where the compute
    (and the H200 bf16 speedup) lives. Keep the physics / SSIM / gradient losses
    in fp32 by casting the model outputs back to ``.float()`` before them, so the
    numerically delicate parts are untouched.
    """
    if not amp_enabled(cfg, device):
        return contextlib.nullcontext()
    return torch.autocast('cuda', dtype=_amp_dtype(cfg))


# ---------------------------------------------------------------------------
# DDP (used by the Layer-3 shared engine; harmless no-ops single-process)
# ---------------------------------------------------------------------------
def ddp_is_active():
    return _HAS_DIST and dist.is_available() and dist.is_initialized()


def ddp_setup():
    """Init the process group from ``torchrun`` env vars.

    Returns ``(local_rank, world_size, device)``. When not launched by torchrun
    (or ``WORLD_SIZE <= 1``) this is a single-process fallback: no process group
    is created and the returned device is the usual cuda:0 / cpu.
    """
    ws = int(os.environ.get('WORLD_SIZE', '1'))
    if _HAS_DIST and ws > 1 and 'RANK' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        return local_rank, dist.get_world_size(), torch.device('cuda', local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return 0, 1, device


def is_main():
    """True on rank 0 (or whenever DDP is not active)."""
    return (not ddp_is_active()) or dist.get_rank() == 0


def barrier():
    if ddp_is_active():
        dist.barrier()


def ddp_cleanup():
    if ddp_is_active():
        dist.barrier()
        dist.destroy_process_group()


def reduce_avg(value_sum, count, device):
    """All-reduce a (weighted sum, count) pair -> global mean (plain float).

    Used to compute the validation loss over the whole val set when each rank
    only saw a shard of it, so the early-stopping decision is identical on every
    rank.
    """
    if not ddp_is_active():
        return value_sum / max(count, 1)
    t = torch.tensor([value_sum, count], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    denom = t[1].item() or 1.0
    return (t[0].item() / denom)
