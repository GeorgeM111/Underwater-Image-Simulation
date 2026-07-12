"""Loss-scale normalisation + weight regularisation for the learned-weight variants.

Two independent mechanisms, both needed. Neither alone is sufficient.

1. :class:`EMANormalizer` -- SCALE. The loss terms live in wildly different units:
   ``loss_depth`` is an L1 in the RECIPROCAL depth domain [1, 25] plus an unweighted
   ``lambda_grad * gradient_loss``, so it is O(1-10); the image losses are on [0,1]
   images, so they are O(0.1-0.5). A learned weight that minimises ``sum_i w_i L_i``
   picks ``argmin_i L_i``, so ``loss_depth`` is essentially NEVER chosen and its weight
   is driven to ~0. Dividing each term by a running estimate of its own magnitude puts
   every term at ~1.0, so the weights arbitrate on IMPROVEMENT, not on units.

2. :func:`weight_log_barrier` -- DEGENERACY. Scale normalisation alone does NOT fix the
   collapse: ``sum_i w_i L_i`` is still LINEAR in ``w``, so a simplex/box objective still
   has a VERTEX minimiser — normalisation only changes WHICH weight wins the race to 1.
   The barrier (together with the floor in ``models/weight_head.py``) removes the vertex
   as an attainable optimum.

Both DIVERGE from the paper, which specifies no normalisation and no regulariser. That
is intentional: the paper's Eq. 12/13/17/21 are provably degenerate as written. Set
``lambda_weight_reg: 0.0`` and ``weight_floor: 0.0`` to reproduce the paper exactly
(and watch it collapse).
"""

import math

import torch


class EMANormalizer:
    """Divides each loss term by an exponential moving average of its own magnitude.

    The EMA is computed from DETACHED values, so it contributes no gradient of its own —
    it is a per-term adaptive scale, not a learnable parameter. Non-finite observations
    are ignored (they would poison the running estimate forever).

    Usage:
        norm = EMANormalizer(n=3)
        ld = norm(0, loss_depth)
        lp = norm(1, loss_complex)
        lt = norm(2, loss_haze)
        total = w[0]*ld + w[1]*lp + w[2]*lt
    """

    def __init__(self, n, momentum=0.99, eps=1e-8, warmup=True):
        self.avg = [None] * int(n)
        self.m = float(momentum)
        self.eps = float(eps)
        self.warmup = bool(warmup)

    def __call__(self, i, loss):
        v = float(loss.detach())
        if not math.isfinite(v):
            # Do not let a NaN/Inf batch corrupt the running scale.
            scale = self.avg[i] if self.avg[i] is not None else 1.0
            return loss / (scale + self.eps)
        v = abs(v)
        if self.avg[i] is None:
            self.avg[i] = v if self.warmup else 1.0
        else:
            self.avg[i] = self.m * self.avg[i] + (1.0 - self.m) * v
        return loss / (self.avg[i] + self.eps)

    def state_dict(self):
        return {'avg': list(self.avg), 'm': self.m, 'eps': self.eps}

    def load_state_dict(self, state):
        if not state:
            return
        self.avg = list(state.get('avg', self.avg))
        self.m = float(state.get('m', self.m))
        self.eps = float(state.get('eps', self.eps))


def weight_log_barrier(*weights):
    """Log-barrier that blows up as any learned weight approaches 0.

    Returns ``-sum(log(w))``, to be ADDED to the training total:

        total = total + cfg.lambda_weight_reg * weight_log_barrier(w_global, w_bb, ...)

    This is a BARRIER, not an entropy bonus (an entropy term would have to be
    SUBTRACTED, and is a different function). It makes ``w -> 0`` infinitely costly, so
    the one-hot vertex that kills a loss term is no longer an attainable optimum.

    Accepts any number of weight tensors; each may be a vector.
    """
    total = None
    for w in weights:
        if w is None:
            continue
        term = -torch.log(w.clamp(min=1e-8)).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return total
