# --- repo-root path bootstrap ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Dump the depth-head health scalars for every run, straight from the TB event files.

A WHITE depth panel in TensorBoard is ambiguous — it only tells you the depth map is
CONSTANT, not why. There are two completely different causes:

  * benign : the head has not broken symmetry yet. out_depth_mean sits near its init (~4.0)
             and out_depth_spread is small but GROWING. It will develop structure.
  * dead   : the head has collapsed to the floor. out_depth_mean == 1.0 and
             out_depth_frac_at_floor == 1.0, and neither moves. z = 10 m everywhere,
             t = exp(-beta*z) -> 0, the haze image is pure airlight, and no loss can
             recover it.

This prints the numbers so you can tell them apart at a glance.

Run:
    python scripts/tb_depth_report.py                    # all runs under runs_dir
    python scripts/tb_depth_report.py --run T1_NYU_var1  # one run
"""

import argparse
import glob

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise SystemExit("Needs the tensorboard package to READ event files:  pip install tensorboard")

from config import CONFIG

TAGS = ['stats/out_depth_mean', 'stats/out_depth_min', 'stats/out_depth_max',
        'stats/out_depth_spread', 'stats/out_depth_frac_at_floor',
        'stats/residual_rel_energy', 'stats/sharpness_ratio',
        'loss/depth', 'loss/total', 'loss/val_total']


def series(acc, tag):
    try:
        return {e.step: e.value for e in acc.Scalars(tag)}
    except KeyError:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default=None)
    ap.add_argument('--run', default=None, help='e.g. T1_NYU_var1')
    args = ap.parse_args()

    root = args.runs_dir or CONFIG.runs_dir
    pattern = _os.path.join(root, args.run or '*', '*')
    run_dirs = sorted(d for d in glob.glob(pattern) if _os.path.isdir(d))
    if not run_dirs:
        raise SystemExit('No runs found under %s' % pattern)

    for rd in run_dirs:
        acc = EventAccumulator(rd, size_guidance={'scalars': 0})
        acc.Reload()
        data = {t: series(acc, t) for t in TAGS}
        steps = sorted(set().union(*[set(v) for v in data.values()]) or {0})
        if not any(data.values()):
            continue

        name = _os.path.relpath(rd, root)
        print('\n' + '=' * 100)
        print(' %s' % name)
        print('=' * 100)
        hdr = ('%-5s %-9s %-9s %-9s %-9s %-8s %-8s %-8s %-9s'
               % ('ep', 'd_mean', 'd_min', 'd_max', 'd_spread', 'at_flr', 'resid', 'sharp', 'L_depth'))
        print(hdr)
        for s in steps:
            def g(t, fmt='%-9.4f'):
                v = data[t].get(s)
                return (fmt % v) if v is not None else ('%-9s' % '-')
            print('%-5d %s %s %s %s %s %s %s %s'
                  % (s,
                     g('stats/out_depth_mean'), g('stats/out_depth_min'),
                     g('stats/out_depth_max'), g('stats/out_depth_spread'),
                     g('stats/out_depth_frac_at_floor', '%-8.3f'),
                     g('stats/residual_rel_energy', '%-8.3f'),
                     g('stats/sharpness_ratio', '%-8.2f'),
                     g('loss/depth')))

        # ---- verdict -------------------------------------------------------------
        mean = data['stats/out_depth_mean']
        spread = data['stats/out_depth_spread']
        floor = data['stats/out_depth_frac_at_floor']
        if not mean:
            print('  (no depth stats logged — are you on the latest code?)')
            continue
        last = max(mean)
        m, sp, fl = mean.get(last), spread.get(last), floor.get(last)
        print('  ' + '-' * 96)
        if m is not None and m <= 1.05 and (fl is None or fl > 0.9):
            print('  VERDICT: DEAD. out_depth is pinned at the floor (1.0) => z = 10 m everywhere =>')
            print('           t -> 0 => the haze image is pure airlight. No loss can recover it.')
        elif sp is not None and sp < 0.05:
            growing = (len(spread) > 1 and sp > list(spread.values())[0] * 1.5)
            print('  VERDICT: depth map is CONSTANT (spread %.4f) at mean %.2f.' % (sp, m))
            print('           spread is %s => %s'
                  % ('GROWING' if growing else 'NOT growing',
                     'benign, still breaking symmetry' if growing
                     else 'STUCK — the head is not learning spatial structure'))
        else:
            print('  VERDICT: HEALTHY. depth varies (spread %.2f) around mean %.2f.' % (sp or -1, m))
    print()


if __name__ == '__main__':
    main()
