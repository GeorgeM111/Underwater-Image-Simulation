# --- repo-root path bootstrap ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Diagnose how the NYU depth PNGs are actually being decoded.

WHY THIS EXISTS. utils/transforms.ToTensor.to_tensor (and its twin in data_2) normalises to
[0, 1] ONLY on the 8-bit path:

    if pic.mode == 'I':      img = np.array(pic, dtype=np.int32)   # RAW ints
    elif pic.mode == 'I;16': img = np.array(pic, dtype=np.int16)   # RAW ints
    else:                    img = ByteTensor(...)                 # -> .div(255)

So if the depth PNGs are 16/32-bit, `depth_half_norm_0_1` is NOT in [0, 1] — it is raw
integers. Everything downstream then saturates against the depth clamp, the depth map goes
effectively CONSTANT, and t = exp(-beta*z) becomes a constant multiplier: the "complex" GT
degenerates into a plain uniform colour filter over the original image.

Run:
    python data_generation/diagnose_depth_png.py            # train split
    python data_generation/diagnose_depth_png.py --test     # official test split
"""

import argparse
from io import BytesIO

import numpy as np
from PIL import Image

from config import CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true', help='inspect nyu2_test.csv instead of train')
    ap.add_argument('-n', type=int, default=6, help='how many samples to inspect')
    args = ap.parse_args()

    csv = 'data/nyu2_test.csv' if args.test else 'data/nyu2_train.csv'
    from data.nyu import loadZipToMem
    data, rows = loadZipToMem(CONFIG.nyu_zip_path, csv=csv)

    print('\n=========================================================')
    print(' NYU depth decode diagnosis  (%s)' % csv)
    print('=========================================================')

    modes = {}
    for i in range(min(args.n, len(rows))):
        img_name, dep_name = rows[i][0], rows[i][1]
        img = Image.open(BytesIO(data[img_name]))
        dep = Image.open(BytesIO(data[dep_name]))
        a = np.asarray(dep)
        modes[dep.mode] = modes.get(dep.mode, 0) + 1

        normalised = (dep.mode not in ('I', 'I;16'))
        print('\n[%d] %s' % (i, dep_name))
        print('    image mode = %-6s size = %s' % (img.mode, img.size))
        print('    depth mode = %-6s size = %s  dtype = %s' % (dep.mode, dep.size, a.dtype))
        print('    depth raw  min = %-8s max = %-8s mean = %.1f' % (a.min(), a.max(), a.mean()))
        print('    -> to_tensor divides by 255 ?  %s' % ('YES (8-bit path)' if normalised
                                                         else 'NO  (raw int path)  <-- BUG'))
        if normalised:
            z = a.astype(np.float64) / 255.0 * 10.0
            print('    -> z metres  min = %.2f  max = %.2f  mean = %.2f   (expect 0..10)'
                  % (z.min(), z.max(), z.mean()))
        else:
            z = a.astype(np.float64) * 10.0     # what the code ACTUALLY does today
            print('    -> z metres  min = %.1f  max = %.1f   <-- NOT 0..10, everything will'
                  % (z.min(), z.max()))
            print('       clamp to the far plane => CONSTANT depth => the "complex" GT becomes')
            print('       a uniform colour multiply, i.e. a plain colour filter.')
            # what it SHOULD be, if these are 16-bit
            for div, label in ((65535.0, '16-bit full scale'), (10000.0, 'millimetres/10'),
                               (1000.0, 'millimetres')):
                zz = a.astype(np.float64) / div * (10.0 if div == 65535.0 else 1.0)
                if div != 65535.0:
                    zz = a.astype(np.float64) / div
                print('       if divisor = %-8s (%-18s) -> z = %.2f .. %.2f m'
                      % (div, label, zz.min(), zz.max()))

    print('\n---------------------------------------------------------')
    print(' depth modes seen: %s' % modes)
    bad = [m for m in modes if m in ('I', 'I;16')]
    if bad:
        print(' VERDICT: depth is %s -> the raw-int path is taken -> depth is NOT normalised' % bad)
        print('          to [0,1]. The GT is being generated from a saturated/constant depth')
        print('          map. Send this output back before regenerating anything.')
    else:
        print(' VERDICT: depth is 8-bit -> the /255 path is taken -> depth IS normalised.')
        print('          The NumPy-2 crash came from the RGB image side, not depth.')
    print('---------------------------------------------------------\n')


if __name__ == '__main__':
    main()
