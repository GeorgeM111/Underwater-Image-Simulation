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

    # ---- FULL SCAN -----------------------------------------------------------------
    # Sampling a handful proves nothing: it only takes ONE 16-bit image in 50k to hit the
    # raw-int branch (which skips the /255 normalisation) and to crash under NumPy 2. Scan
    # every row. Image.open only parses the header here, so this is cheap.
    print('\n---------------------------------------------------------')
    print(' scanning ALL %d rows for PIL modes ...' % len(rows))
    img_modes, dep_modes = {}, {}
    offenders = []
    for i, r in enumerate(rows):
        try:
            im = Image.open(BytesIO(data[r[0]])).mode
            dm = Image.open(BytesIO(data[r[1]])).mode
        except KeyError as e:
            print('   row %d: MISSING KEY %r  (CRLF in the CSV? -> use splitlines())' % (i, e.args[0]))
            break
        img_modes[im] = img_modes.get(im, 0) + 1
        dep_modes[dm] = dep_modes.get(dm, 0) + 1
        if im in ('I', 'I;16') or dm in ('I', 'I;16'):
            if len(offenders) < 10:
                offenders.append((i, r[0], im, r[1], dm))

    print('   image modes: %s' % img_modes)
    print('   depth modes: %s' % dep_modes)

    bad = [m for m in list(img_modes) + list(dep_modes) if m in ('I', 'I;16')]
    print('---------------------------------------------------------')
    if bad:
        print(' VERDICT: %s present -> the RAW-INT path in to_tensor is taken for those images.' % sorted(set(bad)))
        print('          That path does NOT divide by 255, so they are not normalised to [0,1],')
        print('          and under NumPy 2 they raise "Unable to avoid copy".')
        print('          First offenders:')
        for i, ip, im, dp, dm in offenders:
            print('            row %-6d image=%-6s %s' % (i, im, ip))
            print('                       depth=%-6s %s' % (dm, dp))
    else:
        print(' VERDICT: every image is RGB and every depth is 8-bit L across all %d rows.' % len(rows))
        print('          -> to_tensor always takes the /255 path; depth IS correctly normalised.')
        print('          -> the raw-int branch is NEVER reached, so the NumPy-2 "copy=False"')
        print('             error did NOT come from to_tensor. Get the real frame with:')
        print('               python data_generation/generate_gt_nyu_subset.py --jobs 1 --chunk-size 4')
        print('             (--jobs 1 stops joblib from swallowing the traceback.)')
    print('---------------------------------------------------------\n')


if __name__ == '__main__':
    main()
