"""
predict_make3d_tech4.py  –  Single-image inference for Technique 4

Given a clean RGB image, produces and saves:
    1. depth_pred.png          — estimated depth map (colourised)
    2. initial_degraded.png    — physics-only haze image (I_Degraded_Initial)
    3. simulated_predicted.png — black-box residue + physics (I^Simulated_Predicted)
    4. direct_predicted.png    — direct end-to-end prediction (I^Direct_Predicted)

Usage:
    python predict_make3d_tech4.py --image path/to/image.jpg --out path/to/output/folder

The user-defined physics parameters (beta, A_light) can be changed in the
USER PARAMETERS section below. Defaults match the training distribution.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2

from model    import PTModel as Model
from model_3D import PTModel as Model3D

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_PATH = r'C:\home\Georges\DenseDepth_3\checkpoint\densenet_multi_task_make3d\Models_Make3D_Tech4.ckpt'

# ── Image size fed to the encoder (must match training) ───────────────────────
FULL_W, FULL_H = 460, 345
HALF_W, HALF_H = 230, 173

# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS — underwater physics
# beta  : attenuation coefficients [R, G, B]
# A     : ambient light            [R, G, B]  in [0, 1]
# Adjust these to change the water type / visibility conditions.
# ─────────────────────────────────────────────────────────────────────────────
DEPTH_SCALE = 1000.0 / 80.0  # = 12.5  (matches data_make3d.py depth_to_tensor)

BETA = [0.02238375 / DEPTH_SCALE,   # R: 0.00179
        0.052971   / DEPTH_SCALE,   # G: 0.00424
        0.43058    / DEPTH_SCALE]   # B: 0.03445

A = [0.0, 0.15, 0.05]   # much more subtle ambient


def load_image(path):
    """Load a BGR image with OpenCV → resize → return float32 [0,1] tensor (1,3,H,W)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    img = cv2.resize(img, (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0  # (3, H, W)
    return tensor.unsqueeze(0)   # (1, 3, H, W)


def make_spatial(vec3, h, w, device):
    """Broadcast a [r,g,b] list to a (1,3,H,W) tensor."""
    arr = torch.tensor(vec3, dtype=torch.float32, device=device)
    return arr.view(1, 3, 1, 1).expand(1, 3, h, w)


def save_image(tensor, path):
    """Save a (1,C,H,W) or (1,1,H,W) float tensor in [0,1] as a PNG."""
    arr = tensor.squeeze().cpu().detach().numpy()
    if arr.ndim == 3:          # (C, H, W) → (H, W, C)
        arr = arr.transpose(1, 2, 0)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:                      # (H, W) depth — colourise with plasma
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr_uint8 = (arr_norm * 255).astype(np.uint8)
        arr = cv2.applyColorMap(arr_uint8, cv2.COLORMAP_PLASMA)
    cv2.imwrite(path, arr)
    print(f'  Saved → {path}')


def compute_initial_degraded(depth, image_half, beta_t, a_t, unit_t):
    """
    Koschmieder physics model (Equation 1 + 2 in the paper):
        tx     = exp(-beta * depth_3d)
        I_init = image_half * tx + A * (1 - tx)
    depth     : (1, 1, H, W)
    image_half: (1, 3, H, W)  in [0, 1]
    Returns   : (1, 3, H, W)  in [0, 1]
    """
    depth_3d = depth.expand(-1, 3, -1, -1)           # (1,3,H,W)
    tx       = torch.exp(-beta_t * depth_3d)
    return image_half * tx + a_t * (unit_t - tx)


def compute_simulated(depth, output_bb, image_half, beta_t, a_t, unit_t):
    """I_simulated = I_initial_degraded + output_bb"""
    i_init = compute_initial_degraded(depth, image_half, beta_t, a_t, unit_t)
    return torch.clamp(output_bb + i_init, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description='Technique 4 – single image prediction')
    parser.add_argument('--image',  required=True,        help='Path to input RGB image')
    parser.add_argument('--ckpt',   default=CKPT_PATH,    help='Checkpoint .ckpt file')
    parser.add_argument('--out',    default='predictions', help='Output folder')
    parser.add_argument('--beta',   nargs=3, type=float,
                        default=BETA, metavar=('R','G','B'),
                        help='Attenuation coefficients')
    parser.add_argument('--ambient', nargs=3, type=float,
                        default=A, metavar=('R','G','B'),
                        help='Ambient light [0,1]')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Load models ───────────────────────────────────────────────────────────
    model_1 = Model().to(device)
    model_2 = Model3D().to(device)
    model_3 = Model3D().to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_1.load_state_dict(ckpt['state_dict_1'])
    model_2.load_state_dict(ckpt['state_dict_2'])
    model_3.load_state_dict(ckpt['state_dict_3'])
    print(f'Checkpoint loaded  (epoch {ckpt.get("cur_epoch","?")}  '
          f'best loss {ckpt.get("best_loss", float("nan")):.4f})')

    model_1.eval(); model_2.eval(); model_3.eval()

    # ── Load & prepare input ──────────────────────────────────────────────────
    img_full = load_image(args.image).to(device)           # (1,3,FULL_H,FULL_W)

    # Half-size version for the physics model
    img_half_np = cv2.resize(
        cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB),
        (HALF_W, HALF_H), interpolation=cv2.INTER_LINEAR)
    img_half = torch.from_numpy(
        img_half_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

    h, w = HALF_H, HALF_W
    beta_t = make_spatial(args.beta,    h, w, device)   # (1,3,H,W)
    a_t    = make_spatial(args.ambient, h, w, device)   # (1,3,H,W)
    unit_t = torch.ones(1, 3, h, w, device=device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        depth         = model_1(img_full)                  # (1,1,FULL_H,FULL_W)
        output_bb     = model_2(img_full)                  # (1,3,FULL_H,FULL_W)
        output_direct = model_3(img_full)                  # (1,3,FULL_H,FULL_W)

        # Resize depth to half size for physics model
        depth_half = torch.nn.functional.interpolate(
            depth, size=(h, w), mode='bilinear', align_corners=True)

        # Resize output_bb to half size
        output_bb_half = torch.nn.functional.interpolate(
            output_bb, size=(h, w), mode='bilinear', align_corners=True)
        output_bb_half = torch.clamp(output_bb_half, 0.0, 1.0)

        # Resize direct to half size (for consistent output resolution)
        output_direct_half = torch.nn.functional.interpolate(
            output_direct, size=(h, w), mode='bilinear', align_corners=True)
        output_direct_half = torch.clamp(output_direct_half, 0.0, 1.0)

        # Physics outputs
        initial_degraded = compute_initial_degraded(
            depth_half, img_half, beta_t, a_t, unit_t)
        simulated_pred   = compute_simulated(
            depth_half, output_bb_half, img_half, beta_t, a_t, unit_t)

    # ── Save outputs ──────────────────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(args.image))[0]

    save_image(depth,              os.path.join(args.out, f'{base}_depth_pred.png'))
    save_image(initial_degraded,   os.path.join(args.out, f'{base}_initial_degraded.png'))
    save_image(simulated_pred,     os.path.join(args.out, f'{base}_simulated_predicted.png'))
    save_image(output_direct_half, os.path.join(args.out, f'{base}_direct_predicted.png'))

    # Also save the original image for easy side-by-side comparison
    orig = cv2.resize(cv2.imread(args.image), (HALF_W, HALF_H))
    cv2.imwrite(os.path.join(args.out, f'{base}_original.png'), orig)
    print(f'  Saved → {os.path.join(args.out, base + "_original.png")}')

    print('\nDone.')


if __name__ == '__main__':
    main()
