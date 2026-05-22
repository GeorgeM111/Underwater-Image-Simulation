"""
perform_test_make3d_tech4.py  –  Evaluation script for Technique 4 on Make3D

Matches compute_errors_nyu and add_results_1 exactly from perform_test_1_make_3D.py.
Tensors saved raw (no *255 scaling) since data_make3d normalises to [0,1].
Clamps applied to model outputs so physics model stays in [0,1] — prevents
unbounded output_bb from inflating rel/log10.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from model       import PTModel as Model
from model_3D    import PTModel as Model3D
from loss        import ssim
from data_make3d import getTestingData
from utils       import AverageMeter, colorize, simple_save_images

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = r'C:\home\Georges\DenseDepth_3'
CKPT_PATH  = os.path.join(BASE_DIR, 'checkpoint',
                           'densenet_multi_task_make3d',
                           'Models_Make3D_Tech4.ckpt')
SAVE_DIR   = os.path.join(BASE_DIR, 'Simple_Test_Tech4')
TB_LOG_DIR = os.path.join(BASE_DIR, 'runs', 'Make3D', 'tech4_test')
BATCH_SIZE = 5


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    test_loader = getTestingData(batch_size=BATCH_SIZE)
    LogProgress(test_loader)
    CalcAccuracyOnly(test_loader)


# ── Step 1: inference + save tensors ──────────────────────────────────────────

def LogProgress(test_loader):
    writer = SummaryWriter(TB_LOG_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model_1 = Model().to(device)
    model_2 = Model3D().to(device)
    model_3 = Model3D().to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model_1 = nn.DataParallel(model_1.cuda())
        model_2 = nn.DataParallel(model_2.cuda())
        model_3 = nn.DataParallel(model_3.cuda())

    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model_1.load_state_dict(checkpoint['state_dict_1'])
    model_2.load_state_dict(checkpoint['state_dict_2'])
    model_3.load_state_dict(checkpoint['state_dict_3'])
    print(f'Checkpoint loaded  —  epoch {checkpoint.get("cur_epoch", "?")}  '
          f'best loss: {checkpoint.get("best_loss", float("nan")):.4f}')

    model_1.eval(); model_2.eval(); model_3.eval()

    l1_criterion = nn.L1Loss()
    losses = AverageMeter()
    N = len(test_loader)
    print(f'Test batches: {N}')
    print('-' * 10)

    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):

            image_full           = sample_batched['image_full'].to(device)
            image_half           = sample_batched['image_half'].to(device)
            depth_half           = sample_batched['depth_half'].to(device)
            orig_haze_image      = sample_batched['haze_image'].to(device)
            beta_val_half        = sample_batched['beta'].to(device)
            a_mat_half           = sample_batched['a_val'].to(device)
            unit_mat_half        = sample_batched['unit_mat'].to(device)
            complex_image_tensor = sample_batched['complex_noise_img'].to(device)

            # ── Forward pass ──────────────────────────────────────────────────
            output_depth  = model_1(image_full)
            output_bb     = model_2(image_full)
            output_direct = model_3(image_full)

            # Clamp RGB outputs to [0,1] — data_make3d normalises GT to [0,1]
            # so predictions must be in the same range for valid metrics.
            # Physically: you cannot have negative light or light > max.
            output_bb     = torch.clamp(output_bb,     0.0, 1.0)
            output_direct = torch.clamp(output_direct, 0.0, 1.0)

            pred_complex = compute_complex_image(
                output_depth, output_bb,
                beta_val_half, a_mat_half, unit_mat_half, image_half)
            pred_haze = compute_haze_image(
                output_depth,
                beta_val_half, a_mat_half, unit_mat_half, image_half)

            pred_complex = torch.clamp(pred_complex, 0.0, 1.0)
            pred_haze    = torch.clamp(pred_haze,    0.0, 1.0)

            # ── Losses (logging only) ─────────────────────────────────────────
            l_d_ssim   = torch.clamp(
                (1 - ssim(output_depth, depth_half, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
            loss_depth = 0.1 * l_d_ssim + 0.1 * l1_criterion(output_depth, depth_half)

            l_p_ssim     = torch.clamp(
                (1 - ssim(pred_complex.float(), complex_image_tensor.float(), val_range=1)) * 0.5, 0, 1)
            loss_complex = 1.0 * l_p_ssim + 0.1 * l1_criterion(pred_complex, complex_image_tensor)

            l_t_ssim  = torch.clamp(
                (1 - ssim(pred_haze.float(), orig_haze_image.float(), val_range=1)) * 0.5, 0, 1)
            loss_haze = 1.0 * l_t_ssim + 0.1 * l1_criterion(pred_haze, orig_haze_image)

            l_g_ssim    = torch.clamp(
                (1 - ssim(output_direct.float(), complex_image_tensor.float(), val_range=1)) * 0.5, 0, 1)
            loss_direct = 1.0 * l_g_ssim + 0.1 * l1_criterion(output_direct, complex_image_tensor)

            total_loss = loss_depth + loss_complex + loss_haze + loss_direct
            losses.update(total_loss.item(), image_full.size(0))

            # ── Save raw tensors (no scaling — both GT and pred in [0,1]) ─────
            torch.save(complex_image_tensor,
                       os.path.join(SAVE_DIR, f'Batch_{i}_Complex_GT.pt'))
            torch.save(pred_complex,
                       os.path.join(SAVE_DIR, f'Batch_{i}_Complex_Pred.pt'))

            torch.save(orig_haze_image,
                       os.path.join(SAVE_DIR, f'Batch_{i}_Haze_GT.pt'))
            torch.save(pred_haze,
                       os.path.join(SAVE_DIR, f'Batch_{i}_Haze_Pred.pt'))

            torch.save(complex_image_tensor,
                       os.path.join(SAVE_DIR, f'Batch_{i}_Direct_GT.pt'))
            torch.save(output_direct,
                       os.path.join(SAVE_DIR, f'Batch_{i}_Direct_Pred.pt'))

            if i % 5 == 0:
                writer.add_scalar('Test/Loss', total_loss.item(), i)

            if i == N - 1:
                writer.add_image('Test.1.Image_Half',
                    vutils.make_grid(image_half.data,           nrow=6, normalize=True),  0)
                writer.add_image('Test.2.Depth_GT',
                    vutils.make_grid(depth_half.data,           nrow=6, normalize=True),  0)
                writer.add_image('Test.3.GT_Haze_Image',
                    vutils.make_grid(orig_haze_image.data,      nrow=6, normalize=False), 0)
                writer.add_image('Test.4.GT_Complex_Image',
                    vutils.make_grid(complex_image_tensor.data, nrow=6, normalize=False), 0)
                writer.add_image('Test.5.Pred_Depth_Norm',
                    vutils.make_grid(output_depth.data,         nrow=6, normalize=True),  0)
                writer.add_image('Test.6.Pred_BlackBox',
                    vutils.make_grid(output_bb.data,            nrow=6, normalize=True),  0)
                writer.add_image('Test.7.Pred_Complex_Image',
                    vutils.make_grid(pred_complex.data,         nrow=6, normalize=False), 0)
                writer.add_image('Test.8.Pred_Haze_Image',
                    vutils.make_grid(pred_haze.data,            nrow=6, normalize=False), 0)
                writer.add_image('Test.9.Pred_Direct_Image',
                    vutils.make_grid(output_direct.data,        nrow=6, normalize=False), 0)

            if i % 10 == 0:
                print(f'Batch [{i}/{N-1}]  Loss: {losses.avg:.4f}')

    print(f'\nAverage test loss: {losses.avg:.4f}')
    writer.close()


# ── Step 2: metrics ────────────────────────────────────────────────────────────

def CalcAccuracyOnly(test_loader):
    N = len(test_loader)
    print('\n' + '=' * 72)
    print('Computing accuracy metrics...')
    print('=' * 72)

    outputs = {
        'Complex (L_p)': 'Complex',
        'Haze    (L_t)': 'Haze',
        'Direct  (L_g)': 'Direct',
    }

    all_results = {}
    for label in outputs:
        all_results[label] = dict(
            a1=0.0, a2=0.0, a3=0.0, abs_rel=0.0, rmse=0.0, log_10=0.0,
            cnt_a1=0, cnt_a2=0, cnt_a3=0, cnt_rel=0, cnt_rmse=0, cnt_log=0)

    for i in range(N):
        for label, prefix in outputs.items():
            gt_path   = os.path.join(SAVE_DIR, f'Batch_{i}_{prefix}_GT.pt')
            pred_path = os.path.join(SAVE_DIR, f'Batch_{i}_{prefix}_Pred.pt')

            if not os.path.isfile(gt_path) or not os.path.isfile(pred_path):
                print(f'  [skip] missing .pt for batch {i} ({label})')
                continue

            gt   = torch.load(gt_path,   map_location='cpu', weights_only=False)
            pred = torch.load(pred_path, map_location='cpu', weights_only=False)

            abs_rel, rmse, log_10, a1, a2, a3 = add_results_1(gt, pred, border_crop_size=16)

            acc = all_results[label]
            if torch.isfinite(a1):
                acc['a1']    += a1.detach().cpu().numpy();    acc['cnt_a1']  += 1
            if torch.isfinite(a2):
                acc['a2']    += a2.detach().cpu().numpy();    acc['cnt_a2']  += 1
            if torch.isfinite(a3):
                acc['a3']    += a3.detach().cpu().numpy();    acc['cnt_a3']  += 1
            if torch.isfinite(abs_rel):
                acc['abs_rel'] += abs_rel.detach().cpu().numpy(); acc['cnt_rel'] += 1
            if torch.isfinite(rmse):
                acc['rmse']  += rmse.detach().cpu().numpy();  acc['cnt_rmse'] += 1
            if torch.isfinite(log_10):
                acc['log_10'] += log_10.detach().cpu().numpy(); acc['cnt_log'] += 1

    print('\n{:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(
        'Output', 'a1', 'a2', 'a3', 'rel', 'rms', 'log10'))
    print('-' * 72)
    for label, acc in all_results.items():
        a1      = acc['a1']      / max(acc['cnt_a1'],  1)
        a2      = acc['a2']      / max(acc['cnt_a2'],  1)
        a3      = acc['a3']      / max(acc['cnt_a3'],  1)
        abs_rel = acc['abs_rel'] / max(acc['cnt_rel'], 1)
        rmse    = acc['rmse']    / max(acc['cnt_rmse'],1)
        log_10  = acc['log_10'] / max(acc['cnt_log'], 1)
        print('{:<18} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}'.format(
            label, a1, a2, a3, abs_rel, rmse, log_10))

    # Paper table row
    acc     = all_results['Complex (L_p)']
    a1      = acc['a1']      / max(acc['cnt_a1'],  1)
    a2      = acc['a2']      / max(acc['cnt_a2'],  1)
    a3      = acc['a3']      / max(acc['cnt_a3'],  1)
    abs_rel = acc['abs_rel'] / max(acc['cnt_rel'], 1)
    rmse    = acc['rmse']    / max(acc['cnt_rmse'],1)
    log_10  = acc['log_10'] / max(acc['cnt_log'], 1)

    print('\n' + '=' * 72)
    print('PAPER TABLE ROW  →  Proposed: Technique-4')
    print('=' * 72)
    print('{:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(
          'δ1↑', 'δ2↑', 'δ3↑', 'rel↓', 'rms↓', 'log10↓'))
    print('{:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}'.format(
          a1, a2, a3, abs_rel, rmse, log_10))


# ── Metric helpers — identical to original compute_errors_nyu ─────────────────

def compute_errors_nyu(pred, gt):
    y = gt
    x = pred
    thresh  = torch.max((y / x), (x / y))
    a1      = (thresh < 1.25     ).float().mean()
    a2      = (thresh < 1.25 ** 2).float().mean()
    a3      = (thresh < 1.25 ** 3).float().mean()
    abs_rel = torch.mean(torch.abs(y - x) / y)
    rmse    = (y - x) ** 2
    rmse    = torch.sqrt(rmse.mean())
    log_10  = (torch.abs(torch.log10(y) - torch.log10(x))).nanmean()
    return abs_rel, rmse, log_10, a1, a2, a3


def add_results_1(gt_image, pred_image, border_crop_size=16):
    predictions   = []
    testSetDepths = []
    half_border_size = border_crop_size // 2

    gt_image_border_cut   = gt_image[  :, :, half_border_size:-half_border_size,
                                              half_border_size:-half_border_size]
    pred_image_border_cut = pred_image[:, :, half_border_size:-half_border_size,
                                              half_border_size:-half_border_size]
    del gt_image, pred_image

    replicate             = nn.ReplicationPad2d(half_border_size)
    gt_image_border_cut   = replicate(gt_image_border_cut)
    pred_image_border_cut = replicate(pred_image_border_cut)

    gt_image_border_cut   = F.interpolate(gt_image_border_cut,   (480, 640),
                                          mode='bilinear', align_corners=True)
    pred_image_border_cut = F.interpolate(pred_image_border_cut, (480, 640),
                                          mode='bilinear', align_corners=True)

    for j in range(len(gt_image_border_cut)):
        predictions.append(pred_image_border_cut[j])
        testSetDepths.append(gt_image_border_cut[j])

    predictions   = torch.stack(predictions,  axis=0)
    testSetDepths = torch.stack(testSetDepths, axis=0)
    del pred_image_border_cut, gt_image_border_cut

    abs_rel, rmse, log_10, a1, a2, a3 = compute_errors_nyu(predictions, testSetDepths)
    del predictions, testSetDepths
    return abs_rel, rmse, log_10, a1, a2, a3


# ── Physics helpers ────────────────────────────────────────────────────────────

def compute_complex_image(output_depth, output_black_box, beta_val, a_mat, unit_mat, image_half):
    depth_3d    = torch.tile(output_depth, [1, 3, 1, 1])
    tx1         = torch.exp(-torch.mul(beta_val, depth_3d))
    second_term = torch.mul(a_mat, torch.subtract(unit_mat, tx1))
    return output_black_box + torch.add(torch.mul(image_half, tx1), second_term)


def compute_haze_image(output_depth, beta_val, a_mat, unit_mat, image_half):
    depth_3d    = torch.tile(output_depth, [1, 3, 1, 1])
    tx1         = torch.exp(-torch.mul(beta_val, depth_3d))
    second_term = torch.mul(a_mat, torch.subtract(unit_mat, tx1))
    return torch.add(torch.mul(image_half, tx1), second_term)


if __name__ == '__main__':
    main()
