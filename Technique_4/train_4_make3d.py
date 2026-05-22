"""
train_4_make3d.py  –  Technique 4 on Make3D

Fixed equal loss weights.  Includes:
  - Validation loop after every epoch (uses test set)
  - TensorBoard logging of Train/Loss and Val/Loss
  - Early stopping (default patience = 5 epochs)
  - Best model saved based on validation loss, not training loss

Loss structure:
    L_d     = 1.0*L_SSIM_d  +  0.1*L1_d
    L_p/t/g = 0.1*L1  +  0.1*L_SSIM  +  0.1*L_perc
    L_total = L_d + L_p + L_t + L_g

Models:
    model_1  →  model.py     (depth,        1-ch)
    model_2  →  model_3D.py  (black-box,    3-ch)
    model_3  →  model_3D.py  (direct image, 3-ch)
"""

import os
import math
import argparse
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from model       import PTModel as Model
from model_3D    import PTModel as Model3D
from loss        import ssim, VGGPerceptualLoss
from data_make3d import getTrainingTestingData, getTestingData
from utils       import AverageMeter

BASE_DIR = r'C:\home\Georges\DenseDepth_3'


def main():
    parser = argparse.ArgumentParser(description='Technique 4 – Make3D')
    parser.add_argument('--epochs',   default=50,  type=int)
    parser.add_argument('--lr',       default=1e-6, type=float)
    parser.add_argument('--bs',       default=5,   type=int, help='batch size')
    parser.add_argument('--patience', default=5,   type=int, help='early stopping patience')
    args = parser.parse_args()

    train_me_where = "from_middle"
    model_name     = "densenet_multi_task_make3d"
    ckpt_filename  = "Models_Make3D_Tech4"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # ── Models ────────────────────────────────────────────────────────────────
    model_1 = Model().to(device)
    model_2 = Model3D().to(device)
    model_3 = Model3D().to(device)
    print('Models created.')

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model_1 = nn.DataParallel(model_1.cuda())
        model_2 = nn.DataParallel(model_2.cuda())
        model_3 = nn.DataParallel(model_3.cuda())

    # ── Optimizers ────────────────────────────────────────────────────────────
    optimizer_1 = torch.optim.Adam(model_1.parameters(), args.lr)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), args.lr)
    optimizer_3 = torch.optim.Adam(model_3.parameters(), args.lr)

    # ── Loss functions ────────────────────────────────────────────────────────
    l1       = nn.L1Loss()
    vgg_loss = VGGPerceptualLoss().to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = getTrainingTestingData(batch_size=args.bs)
    val_loader   = getTestingData(batch_size=args.bs)
    writer       = SummaryWriter(os.path.join(BASE_DIR, 'runs', 'Make3D', 'tech4'))
    print(f'Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}')

    # ── Optional resume ───────────────────────────────────────────────────────
    if train_me_where == "from_middle":
        ckpt_path  = os.path.join(BASE_DIR, 'checkpoint', model_name, ckpt_filename + '.ckpt')
        checkpoint = torch.load(ckpt_path)
        model_1.load_state_dict(checkpoint['state_dict_1'])
        model_2.load_state_dict(checkpoint['state_dict_2'])
        model_3.load_state_dict(checkpoint['state_dict_3'])

    best_val_loss    = float('inf')
    patience_counter = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch}/{args.epochs - 1}')
        print('-' * 10)

        # ── Train phase ───────────────────────────────────────────────────────
        model_1.train(); model_2.train(); model_3.train()
        train_losses = AverageMeter()

        for i, sample_batched in enumerate(train_loader):
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()

            image_full           = sample_batched['image_full'].to(device)
            image_half           = sample_batched['image_half'].to(device)
            depth_half           = sample_batched['depth_half'].to(device)
            orig_haze_image      = sample_batched['haze_image'].to(device)
            beta_val_half        = sample_batched['beta'].to(device)
            a_mat_half           = sample_batched['a_val'].to(device)
            unit_mat_half        = sample_batched['unit_mat'].to(device)
            complex_image_tensor = sample_batched['complex_noise_img'].to(device)

            output_depth  = model_1(image_full)
            output_bb     = model_2(image_full)
            output_direct = model_3(image_full)

            pred_complex = compute_complex_image(
                output_depth, output_bb, beta_val_half, a_mat_half, unit_mat_half, image_half)
            pred_haze    = compute_haze_image(
                output_depth, beta_val_half, a_mat_half, unit_mat_half, image_half)

            total_loss = _compute_total_loss(
                l1, vgg_loss, output_depth, depth_half,
                pred_complex, pred_haze,
                output_direct, complex_image_tensor, orig_haze_image)

            train_losses.update(total_loss.data.item(), image_full.size(0))
            total_loss.backward()
            optimizer_1.step(); optimizer_2.step(); optimizer_3.step()

            if i % 5 == 0:
                writer.add_scalar('Train/Loss_step', total_loss.item(),
                                  epoch * len(train_loader) + i)

        # ── Validation phase ──────────────────────────────────────────────────
        model_1.eval(); model_2.eval(); model_3.eval()
        val_losses = AverageMeter()

        with torch.no_grad():
            for sample_batched in val_loader:
                image_full           = sample_batched['image_full'].to(device)
                image_half           = sample_batched['image_half'].to(device)
                depth_half           = sample_batched['depth_half'].to(device)
                orig_haze_image      = sample_batched['haze_image'].to(device)
                beta_val_half        = sample_batched['beta'].to(device)
                a_mat_half           = sample_batched['a_val'].to(device)
                unit_mat_half        = sample_batched['unit_mat'].to(device)
                complex_image_tensor = sample_batched['complex_noise_img'].to(device)

                output_depth  = model_1(image_full)
                output_bb     = model_2(image_full)
                output_direct = model_3(image_full)

                pred_complex = compute_complex_image(
                    output_depth, output_bb, beta_val_half, a_mat_half, unit_mat_half, image_half)
                pred_haze    = compute_haze_image(
                    output_depth, beta_val_half, a_mat_half, unit_mat_half, image_half)

                val_loss = _compute_total_loss(
                    l1, vgg_loss, output_depth, depth_half,
                    pred_complex, pred_haze,
                    output_direct, complex_image_tensor, orig_haze_image)

                val_losses.update(val_loss.item(), image_full.size(0))

        # ── Per-epoch logging ─────────────────────────────────────────────────
        if math.isnan(train_losses.avg) or math.isnan(val_losses.avg):
            print('Warning: NaN loss detected.')

        print(f'Train Loss: {train_losses.avg:.4f}  |  Val Loss: {val_losses.avg:.4f}')
        writer.add_scalar('Train/Loss_epoch', train_losses.avg, epoch)
        writer.add_scalar('Val/Loss',         val_losses.avg,   epoch)

        # ── Early stopping (based on val loss) ────────────────────────────────
        if val_losses.avg < best_val_loss:
            print(f'Val loss improved: {val_losses.avg:.4f}  '
                  f'(prev best: {best_val_loss:.4f})  → saving model')
            best_val_loss    = val_losses.avg
            patience_counter = 0
            _save_best_model(model_1, model_2, model_3, best_val_loss, epoch,
                             model_name, ckpt_filename)
        else:
            patience_counter += 1
            print(f'No val improvement. Patience: {patience_counter}/{args.patience}')
            if patience_counter >= args.patience:
                print(f'Early stopping triggered at epoch {epoch}.')
                break

    writer.close()
    print(f'\nTraining complete. Best val loss: {best_val_loss:.4f}')


# ── Shared loss computation ────────────────────────────────────────────────────

def _compute_total_loss(l1, vgg_loss,
                        output_depth, depth_half,
                        pred_complex, pred_haze,
                        output_direct, complex_image_tensor, orig_haze_image):
    l_d_ssim   = torch.clamp(
        (1 - ssim(output_depth, depth_half, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
    loss_depth = 1.0 * l_d_ssim + 0.1 * l1(output_depth, depth_half)

    l_p_ssim     = torch.clamp(
        (1 - ssim(pred_complex.float(), complex_image_tensor.float(), val_range=1)) * 0.5, 0, 1)
    loss_complex = (0.1 * l1(pred_complex, complex_image_tensor) +
                    0.1 * l_p_ssim +
                    0.1 * vgg_loss(pred_complex.float(), complex_image_tensor.float()))

    l_t_ssim   = torch.clamp(
        (1 - ssim(pred_haze.float(), orig_haze_image.float(), val_range=1)) * 0.5, 0, 1)
    loss_haze  = (0.1 * l1(pred_haze, orig_haze_image) +
                  0.1 * l_t_ssim +
                  0.1 * vgg_loss(pred_haze.float(), orig_haze_image.float()))

    l_g_ssim    = torch.clamp(
        (1 - ssim(output_direct.float(), complex_image_tensor.float(), val_range=1)) * 0.5, 0, 1)
    loss_direct = (0.1 * l1(output_direct, complex_image_tensor) +
                   0.1 * l_g_ssim +
                   0.1 * vgg_loss(output_direct.float(), complex_image_tensor.float()))

    return loss_depth + loss_complex + loss_haze + loss_direct


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


def _save_best_model(model_1, model_2, model_3, best_loss, epoch, model_name, filename):
    save_dir = os.path.join(BASE_DIR, 'checkpoint', model_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'state_dict_1': model_1.state_dict(),
        'state_dict_2': model_2.state_dict(),
        'state_dict_3': model_3.state_dict(),
        'best_loss':    best_loss,
        'cur_epoch':    epoch,
    }, os.path.join(save_dir, filename + '.ckpt'))


if __name__ == '__main__':
    main()