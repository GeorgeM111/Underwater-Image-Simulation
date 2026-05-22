"""
train_4.py  –  Technique 4

Loss structure per RGB reconstruction task:
    L_x = 0.1 * L1  +  0.1 * L_SSIM  +  0.1 * L_perc       (fixed equal weights)

Depth loss (unchanged from baseline):
    L_d = 1.0 * L_SSIM  +  0.1 * L1

Total loss:
    L_total = L_d  +  L_p  +  L_t  +  L_g

Models:
    model_1  →  model.py       (DenseNet-169 depth estimator, 1-ch output)
    model_2  →  model_3D.py    (DenseNet-169 black-box residue, 3-ch output)
    model_3  →  model_3D.py    (DenseNet-169 direct image,     3-ch output)

Dataset: NYU Depth V2  (via data_3.py)
"""

import os
import math
import argparse
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from model   import PTModel as Model
from model_3D import PTModel as Model3D
from loss    import ssim, VGGPerceptualLoss
from data_3  import getTrainingTestingData
from utils   import AverageMeter

BASE_DIR = r'C:\home\Georges'


def main():
    parser = argparse.ArgumentParser(description='Technique 4 – NYU Depth V2')
    parser.add_argument('--epochs', default=50,   type=int)
    parser.add_argument('--lr',     default=1e-6, type=float)
    parser.add_argument('--bs',     default=5,    type=int, help='batch size')
    args = parser.parse_args()

    train_me_where = "from_begining"
    model_name     = "densenet_multi_task"
    ckpt_filename  = "Models_NYU_Tech4"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Optimizers
    optimizer_1 = torch.optim.Adam(model_1.parameters(), args.lr)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), args.lr)
    optimizer_3 = torch.optim.Adam(model_3.parameters(), args.lr)

    # Loss functions
    l1 = nn.L1Loss()
    vgg_loss = VGGPerceptualLoss().to(device)

    train_loader = getTrainingTestingData(batch_size=args.bs)
    writer = SummaryWriter(os.path.join(BASE_DIR, 'tech4_run'))
    print("Total batches:", len(train_loader))

    # ── Optional resume ───────────────────────────────────────────────────────
    if train_me_where == "from_middle":
        ckpt_path = os.path.join(BASE_DIR, 'tech4_check', model_name, ckpt_filename + '.ckpt')
        checkpoint = torch.load(ckpt_path)
        model_1.load_state_dict(checkpoint['state_dict_1'])
        model_2.load_state_dict(checkpoint['state_dict_2'])
        model_3.load_state_dict(checkpoint['state_dict_3'])

    best_loss = 100.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        losses = AverageMeter()

        model_1.train()
        model_2.train()
        model_3.train()

        for i, sample_batched in enumerate(train_loader):

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()

            image_full          = sample_batched['image'].to(device)
            image_half          = sample_batched['image_half'].to(device)
            depth_half          = sample_batched['depth'].to(device)
            orig_haze_image     = sample_batched['haze_image'].to(device)
            beta_val_half       = sample_batched['beta'].to(device)
            a_mat_half          = sample_batched['a_val'].to(device)
            unit_mat_half       = sample_batched['unit_mat'].to(device)
            complex_image_tensor = sample_batched['complex_noise_img'].to(device)

            output_depth  = model_1(image_full)
            output_bb     = model_2(image_full)
            output_direct = model_3(image_full)

            pred_complex = compute_complex_image(
                output_depth, output_bb, beta_val_half, a_mat_half, unit_mat_half, image_half)
            pred_haze = compute_haze_image(
                output_depth, beta_val_half, a_mat_half, unit_mat_half, image_half)

            # Depth loss
            l_d_l1   = l1(output_depth, depth_half)
            l_d_ssim = torch.clamp(
                (1 - ssim(output_depth, depth_half, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
            loss_depth = 1.0 * l_d_ssim + 0.1 * l_d_l1

            # Complex image loss (L_p)
            l_p_l1   = l1(pred_complex, complex_image_tensor)
            l_p_ssim = torch.clamp(
                (1 - ssim(pred_complex.float(), complex_image_tensor.float(), val_range=1)) * 0.5, 0, 1)
            l_p_perc = vgg_loss(pred_complex.float(), complex_image_tensor.float())
            loss_complex = 0.1 * l_p_l1 + 0.1 * l_p_ssim + 0.1 * l_p_perc

            # Haze loss (L_t)
            l_t_l1   = l1(pred_haze, orig_haze_image)
            l_t_ssim = torch.clamp(
                (1 - ssim(pred_haze.float(), orig_haze_image.float(), val_range=1)) * 0.5, 0, 1)
            l_t_perc = vgg_loss(pred_haze.float(), orig_haze_image.float())
            loss_haze = 0.1 * l_t_l1 + 0.1 * l_t_ssim + 0.1 * l_t_perc

            # Direct loss (L_g)
            l_g_l1   = l1(output_direct, complex_image_tensor)
            l_g_ssim = torch.clamp(
                (1 - ssim(output_direct.float(), complex_image_tensor.float(), val_range=1)) * 0.5, 0, 1)
            l_g_perc = vgg_loss(output_direct.float(), complex_image_tensor.float())
            loss_direct = 0.1 * l_g_l1 + 0.1 * l_g_ssim + 0.1 * l_g_perc

            total_loss = loss_depth + loss_complex + loss_haze + loss_direct
            losses.update(total_loss.data.item(), image_full.size(0))
            total_loss.backward()

            optimizer_1.step()
            optimizer_2.step()
            optimizer_3.step()

        # ── Per-epoch output (the only terminal output you'll see) ────────────
        if math.isnan(losses.avg):
            print('Warning: NaN loss detected.')

        print(f'Epoch {epoch}/{args.epochs - 1}')
        print(f'Loss: {losses.avg:.4f}')

        # ── Save best model ───────────────────────────────────────────────────
        if losses.avg < best_loss:
            best_loss = losses.avg
            _save_best_model(model_1, model_2, model_3, best_loss, epoch, model_name, ckpt_filename)

    writer.close()


# ── Physics helpers ────────────────────────────────────────────────────────────

def compute_complex_image(output_depth, output_black_box, beta_val, a_mat, unit_mat, image_half):
    """I_complex = black_box + I_haze"""
    depth_3d    = torch.tile(output_depth, [1, 3, 1, 1])
    tx1         = torch.exp(-torch.mul(beta_val, depth_3d))
    second_term = torch.mul(a_mat, torch.subtract(unit_mat, tx1))
    haze_image  = torch.add(torch.mul(image_half, tx1), second_term)
    return output_black_box + haze_image


def compute_haze_image(output_depth, beta_val, a_mat, unit_mat, image_half):
    """Koschmieder atmospheric scattering model."""
    depth_3d    = torch.tile(output_depth, [1, 3, 1, 1])
    tx1         = torch.exp(-torch.mul(beta_val, depth_3d))
    second_term = torch.mul(a_mat, torch.subtract(unit_mat, tx1))
    return torch.add(torch.mul(image_half, tx1), second_term)


def _save_best_model(model_1, model_2, model_3, best_loss, epoch, model_name, filename):
    save_dir = os.path.join(BASE_DIR, 'tech4_check', model_name)
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