import time
import argparse
from datetime import datetime
import numpy as np
import os
import math 
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils  
from torchvision import transforms  
from tensorboardX import SummaryWriter

from model import PTModel as Model
from model_3D import PTModel as Model3D
from loss import ssim
from data_4 import getTrainingValidationData
from utils import AverageMeter, DepthNorm, colorize, simple_save_images


# This code is very same as "train_2.py" but here I am using the saved "atmospheric_light.npy" and "beta.npy" files
# we are also using here the "index_haze_image.npz", "index_complex_haze_image.npz" and "index_complex_depth_half_3d.npz" files, during training 


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.000001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=10, type=int, help='batch size')
    args = parser.parse_args()
    
    train_me_where = "from_middle" # "from_begining"
    model_name = "densenet_multi_task"

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # Create model
    model_1 = Model().to(device)
    model_2 = Model3D().to(device)

    print('Model created.')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_1 = nn.DataParallel(model_1.cuda())
        model_2 = nn.DataParallel(model_2.cuda())

    print('model and cuda mixing done')

    # Training parameters
    optimizer_1 = torch.optim.Adam(model_1.parameters(), args.lr)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), args.lr)

    best_loss = 100.0
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)
    
    # Create and save  data
    # train_loader = getTrainingTestingData(batch_size=batch_size)
    # torch.save(train_loader, '/sanssauvegarde/homes/t20monda/Dense_Depth_1/train_loader.pkl')
    
    # Load data
    train_loader, val_loader = getTrainingValidationData(batch_size=args.bs, val_fraction=0.1)


    # Logging
    # writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer_1 = SummaryWriter(f'/datas/sandbox/gmoussa/runs/Technique_1/{run_id}')
    
    # Loss
    l1_criterion = nn.L1Loss()

    print("Total number of batches in train loader are :", len(train_loader))

    if train_me_where == "from_middle":
        checkpoint = torch.load('/datas/sandbox/gmoussa/runs/Technique_1/checkpoint/' + model_name + '/Models' + '.ckpt')

        model_1.load_state_dict(checkpoint['state_dict_1'])
        model_2.load_state_dict(checkpoint['state_dict_2'])

    # Start training...
    for epoch in range(args.epochs):
        
        losses = AverageMeter()
        N = len(train_loader)

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Switch to train model
        model_1.train()
        model_2.train()

        keep_all_batch_losses = []
        total_batch_loss = 0.0
        running_batch_loss = 0.0

        need_train = round((N*96)/100)
        cnt_batch = 0

        for i, sample_batched in enumerate(train_loader):
            if cnt_batch < need_train : # to skip the last batch from training    
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()

                # Prepare sample and target
                image = torch.autograd.Variable(sample_batched['image'].to(device))  # full size
                image_half = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size

                depth = torch.autograd.Variable(sample_batched['depth'].to(device))  # half size

                # depth_half = torch.autograd.Variable(sample_batched['depth_norm_simple'].to(device))  # half size
                orig_haze_image = torch.autograd.Variable(sample_batched['haze_image'].to(device))  # half size

                beta_val = torch.autograd.Variable(sample_batched['beta'].to(device))  # half size
                a_mat = torch.autograd.Variable(sample_batched['a_val'].to(device))  # half size
                unit_mat = torch.autograd.Variable(sample_batched['unit_mat'].to(device))  # half size
                complex_image_tensor = torch.autograd.Variable(sample_batched['complex_noise_img'].to(device))  # half size

                # Normalize depth
                depth_n = DepthNorm(depth)  # I think by this normalization, they will bring back the values within 0-1
                # so that they can calculate the loss w.r.t the estimated depth, which is within 0-1

                # Predict
                output_depth = model_1(image)
                output_black_box = model_2(image)

                pred_complex_image = compute_complex_image(output_depth, output_black_box, beta_val, a_mat, unit_mat,
                                                        image_half)

                # Compute the loss
                l_depth = l1_criterion(output_depth, depth_n)
                l_ssim = torch.clamp((1 - ssim(output_depth, depth_n, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
                loss_depth = (0.1 * l_ssim) + (0.1 * l_depth)

                loss_complex_image = l1_criterion(pred_complex_image, complex_image_tensor)
                loss_ssim_complex_image = torch.clamp((1 - ssim(pred_complex_image.float(), complex_image_tensor.float(),
                                                                val_range=1)) * 0.5, 0, 1)
                loss_complex_total = (1.0 * loss_ssim_complex_image) + (0.1 * loss_complex_image)

                # Update step
                total_batch_loss = loss_complex_total + loss_depth
                running_batch_loss += total_batch_loss.item() * image.size(0)  # dividing by number of images in the batch

                keep_all_batch_losses.append(total_batch_loss.item())

                losses.update(total_batch_loss.data.item(), image.size(0))
                total_batch_loss.backward()

                # Log progress
                niter = epoch*N +i

                if i % 5 == 0:
                    writer_1.add_scalar('Train/BatchLoss',       total_batch_loss.item(),       niter)
                    writer_1.add_scalar('Train/L1_Depth',        l_depth.item(),                niter)
                    writer_1.add_scalar('Train/SSIM_Depth',      l_ssim.item(),                 niter)
                    writer_1.add_scalar('Train/L1_Complex',      loss_complex_image.item(),     niter)
                    writer_1.add_scalar('Train/SSIM_Complex',    loss_ssim_complex_image.item(),niter)
                    writer_1.add_scalar('Train/LearningRate',    optimizer_1.param_groups[0]['lr'], niter)


                optimizer_1.step()
                optimizer_2.step()

            else :
                break
            cnt_batch = cnt_batch+1

        if math.isnan(losses.avg):
            print('I need to check you')

        # ---------------- Validation pass ----------------
        model_1.eval()
        model_2.eval()
        val_losses = AverageMeter()
        last_batch = None   # ← keep refs for image logging
        with torch.no_grad():
            for sample_batched in val_loader:
                image                = sample_batched['image'].to(device)
                image_half           = sample_batched['image_half'].to(device)
                depth                = sample_batched['depth'].to(device)
                beta_val             = sample_batched['beta'].to(device)
                a_mat                = sample_batched['a_val'].to(device)
                unit_mat             = sample_batched['unit_mat'].to(device)
                complex_image_tensor = sample_batched['complex_noise_img'].to(device)

                depth_n = DepthNorm(depth)

                output_depth     = model_1(image)
                output_black_box = model_2(image)
                pred_complex_image = compute_complex_image(
                    output_depth, output_black_box, beta_val, a_mat, unit_mat, image_half
                )

                l_depth = l1_criterion(output_depth, depth_n)
                l_ssim  = torch.clamp((1 - ssim(output_depth, depth_n, val_range=1000.0/10.0)) * 0.5, 0, 1)
                loss_depth_v = (1.0 * l_ssim) + (0.1 * l_depth)

                loss_complex_image      = l1_criterion(pred_complex_image, complex_image_tensor)
                loss_ssim_complex_image = torch.clamp(
                    (1 - ssim(pred_complex_image.float(), complex_image_tensor.float(), val_range=1)) * 0.5, 0, 1
                )
                loss_complex_total_v = (1.0 * loss_ssim_complex_image) + (0.1 * loss_complex_image)

                total_val_loss = loss_complex_total_v + loss_depth_v
                val_losses.update(total_val_loss.item(), image.size(0))

                # remember last batch tensors so we can show them in TB
                last_batch = {
                    'image_half':        image_half,
                    'depth':             depth,
                    'output_depth':      output_depth,
                    'complex_gt':        complex_image_tensor,
                    'pred_complex':      pred_complex_image,
                    'output_black_box':  output_black_box,
                }

        # ---------------- Logging + checkpoint ----------------
        print(f'Epoch [{epoch:>3d}/{args.epochs-1}]  '
              f'train_loss={losses.avg:.4f}  val_loss={val_losses.avg:.4f}')

        # Scalars
        writer_1.add_scalar('Train/EpochAvg', losses.avg,     epoch)
        writer_1.add_scalar('Val/EpochAvg',   val_losses.avg, epoch)

        # Images — only once per epoch, from the last val batch
        if last_batch is not None:
            writer_1.add_image('Val/1_Input_Image_Half',
                vutils.make_grid(last_batch['image_half'].data, nrow=4, normalize=True), epoch)
            writer_1.add_image('Val/2_GT_Depth',
                colorize(vutils.make_grid(last_batch['depth'].data, nrow=4, normalize=False)), epoch)
            writer_1.add_image('Val/3_Pred_Depth',
                colorize(vutils.make_grid(DepthNorm(last_batch['output_depth']).data, nrow=4, normalize=False)), epoch)
            writer_1.add_image('Val/4_GT_ComplexImage',
                vutils.make_grid(last_batch['complex_gt'].data, nrow=4, normalize=False), epoch)
            writer_1.add_image('Val/5_Pred_ComplexImage',
                vutils.make_grid(last_batch['pred_complex'].data, nrow=4, normalize=False), epoch)
            writer_1.add_image('Val/6_Residual_BlackBox',
                vutils.make_grid(last_batch['output_black_box'].data, nrow=4, normalize=True), epoch)

        writer_1.flush()  # ← force events to disk so TensorBoard sees them immediately

        if val_losses.avg < best_loss:
            print(f'  Val loss improved: {val_losses.avg:.4f} (was {best_loss:.4f}) — saving checkpoint')
            best_loss = val_losses.avg
            _save_best_model(model_1, model_2, best_loss, epoch)


def compute_complex_image(output_depth, output_black_box, beta_val, a_mat, unit_mat, image_half):

    output_depth_3d = torch.tile(output_depth, [1, 3, 1, 1])
    output_black_box_3d = output_black_box # torch.tile(output_black_box, [1, 3, 1, 1])

    tx1 = torch.exp(-torch.mul(beta_val, output_depth_3d))
    second_term = torch.mul(a_mat, (torch.subtract(unit_mat, tx1)))
    haze_image = torch.add((torch.mul(image_half, tx1)), second_term)

    pred_complex_image = output_black_box_3d + haze_image
    return pred_complex_image

def _save_best_model(model_1, model_2, best_loss, epoch):
    # Save Model
    model_name = "densenet_multi_task"
    state = {
        'state_dict_1': model_1.state_dict(),
        'state_dict_2': model_2.state_dict(),
        'best_acc': best_loss,
        'cur_epoch': epoch
    }

    if not os.path.isdir('/datas/sandbox/gmoussa/runs/Technique_1/checkpoint/' + model_name):
        os.makedirs('/datas/sandbox/gmoussa/runs/Technique_1/checkpoint/' + model_name)

    torch.save(state,
               '/datas/sandbox/gmoussa/runs/Technique_1/checkpoint/' +
               model_name + '/Models' + '.ckpt')  


if __name__ == '__main__':
    main()
