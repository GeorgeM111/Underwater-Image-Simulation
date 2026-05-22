import torch
import torch.nn as nn
from math import exp
import torch.nn.functional as F
import torchvision.models as vgg_models
import math
import sys

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        # ret = ssim_map[~torch.isnan(ssim_map)].mean()
        ret = ssim_map.nanmean()
    else:
        # ret = ssim_map[~torch.isnan(ssim_map)].mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if math.isnan(ret) or math.isinf(ret):
        print("Check me, I have issues")
        print("The value of ret is {}".format(ret))

        if (ssim_map.isnan().any()):
            print("Print ssim_map : ", ssim_map)
        if (img1.isnan().any()):
            print("Print pred image : ", img1)
        if (img2.isnan().any()):
            print("Print original image : ", img2)

        sys.exit("Due to NaN value, I am exiting")

    if full:
        return ret, cs

    return ret


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 feature maps up to relu3_3 (layer index 15).
 
    Formula (from paper):
        L_perc(I, I_hat) = (1 / C_j H_j W_j) * || phi_j(I) - phi_j(I_hat) ||_F^2
 
    where phi_j denotes the feature map at layer j of VGG16 pre-trained on ImageNet,
    and ||.||_F^2 is the squared Frobenius norm.
 
    Notes:
        - Applied ONLY to RGB image reconstruction losses (L_p, L_t, L_g).
        - NOT applied to the depth loss L_d.
        - VGG16 weights are frozen throughout training.
        - Inputs are expected to be in [0, 1]; they are clamped internally.
    """
 
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg_models.vgg16(pretrained=True)
        # Extract features up to relu3_3 (first 16 layers, indices 0-15)
        # VGG16 feature layers: conv1_1, relu, conv1_2, relu, pool1,
        #                        conv2_1, relu, conv2_2, relu, pool2,
        #                        conv3_1, relu, conv3_2, relu, conv3_3, relu ← index 15
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:16])
        # Freeze all VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
 
    def forward(self, pred, target):
        """
        Args:
            pred   : Tensor of shape B x 3 x H x W, values in [0, 1]
            target : Tensor of shape B x 3 x H x W, values in [0, 1]
        Returns:
            Scalar perceptual loss (batch mean).
        """
        pred   = torch.clamp(pred.float(),   0.0, 1.0)
        target = torch.clamp(target.float(), 0.0, 1.0)
 
        phi_pred   = self.feature_extractor(pred)
        phi_target = self.feature_extractor(target)
 
        _, C_j, H_j, W_j = phi_pred.shape
 
        # Squared Frobenius norm per sample, normalised by spatial-channel volume
        diff            = phi_pred - phi_target
        loss_per_sample = torch.sum(diff ** 2, dim=[1, 2, 3]) / (C_j * H_j * W_j)
        return loss_per_sample.mean()