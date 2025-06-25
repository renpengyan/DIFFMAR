import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10

def calculate_psnr(pred, target):
    """
    pred, target: Tensor (1, H, W) or (B, 1, H, W)
    输出：PSNR (float or list)
    """
    if pred.ndim == 4:
        psnr_list = []
        for p, t in zip(pred, target):
            psnr_list.append(calculate_psnr(p, t))
        return psnr_list

    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 10 * log10(1.0 / mse)


def calculate_ssim(pred, target):
    """
    pred, target: Tensor (1, H, W) or (B, 1, H, W)
    输出：SSIM (float or list)
    """
    if pred.ndim == 4:
        ssim_list = []
        for p, t in zip(pred, target):
            ssim_list.append(calculate_ssim(p, t))
        return ssim_list

    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    return ssim(pred, target, data_range=1.0)
