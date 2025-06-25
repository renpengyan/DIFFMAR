import torch
import torch.nn.functional as F

def masked_l1_loss(pred, target, mask):
    """
    mask: 1 表示正常区域，0 表示金属区域。
    实际优化目标为去除金属影响区域的误差，因此计算 (1 - mask) 区域的误差。
    """
    masked = (1 - mask)
    diff = torch.abs(pred - target) * masked
    loss = diff.sum() / (masked.sum() + 1e-8)
    return loss
