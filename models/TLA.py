import torch
import torch.nn as nn


class TLAModule(nn.Module):
    """
    Time Latent-variable Adjustment (TLA) module.
    输入包括：
      - Lt_minus_1: t-1 时刻的时间嵌入（latent vector）
      - xt_minus_1: 当前时刻输入图像
      - x0_hat: 当前步骤恢复的图像
      - xT: 原始退化图像
    输出：
      - 调整后的 Lt_minus_1（即 L'_t = μ * L_{t-1} + λ）
    """
    def __init__(self, latent_dim=128, image_size=(416, 416)):
        super().__init__()
        C, H, W = 1, *image_size  # 图像输入 assumed shape (B,1,H,W)
        flat_size = C * H * W

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 3 * flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * latent_dim)  # 输出 μ 和 λ
        )

    def forward(self, Lt_minus_1, xt_minus_1, x0_hat, xT):
        B, C, H, W = xt_minus_1.shape
        flat_xt = self.flatten(xt_minus_1)
        flat_x0 = self.flatten(x0_hat)
        flat_xT = self.flatten(xT)

        input_vec = torch.cat([Lt_minus_1, flat_xt, flat_x0, flat_xT], dim=1)
        output = self.fc(input_vec)
        mu, lam = output.chunk(2, dim=1)

        return mu * Lt_minus_1 + lam  # Eq. (7)
