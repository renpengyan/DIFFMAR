import torch
import torch.nn as nn

class TLAModule(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.initialized = False

    def forward(self, Lt_minus_1, xt_minus_1, x0_hat, xT):
        B = Lt_minus_1.size(0)
        flat_xt = xt_minus_1.view(B, -1)
        flat_x0 = x0_hat.view(B, -1)
        flat_xT = xT.view(B, -1)

        input_vec = torch.cat([Lt_minus_1, flat_xt, flat_x0, flat_xT], dim=1)

        # 动态初始化 fc 层（只初始化一次）
        if not hasattr(self, 'fc'):
            input_dim = input_vec.size(1)
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 2 * self.latent_dim)
            ).to(input_vec.device)
            self.initialized = True

        output = self.fc(input_vec)
        mu, lam = output.chunk(2, dim=1)
        return mu * Lt_minus_1 + lam
