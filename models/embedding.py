import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        输入:
            t: Tensor of shape (B,) - 当前的时间步整数
        输出:
            pos_embedding: Tensor of shape (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        """
        输入:
            t: (B,) 或 (B, 1) 形式的整数时间步
        输出:
            时间嵌入向量 Lt: (B, dim)
        """
        emb = self.pos_emb(t)
        return self.mlp(emb)
