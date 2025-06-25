import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x + self.block(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attention = torch.bmm(q, k)
        attention = F.softmax(attention / (C ** 0.5), dim=-1)

        out = torch.bmm(v, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.ReLU(),
            ResBlock(out_ch)
        )

    def forward(self, x):
        return self.encode(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.ReLU(),
            ResBlock(out_ch)
        )

    def forward(self, x, skip):
        x = self.decode(x)
        return x + skip


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.down1 = UNetBlock(in_channels, base_channels)
        self.down2 = UNetBlock(base_channels, base_channels * 2)
        self.down3 = UNetBlock(base_channels * 2, base_channels * 4)
        self.down4 = UNetBlock(base_channels * 4, base_channels * 8)

        self.attn = SelfAttentionBlock(base_channels * 8)

        self.up3 = UNetUpBlock(base_channels * 8, base_channels * 4)
        self.up2 = UNetUpBlock(base_channels * 4, base_channels * 2)
        self.up1 = UNetUpBlock(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x, time_embedding, sie_features=None):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        d4 = self.attn(d4)

        u3 = self.up3(d4, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)

        if sie_features is not None:
            u1 += sie_features  # 融合来自SIE的引导特征

        out = self.final(u1)
        return out
