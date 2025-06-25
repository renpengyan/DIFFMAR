import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2


class CTMetalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir 下应包含：x0/, xT/, mask/, xLI/
        """
        super().__init__()
        self.root_dir = root_dir
        self.x0_files = sorted(os.listdir(os.path.join(root_dir, 'x0')))
        self.xT_files = sorted(os.listdir(os.path.join(root_dir, 'xT')))
        self.mask_files = sorted(os.listdir(os.path.join(root_dir, 'mask')))
        self.xLI_files = sorted(os.listdir(os.path.join(root_dir, 'xLI')))

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # 自动将 HWC 图像转为 CHW 格式
        ])

    def __len__(self):
        return len(self.x0_files)

    def __getitem__(self, idx):
        def load_gray(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 归一化到 [0, 1]
            return img

        x0 = load_gray(os.path.join(self.root_dir, 'x0', self.x0_files[idx]))
        xT = load_gray(os.path.join(self.root_dir, 'xT', self.xT_files[idx]))
        mask = load_gray(os.path.join(self.root_dir, 'mask', self.mask_files[idx]))
        xLI = load_gray(os.path.join(self.root_dir, 'xLI', self.xLI_files[idx]))

        return {
            'x0': self.transform(x0),     # 干净图像
            'xT': self.transform(xT),     # 带伪影图像
            'mask': self.transform(mask), # 掩膜
            'xLI': self.transform(xLI)    # 线性插值图像
        }
