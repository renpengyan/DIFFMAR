import random
import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img.astype('float32') / 255.0

class CTMetalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.x0_files = sorted(os.listdir(os.path.join(root_dir, 'x0')))
        self.mask_files = sorted(os.listdir(os.path.join(root_dir, 'mask')))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.x0_files)  

    def __getitem__(self, idx):
        # 当前样本的文件名
        fname = self.x0_files[idx]

        # 路径拼接（要求 x0/xT/xLI 同名）
        x0_path = os.path.join(self.root_dir, 'x0', fname)
        xT_path = os.path.join(self.root_dir, 'xT', fname)
        xLI_path = os.path.join(self.root_dir, 'xLI', fname)

       
        mask_fname = random.choice(self.mask_files)
        mask_path = os.path.join(self.root_dir, 'mask', mask_fname)

        # 读取图像（转换为 tensor）
        x0 = self.to_tensor(load_gray(x0_path))
        xT = self.to_tensor(load_gray(xT_path))
        xLI = self.to_tensor(load_gray(xLI_path))
        mask = self.to_tensor(load_gray(mask_path))

        return {
            'x0': x0,
            'xT': xT,
            'xLI': xLI,
            'mask': mask
        }

