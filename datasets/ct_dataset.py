import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img.astype('float32') / 255.0

class CTMetalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.to_tensor = transforms.ToTensor()

        x0_dir = os.path.join(root_dir, 'x0')
        xT_dir = os.path.join(root_dir, 'xT')
        xLI_dir = os.path.join(root_dir, 'xLI')
        mask_dir = os.path.join(root_dir, 'mask')

        # 只保留同时在 x0/xT/xLI 中存在的文件
        all_x0 = sorted([f for f in os.listdir(x0_dir) if f.endswith('.png')])
        xT_set = set(os.listdir(xT_dir)) if os.path.exists(xT_dir) else set()
        xLI_set = set(os.listdir(xLI_dir)) if os.path.exists(xLI_dir) else set()

        self.x0_files = [f for f in all_x0 if f in xT_set and f in xLI_set]
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.x0_files)

    def __getitem__(self, idx):
        fname = self.x0_files[idx]

        x0_path = os.path.join(self.root_dir, 'x0', fname)
        xT_path = os.path.join(self.root_dir, 'xT', fname)
        xLI_path = os.path.join(self.root_dir, 'xLI', fname)
        mask_path = os.path.join(self.root_dir, 'mask', random.choice(self.mask_files))

        x0 = load_gray(x0_path)
        xT = load_gray(xT_path)
        xLI = load_gray(xLI_path)
        mask = load_gray(mask_path)

        return {
            'x0': self.to_tensor(x0),
            'xT': self.to_tensor(xT),
            'xLI': self.to_tensor(xLI),
            'mask': self.to_tensor(mask)
        }


