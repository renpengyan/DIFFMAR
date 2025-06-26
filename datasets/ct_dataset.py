import random
import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype('float32') / 255.0

class CTMetalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.x0_files = sorted(os.listdir(os.path.join(root_dir, 'x0')))
        self.mask_files = sorted(os.listdir(os.path.join(root_dir, 'mask')))
        self.xT_files = sorted(os.listdir(os.path.join(root_dir, 'xT')))
        self.xLI_files = sorted(os.listdir(os.path.join(root_dir, 'xLI')))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.x0_files) 

    def __getitem__(self, idx):
        x0_path = os.path.join(self.root_dir, 'x0', self.x0_files[idx])
        xT_path = os.path.join(self.root_dir, 'xT', self.xT_files[idx])
        xLI_path = os.path.join(self.root_dir, 'xLI', self.xLI_files[idx])

  
        mask_file = random.choice(self.mask_files)
        mask_path = os.path.join(self.root_dir, 'mask', mask_file)

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

