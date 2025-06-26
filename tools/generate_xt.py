import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def inject_metal_artifact(x0, mask, artifact_intensity=1.5):
    """
    将金属伪影注入干净图像
    参数:
        x0: numpy 数组, 范围 0~1
        mask: numpy 数组, 0或1（表示金属区域）
        artifact_intensity: 金属区域强度增幅（模拟伪影）
    返回:
        合成图像 xT
    """
    xT = x0.copy()
    xT[mask == 1] *= artifact_intensity
    xT = np.clip(xT, 0.0, 1.0)
    return xT

def main(x0_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    x0_files = sorted(os.listdir(x0_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for fname in tqdm(x0_files):
        # 随机从 mask 文件夹选择一个图像
        mask_fname = random.choice(mask_files)

        # 读取 x0 和 mask 图像
        x0 = cv2.imread(os.path.join(x0_dir, fname), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(os.path.join(mask_dir, mask_fname), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)  # 将 mask 转为二值

        # 合成带伪影的图像
        xT = inject_metal_artifact(x0, mask, artifact_intensity=1.8)
        xT_uint8 = (xT * 255).astype(np.uint8)

        # 保存生成的 xT 图像
        cv2.imwrite(os.path.join(out_dir, fname), xT_uint8)

if __name__ == '__main__':
    x0_dir = 'data/train/x0'  # `x0` 文件夹路径
    mask_dir = 'data/train/mask'  # `mask` 文件夹路径
    out_dir = 'data/train/xT'  # 输出的 `xT` 文件夹路径
    main(x0_dir, mask_dir, out_dir)



      
      


