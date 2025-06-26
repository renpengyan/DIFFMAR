import random
import os
import cv2
import numpy as np
from tqdm import tqdm

def linear_interpolate(xT, mask):
    """
    使用 mask 对应的金属区域进行线性插值修复
    """
    xLI = xT.copy()
    H, W = xT.shape

    for i in range(H):
        row = xLI[i]
        m = mask[i]
        if m.sum() == 0:
            continue

        indices = np.where(m == 1)[0]
        for j in indices:
            left = j - 1
            while left >= 0 and m[left] == 1:
                left -= 1
            right = j + 1
            while right < W and m[right] == 1:
                right += 1

            if left >= 0 and right < W:
                xLI[i, j] = (xLI[i, left] * (right - j) + xLI[i, right] * (j - left)) / (right - left)
            elif left >= 0:
                xLI[i, j] = xLI[i, left]
            elif right < W:
                xLI[i, j] = xLI[i, right]
            else:
                xLI[i, j] = 0.0

    return np.clip(xLI, 0.0, 1.0)

def main(xT_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xT_files = sorted(os.listdir(xT_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for fname in tqdm(xT_files):
        # 随机从 mask 文件夹选择一个图像
        mask_fname = random.choice(mask_files)

        # 构建路径
        xT_path = os.path.join(xT_dir, fname)
        mask_path = os.path.join(mask_dir, mask_fname)

        # 读取图像
        xT = cv2.imread(xT_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if xT is None:
            print(f"[ERROR] Failed to read xT image: {xT_path}")
            continue
        if mask is None:
            print(f"[ERROR] Failed to read mask image: {mask_path}")
            continue

        xT = xT.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.uint8)

        # 合成线性插值图像
        xLI = linear_interpolate(xT, mask)
        xLI_uint8 = (xLI * 255).astype(np.uint8)

        # 保存生成图像
        cv2.imwrite(os.path.join(out_dir, fname), xLI_uint8)



