import os
import cv2
import numpy as np
from tqdm import tqdm

def linear_interpolate(xT, mask):
    """
    使用 mask 对应的金属区域进行线性插值修复
    Args:
        xT: 伪影图像 (float32, 0~1)
        mask: 二值金属区域 (0: normal, 1: metal)
    Returns:
        xLI: 修复后的图像
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
            # 找到非金属区域左右边界
            left = j - 1
            while left >= 0 and m[left] == 1:
                left -= 1
            right = j + 1
            while right < W and m[right] == 1:
                right += 1

            # 边界检查与线性插值
            if left >= 0 and right < W:
                xLI[i, j] = (xLI[i, left] * (right - j) + xLI[i, right] * (j - left)) / (right - left)
            elif left >= 0:
                xLI[i, j] = xLI[i, left]
            elif right < W:
                xLI[i, j] = xLI[i, right]
            else:
                xLI[i, j] = 0.0  # 退路

    return np.clip(xLI, 0.0, 1.0)

def main(xT_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.listdir(xT_dir))

    for fname in tqdm(files):
        xT = cv2.imread(os.path.join(xT_dir, fname), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = (mask = 127).astype(np.uint8)

        xLI = linear_interpolate(xT, mask)
        xLI_uint8 = (xLI * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, fname), xLI_uint8)

if __name__ == '__main__':
    xT_dir = 'data/train/xT'
    mask_dir = 'data/train/mask'
    out_dir = 'data/train/xLI'
    main(xT_dir, mask_dir, out_dir)
