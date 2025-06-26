import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def linear_interpolate(xT, mask):
    xLI = xT.copy()
    H, W = xT.shape
    for i in range(H):
        m = mask[i]
        if m.sum() == 0:
            continue
        indices = np.where(m == 1)[0]
        for j in indices:
            left, right = j - 1, j + 1
            while left >= 0 and m[left] == 1: left -= 1
            while right < W and m[right] == 1: right += 1
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
    xT_files = sorted([f for f in os.listdir(xT_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    print(f"🔧 Generating xLI for {len(xT_files)} xT images with {len(mask_files)} masks")

    for fname in tqdm(xT_files):
        mask_fname = random.choice(mask_files)
        xT = cv2.imread(os.path.join(xT_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_dir, mask_fname), cv2.IMREAD_GRAYSCALE)

        if xT is None or mask is None:
            print(f"[ERROR] Reading failed: {fname} or {mask_fname}")
            continue

        xT = xT.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.uint8)
        xLI = linear_interpolate(xT, mask)
        xLI_uint8 = (xLI * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, fname), xLI_uint8)
        if __name__ == '__main__':
            import sys
            if len(sys.argv) == 4:
                main(sys.argv[1], sys.argv[2], sys.argv[3])
            else:
                print("Usage: python generate_xli.py <xT_dir> <mask_dir> <out_dir>")






