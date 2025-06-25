import os
import cv2
import numpy as np
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
    files = sorted(os.listdir(x0_dir))
    for fname in tqdm(files):
        x0 = cv2.imread(os.path.join(x0_dir, fname), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = (mask = 127).astype(np.uint8)

        xT = inject_metal_artifact(x0, mask, artifact_intensity=1.8)
        xT_uint8 = (xT * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, fname), xT_uint8)

if __name__ == '__main__':
    # 示例路径
    x0_dir = 'data/train/x0'
    mask_dir = 'data/train/mask'
    out_dir = 'data/train/xT'

    main(x0_dir, mask_dir, out_dir)
