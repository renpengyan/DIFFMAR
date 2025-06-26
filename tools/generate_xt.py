import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def inject_metal_artifact(x0, mask, artifact_intensity=1.5):
    xT = x0.copy()
    xT[mask == 1] *= artifact_intensity
    xT = np.clip(xT, 0.0, 1.0)
    return xT

def main(x0_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    x0_files = sorted([f for f in os.listdir(x0_dir) if f.lower().endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])

    print(f"🔎 Found {len(x0_files)} x0 images, {len(mask_files)} mask images")

    for fname in tqdm(x0_files, desc="Generating xT"):
        mask_fname = random.choice(mask_files)
        x0_path = os.path.join(x0_dir, fname)
        mask_path = os.path.join(mask_dir, mask_fname)

        x0 = cv2.imread(x0_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if x0 is None:
            print(f"[ERROR] ❌ Failed to read x0 image: {x0_path}")
            continue
        if mask is None:
            print(f"[ERROR] ❌ Failed to read mask image: {mask_path}")
            continue

        x0 = x0.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.uint8)

        xT = inject_metal_artifact(x0, mask, artifact_intensity=1.8)
        xT_uint8 = (xT * 255).astype(np.uint8)

        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, xT_uint8)
        print(f"[✔] Saved xT image: {out_path}")

if __name__ == '__main__':
    # 示例：可手动执行测试用
    main('data/train/x0', 'data/train/mask', 'data/train/xT')

      
      


