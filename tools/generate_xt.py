import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def inject_metal_artifact(x0, mask, artifact_intensity=1.8):
    xT = x0.copy()
    xT[mask == 1] *= artifact_intensity
    return np.clip(xT, 0.0, 1.0)

def main(x0_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    x0_files = sorted([f for f in os.listdir(x0_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    print(f"🔧 Generating xT for {len(x0_files)} x0 images with {len(mask_files)} mask images")

    for fname in tqdm(x0_files):
        mask_fname = random.choice(mask_files)
        x0 = cv2.imread(os.path.join(x0_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_dir, mask_fname), cv2.IMREAD_GRAYSCALE)

        if x0 is None or mask is None:
            print(f"[ERROR] Reading failed: {fname} or {mask_fname}")
            continue

        x0 = x0.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.uint8)
        xT = inject_metal_artifact(x0, mask)
        xT_uint8 = (xT * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, fname), xT_uint8)
    

      
      


