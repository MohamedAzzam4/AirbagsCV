import os
import cv2
import numpy as np
import glob
from pathlib import Path
import random

def prepare_aitex():
    dataset_root = Path('AITEX dataset')
    output_root = Path('datasets/aitex')
    
    # Create directories
    train_good = output_root / 'train' / 'good'
    test_good = output_root / 'test' / 'good'
    test_anomaly = output_root / 'test' / 'anomaly'
    gt_anomaly = output_root / 'ground_truth' / 'anomaly'
    
    for d in [train_good, test_good, test_anomaly, gt_anomaly]:
        d.mkdir(parents=True, exist_ok=True)
        
    patch_size = 256
    
    # 1. Process Normal Images
    normal_images = list(dataset_root.glob('NODefect_images/**/*.png'))
    random.seed(42)
    random.shuffle(normal_images)
    
    # 80/20 split
    split_idx = int(len(normal_images) * 0.8)
    train_normal = normal_images[:split_idx]
    test_normal = normal_images[split_idx:]
    
    print(f"Found {len(normal_images)} normal images. {len(train_normal)} for training, {len(test_normal)} for testing.")
    
    def process_normal(img_paths, output_dir):
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            num_patches = w // patch_size
            for i in range(num_patches):
                patch = img[:, i*patch_size:(i+1)*patch_size]
                out_path = output_dir / f"{img_path.stem}_patch{i}.png"
                cv2.imwrite(str(out_path), patch)
                
    process_normal(train_normal, train_good)
    process_normal(test_normal, test_good)
    
    # 2. Process Defect Images
    defect_images = list((dataset_root / 'Defect_images').glob('*.png'))
    mask_dir = dataset_root / 'Mask_images'
    
    print(f"Found {len(defect_images)} defect images.")
    
    for img_path in defect_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Load mask(s)
        base_name = img_path.stem
        # sometimes masks are named base_name + '_mask.png', sometimes '_mask1.png'
        mask_paths = list(mask_dir.glob(f"{base_name}_mask*.png"))
        
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mp in mask_paths:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                combined_mask = cv2.bitwise_or(combined_mask, m)
                
        num_patches = w // patch_size
        for i in range(num_patches):
            patch_img = img[:, i*patch_size:(i+1)*patch_size]
            patch_mask = combined_mask[:, i*patch_size:(i+1)*patch_size]
            
            # Check if mask has anomaly
            if np.sum(patch_mask) > 0:
                # Anomaly patch
                out_img_path = test_anomaly / f"{base_name}_patch{i}.png"
                # For Anomalib Folder structure, ground truth mask name must match test image name exactly or with _mask
                # Usually matching the exact same name is safest, but Folder allows suffixes. Let's use exact same name
                # or _mask. Actually, anomalib's Folder uses exact same filename for masks by default or expects a specific structure.
                # In anomalib 2.5, the mask file must have the same name as the image file, or end with _mask.
                # Let's just name the mask the EXACT SAME name as the image, this always works.
                out_mask_path = gt_anomaly / f"{base_name}_patch{i}.png"
                cv2.imwrite(str(out_img_path), patch_img)
                cv2.imwrite(str(out_mask_path), patch_mask)
            else:
                # Normal patch from defect image (put in test_good)
                out_img_path = test_good / f"{base_name}_patch{i}.png"
                cv2.imwrite(str(out_img_path), patch_img)
                
    print("AITEX dataset preprocessing complete.")

if __name__ == '__main__':
    prepare_aitex()
