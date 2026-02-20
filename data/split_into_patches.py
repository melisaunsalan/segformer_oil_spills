import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split large images into smaller patches.")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Directory containing large images.")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save the image patches.")
    parser.add_argument("--patch_size", "-p",type=int, default=512, help="Size of the square patches.")
    parser.add_argument("--overlap", "-ov", type=float, default=0.5, help="Overlap between patches.")
    
    parser.add_argument("--split", "-s", type=str, default="both", choices=["train", "test", "both"], help="Dataset split: train, test or both.")
    args = parser.parse_args()

    assert 0.0 <= args.overlap <= 0.5, "Overlap must be in the range [0.0, 0.5]."

    patch_size = args.patch_size
    overlap = int(args.overlap * patch_size)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.split == "both":
        splits = ['train', 'test']
    else:
        splits = [args.split]

    for split in splits:
        print(f"Processing {split} set...")
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'labels_1D'), exist_ok=True)

        for img_name in tqdm(os.listdir(os.path.join(args.input_dir, split, 'images'))):
            img_path = os.path.join(args.input_dir, split, 'images', img_name)
            label_path = os.path.join(args.input_dir, split, 'labels_1D', img_name.replace(".jpg", ".png"))
            if img_name == '.DS_Store':
                continue
           
            # load RGB image as H x W x 3 uint8
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)

            h, w, _ = img.shape

            label = Image.open(label_path)
            label = np.array(label, dtype=np.uint8)

            h_idx = 0
            w_idx = 0

            for i in range(0, h, patch_size - overlap):
                w_idx = 0
                for j in range(0, w, patch_size - overlap):
                    patch = img[i:i + patch_size, j:j + patch_size]
                    label_patch = label[i:i + patch_size, j:j + patch_size]
                    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                        continue  # Skip incomplete patches

                    patch_filename = f"{os.path.splitext(img_name)[0]}_patch_{h_idx}_{w_idx}.png"
                    patch_path = os.path.join(args.output_dir, split, 'images', patch_filename)
                    # patch_path = os.path.join(args.output_dir, split, img_name.split('.')[0], patch_filename)

                    label_patch_path = os.path.join(args.output_dir, split, 'labels_1D', patch_filename)
           
                    cv2.imwrite(label_patch_path, label_patch)
                    cv2.imwrite(patch_path, patch)

                    w_idx += 1
                h_idx+= 1
    

