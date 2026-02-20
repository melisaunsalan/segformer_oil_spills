import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import evaluate
import argparse
import yaml

import json

from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch import nn
from tqdm import tqdm
from datasets import disable_caching
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import random

from torch.utils.data import Dataset
from PIL import Image
import torch

import matplotlib.pylab as plt

if __name__ == '__main__':

    disable_caching()
    torch.manual_seed(0)
    np.random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/test_config.yaml")
    parser.add_argument("--patch_size", type=int, default=512, help="Size of the square patches.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap between patches.")
    parser.add_argument("--use_ground_truths", action='store_true', help="Whether to use ground truth labels for evaluation.")
    args = parser.parse_args() 

    PATCH_SIZE = args.patch_size
    OVERLAP = args.overlap

    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    num_classes = cfg['DATASET_PARAMS']['num_classes']

    # Init the image processor and transform
    image_processor = SegformerImageProcessor(do_rescale=False,
                                              do_normalize=False)
    
    # Label and id
    id2label = {"0": "sea", "1": "oil_spill", "2": "look_alike", "3":"ship", "4":"land"}
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

        # Color palette 
    palette = np.array([[0, 0, 0], # black - sea
                        [0, 255, 255], # cyan - oil spill
                        [255, 0, 0], # red - look-alike
                        [153, 76, 0], # brown - ship
                        [0, 153, 0]])  # green - land
        
    root_folder = cfg['DATASET_PARAMS']['db_path']

    # Define the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: ', device)
    ckp_path = cfg['MODEL_PARAMS']['model_path']
    print(ckp_path)
    
    if ckp_path and os.path.isfile(ckp_path):
        print(f"Loading model from checkpoint: {ckp_path}")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-" + cfg['MODEL_PARAMS']['model_config'], 
        num_labels=num_classes, id2label=id2label, label2id=label2id)
        model.load_state_dict(torch.load(ckp_path, map_location=device), strict=True)
    else:
        print("No valid checkpoint provided, loading pretrained model.")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-" + cfg['MODEL_PARAMS']['model_config'], 
                                                                num_labels=num_classes, 
                                                                id2label=id2label, 
                                                                label2id=label2id)
                                                                    
    # Move model to device
    model.to(device)  
    model.eval()

    if args.use_ground_truths:
        metric = evaluate.load("mean_iou")
        prec = evaluate.load("precision")
        rec = evaluate.load("recall")

        all_predictions, all_references = [], []



    # Read each image in the folder, split into patches, run inference on each patch, and reconstruct the full image
    for image in tqdm(os.listdir(root_folder)):
        if image == '.DS_Store':
            continue
        image_path = os.path.join(root_folder, image)
        img = Image.open(image_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        img = img/255.0

        if args.use_ground_truths:
            label_path = os.path.join(root_folder.replace('images', 'labels_1D'), image.replace(".jpg", ".png"))
            ground_truth = Image.open(label_path)
            ground_truth = np.array(ground_truth, dtype=np.uint8)

        # Split into patches
        h, w, _ = img.shape
        assert h >= PATCH_SIZE and w >= PATCH_SIZE, "Image is smaller than patch size"
        patches = []
        for start_i in range(0, h - PATCH_SIZE + 1, int(PATCH_SIZE * (1 - OVERLAP))):
            for start_j in range(0, w - PATCH_SIZE + 1, int(PATCH_SIZE * (1 - OVERLAP))):
                patch = img[start_i:start_i + PATCH_SIZE, start_j:start_j + PATCH_SIZE]
                patches.append((patch, start_i, start_j))  

        # Handle edge cases for remaining pixels at the borders
        if (h - PATCH_SIZE) % int(PATCH_SIZE * (1 - OVERLAP)) != 0:
            start_i = h - PATCH_SIZE
            for start_j in range(0, w - PATCH_SIZE + 1, int(PATCH_SIZE * (1 - OVERLAP))):
                patch = img[start_i:start_i + PATCH_SIZE, start_j:start_j + PATCH_SIZE]
                patches.append((patch, start_i, start_j))

        if (w - PATCH_SIZE) % int(PATCH_SIZE * (1 - OVERLAP)) != 0:
            start_j = w - PATCH_SIZE
            for start_i in range(0, h - PATCH_SIZE + 1, int(PATCH_SIZE * (1 - OVERLAP))):
                patch = img[start_i:start_i + PATCH_SIZE, start_j:start_j + PATCH_SIZE]
                patches.append((patch, start_i, start_j))

        if (h - PATCH_SIZE) % int(PATCH_SIZE * (1 - OVERLAP)) != 0 and (w - PATCH_SIZE) % int(PATCH_SIZE * (1 - OVERLAP)) != 0:
            start_i = h - PATCH_SIZE
            start_j = w - PATCH_SIZE
            patch = img[start_i:start_i + PATCH_SIZE, start_j:start_j + PATCH_SIZE]
            patches.append((patch, start_i, start_j))
        
        # Run inference on each patch and reconstruct the full image
        reconstructed_logits = np.zeros((num_classes,h, w), dtype=np.float32)
        sample_counts = np.zeros((h, w), dtype=np.int8)   

        for patch, start_i, start_j in patches:

            # Prepare the patch for inference
            patch = np.expand_dims(patch, axis=0)  # Add batch dimension
            encoded_inputs = image_processor(patch, return_tensors="pt")
            encoded_inputs["pixel_values"] = encoded_inputs["pixel_values"].squeeze()

            # Run inference on the patch
            with torch.no_grad():
                outputs = model(pixel_values=encoded_inputs["pixel_values"].unsqueeze(0).to(device))

            # Upsample logits to match full image size
            upsampled_logits = nn.functional.interpolate(outputs.logits.cpu(), 
                                                        size=(PATCH_SIZE, PATCH_SIZE), 
                                                        mode="bilinear", 
                                                        align_corners=False)
            
            # Update reconstructed logits and sample counts
            reconstructed_logits[:, start_i:start_i + PATCH_SIZE, start_j:start_j + PATCH_SIZE] += upsampled_logits.squeeze().numpy()
            sample_counts[start_i:start_i + PATCH_SIZE, start_j:start_j + PATCH_SIZE] += 1

        # Normalize by sample counts (avoid division by zero)
        mask = sample_counts > 0
        reconstructed_logits[:, mask] /= sample_counts[mask]

        # Save the reconstructed logits
        # output_path = os.path.join(cfg['OUTPUT_PATH']["inference_path"], image.replace(".jpg", ".npy"))
        # np.save(output_path, reconstructed_logits)

        predicted_segmentation_map = np.argmax(reconstructed_logits, axis=0)

        h, w = predicted_segmentation_map.shape
        color_seg = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[predicted_segmentation_map == label, :] = color

        reconstructed_segmentation = color_seg

        # Save the reconstructed segmentation
        output_label_path = os.path.join(cfg['OUTPUT_PATH']["results_path"], f"reconstructed_segmentation_{image.split('.')[0]}.png")
        plt.imsave(output_label_path, reconstructed_segmentation.astype(np.uint8))

        if args.use_ground_truths:
            # Append results
            all_predictions.append(predicted_segmentation_map)
            all_references.append(ground_truth)

        if args.use_ground_truths:

            ground_truth_map = np.zeros((h, w, 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                ground_truth_map[ground_truth == label, :] = color
            plt.figure(figsize=(5,10))
            plt.subplot(3, 1, 1)
            plt.imshow((img * 255).astype(np.uint8))
            plt.title("Original Image")
            plt.subplot(3, 1, 2)
            plt.imshow(color_seg)
            plt.title("Predicted Segmentation")
            plt.subplot(3, 1, 3)
            plt.imshow(ground_truth_map)
            plt.title("Ground Truth Segmentation")

            output_path = os.path.join(cfg['OUTPUT_PATH']["results_path"], f"comparison_{image.split('.')[0]}.png")
            plt.savefig(output_path)
            plt.close()

        else:
            plt.figure(figsize=(5,10))
            plt.subplot(2, 1, 1)
            plt.imshow((img * 255).astype(np.uint8))
            plt.title("Original Image")
            plt.subplot(2, 1, 2)
            plt.imshow(color_seg)
            plt.title("Predicted Segmentation")
        
            output_path = os.path.join(cfg['OUTPUT_PATH']["results_path"], f"comparison_{image.split('.')[0]}.png")
            plt.savefig(output_path)
            plt.close()
    
    if args.use_ground_truths:
        metrics = metric._compute(
                            predictions=np.array(all_predictions),
                            references=np.array(all_references),
                            num_labels=len(id2label),
                            ignore_index=255,
                            reduce_labels=False,
                        )
        
        precisions = prec._compute(
                            predictions=np.array(all_predictions).flatten(),
                            references=np.array(all_references).flatten(),
                            average = 'macro'
                        )    
        recalls = rec._compute(
                            predictions=np.array(all_predictions).flatten(),
                            references=np.array(all_references).flatten(),
                            average = 'macro'
                        )
        
        # Log metrics
        print("Mean IoU:", metrics["mean_iou"])
        print("Precision:", precisions["precision"])
        print("Recall:", recalls["recall"])

        metrics_serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in metrics.items()}
        with open(os.path.join(cfg['OUTPUT_PATH']["results_path"], "metrics.json"), "w") as f:
            json.dump(metrics_serializable, f, indent=4)
   
