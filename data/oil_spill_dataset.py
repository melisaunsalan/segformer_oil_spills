import os
import numpy as np
import random

from torch.utils.data import Dataset
from PIL import Image
import torch

import matplotlib.pylab as plt

class OilSpillDataset(Dataset):
  def __init__(self, root_dir, split = 'train', image_processor = None):
    self.path = root_dir
    self.split = split
    self.image_processor = image_processor
    self.id2label = {"0": "sea", "1": "oil_spill", "2": "look_alike", "3":"ship", "4":"land"}
  
    self.images = sorted(os.listdir(os.path.join(self.path, self.split, 'images'))) 

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
    # read image (JPEG/PNG) and corresponding mask (assumes same filename under masks/)
    img_name = self.images[idx]
    img_path = os.path.join(self.path, self.split, 'images', img_name)
    label_path = os.path.join(self.path, self.split, 'labels_1D', img_name.replace(".jpg", ".png"))

    # load RGB image as H x W x 3 uint8
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = img/255.0

    # load mask as single-channel (0/1 or grayscale); fallback to empty mask if missing
    if os.path.exists(label_path):
      label = Image.open(label_path)
      label = np.array(label, dtype=np.uint8)
    else:
      label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    encoded_inputs = self.image_processor(img, return_tensors="pt")
    encoded_inputs["pixel_values"] = encoded_inputs["pixel_values"].squeeze()

    # Add the original image and labels as well
    encoded_inputs["original_image"] = img
    encoded_inputs["original_labels"] = torch.from_numpy(label).long()

    return encoded_inputs
