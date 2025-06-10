# dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json

class ICLEVRDataset(Dataset):
    def __init__(self, image_dir, json_file, object_json, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        with open(object_json, 'r') as f:
            self.obj2idx = json.load(f)
        
        self.idx2obj = {v: k for k, v in self.obj2idx.items()}
        self.labels = list(self.data.values())
        self.filenames = list(self.data.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(image_path).convert("RGB")
        label = torch.zeros(len(self.obj2idx))
        for obj in self.labels[idx]:
            label[self.obj2idx[obj]] = 1
        if self.transform:
            image = self.transform(image)
        return image, label
