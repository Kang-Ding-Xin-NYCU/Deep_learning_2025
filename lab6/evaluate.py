# evaluate.py
import torch
import os
import json
from PIL import Image
from torchvision import transforms
from evaluator import evaluation_model

def evaluate_images(image_dir, label_json_path, object_json_path, device):
    with open(label_json_path, 'r') as f:
        data = json.load(f)

    with open(object_json_path, 'r') as f:
        obj2idx = json.load(f)

    evaluator = evaluation_model()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    total = 0
    correct = 0
    for i, objects in enumerate(data):
        fname = f"{i}"
        img_path = os.path.join(image_dir, fname + ".png")
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        label = torch.zeros((1, len(obj2idx)), device=device)
        for obj in objects:
            if obj in obj2idx:
                label[0, obj2idx[obj]] = 1

        acc = evaluator.eval(img, label)
        total += 1
        correct += acc

    print(f"Accuracy: {correct / total:.4f}")
    return correct / total