import torch
import argparse
import numpy as np
from PIL import Image
from oxford_pet import load_dataset, visualize_samples
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os
from evaluate import evaluate
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/models")

from unet import UNet
from resnet34_unet import ResNet34UNet

def inference(model_path, data_path, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "unet":
        model = UNet()
    else:
        model = ResNet34UNet()
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    test_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], additional_targets={"trimap": "mask"})
    
    test_dataset = load_dataset(data_path, "test", test_transform)

    os.makedirs("./check/test", exist_ok=True)

    visualize_samples(test_dataset, 25, "./check/test")

    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    test_dice = evaluate(model, test_dataloader)
    print(f"Test Set Dice Score: {test_dice:.4f}")
    
    os.makedirs(f"./result/infer/{model_type}", exist_ok=True)
    for i, sample in enumerate(test_dataset):
        image_np = sample["image"]
        image_tensor = torch.tensor(image_np).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor).squeeze().cpu().numpy()
        
        pred_mask = (output > 0.5).astype(np.uint8) * 255

        image_np = image_np.transpose(1, 2, 0)
        image_np = image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)

        overlay = image_np.copy()
        red_mask = np.zeros_like(overlay)
        red_mask[:, :, 0] = pred_mask
        blended = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)

        blended_img = Image.fromarray(blended)
        blended_img.save(f"./result/infer/{model_type}/infer_{i}.png")
        
    print("Inference completed. Masks saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on test images')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data_path', required=True, help='Path to test dataset')
    parser.add_argument('--model_type', choices=['unet', 'resnet'], required=True, help='Model type')
    
    args = parser.parse_args()
    inference(args.model, args.data_path, args.model_type)
