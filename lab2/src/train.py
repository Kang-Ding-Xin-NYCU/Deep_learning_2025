import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from oxford_pet import load_dataset, visualize_samples
from evaluate import evaluate
from torchsummary import summary
import albumentations as A

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/models")

from unet import UNet
from resnet34_unet import ResNet34UNet

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "unet":
        model = UNet()
    else:
        model = ResNet34UNet()
    model.to(device)
    summary(model, (3, 256, 256))

    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-45, 45), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], additional_targets={"trimap": "mask"})
    
    valid_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], additional_targets={"trimap": "mask"})
    
    train_dataset = load_dataset(args.data_path, "train", train_transform)
    valid_dataset = load_dataset(args.data_path, "valid", valid_transform)

    os.makedirs("./check/train", exist_ok=True)
    os.makedirs("./check/valid", exist_ok=True)

    visualize_samples(train_dataset, 25, "./check/train")
    visualize_samples(valid_dataset, 25, "./check/valid")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses, valid_losses, valid_dices = [], [], []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        total_loss = 0

        for batch in valid_dataloader:
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
        
        avg_valid_loss = total_loss / len(valid_dataloader)

        valid_dice = evaluate(model, valid_dataloader)
        if torch.is_tensor(valid_dice):
            valid_dice = valid_dice.cpu().numpy()
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        valid_dices.append(valid_dice)

        print(f"Epoch {epoch+1}/{args.epochs}, Train_Loss: {avg_train_loss:.4f}, Valid_Loss: {avg_valid_loss:.4f}, Valid Dice: {valid_dice:.4f}")
    
    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs("./result/fig", exist_ok=True)

    valid_dices = np.array([d.cpu().numpy() if torch.is_tensor(d) else d for d in valid_dices])

    torch.save(model.state_dict(), f"./saved_models/{args.model}.pth")
    print("Model saved!")

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs+1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, args.epochs+1), valid_losses, marker='x', label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs+1), valid_dices, marker='o', label='Valid Dice Score', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Dice Score vs Epoch')
    plt.legend()

    plt.savefig(f"./result/fig/loss_dice_plot_{args.model}.png")

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks')
    parser.add_argument('--data_path', type=str, required=True, help='Path of the dataset')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model', choices=['unet', 'resnet'], required=True, help='Model type')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
