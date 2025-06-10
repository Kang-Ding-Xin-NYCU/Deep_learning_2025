import torch
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from utils import dice_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    total_dice = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().to(device)
            outputs = model(images)
            total_dice += dice_score(outputs, masks)
    
    avg_dice = total_dice / len(dataloader)
    
    return avg_dice
