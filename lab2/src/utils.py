import torch

def dice_score(pred_mask, gt_mask, smooth=1e-6):
    pred_mask = (pred_mask > 0.5).float()
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    return (2. * intersection + smooth) / (union + smooth)
