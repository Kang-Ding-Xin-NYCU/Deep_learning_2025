import os
import math
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.device = args.device
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            images = batch.to(self.device)
            logits, targets, mask = self.model(images)

            logits = logits[mask]
            targets = targets[mask]
            loss = F.cross_entropy(logits, targets)
            loss.backward()

            self.optim.step()
            self.optim.zero_grad()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        return total_loss / len(loader)

    def eval_one_epoch(self, loader, epoch):
        self.model.eval()
        total_loss = 0
        os.makedirs("eval_outputs", exist_ok=True)
        pred_images = []

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {epoch} [Eval]")
            for idx, batch in enumerate(pbar):
                images = batch.to(self.device)
                logits, targets, mask = self.model(images)
                logits = logits[mask]
                targets = targets[mask]

                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

                if idx < 10:
                    z_indices = self.model.encode_to_z(images)
                    z_indices = z_indices.view(images.size(0), -1)

                    B, N = z_indices.shape
                    rand_mask = torch.rand_like(z_indices.float()) < 0.4

                    masked_input = z_indices.clone()
                    masked_input[rand_mask] = self.model.mask_token_id

                    pred_logits = self.model.transformer(masked_input)
                    pred_indices = pred_logits.argmax(dim=-1)

                    z_q = self.model.vqgan.codebook.embedding(pred_indices)
                    z_q = z_q.view(-1, 16, 16, 256).permute(0, 3, 1, 2)
                    decoded_image = self.model.vqgan.decode(z_q)

                    mean = torch.tensor([0.4868, 0.4341, 0.3844], device=self.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.2620, 0.2527, 0.2543], device=self.device).view(1, 3, 1, 1)
                    decoded_image = decoded_image * std + mean

                    pred_images.append(decoded_image[0].cpu())

            if pred_images:
                grid = make_grid(pred_images, nrow=5, normalize=True, padding=2)
                save_image(grid, f"eval_outputs/epoch_{epoch:03d}.png")

        return total_loss / len(loader)

    @staticmethod
    def warmup_cosine_schedule(optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = self.warmup_cosine_schedule(optimizer, warmup_steps=500, total_steps=10000)
        return optimizer, scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./transformer_checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=4, help='Number for gradient accumulation.')

    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.002, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader, epoch)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.transformer.state_dict(), f'transformer_checkpoints/0.3/epoch_{epoch}.pt')
