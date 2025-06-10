# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ddpm_model import extract
from ddpm_model import linear_beta_schedule
import os
import wandb

def train_ddpm(model, dataloader, num_timesteps, epochs, device, guidance_scale = 2.0, checkpoint_path='./ckpt', 
               test_json_path='./test.json', object_json_path='./objects.json', output_dir='./outputs'):
    from sample import sample_ddpm
    from evaluate import evaluate_images
    import json

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(
        project="conditional-ddpm",
        config={
            "epochs": epochs,
            "lr": 2e-4,
            "optimizer": "Adam",
            "loss": "MSE + 0.1 * L1 + 0.01 * Cosine_Loss",
            "base_channels": 64,
            "num_heads": 8,
            "dropout": True,
            "image_size": 64
        }
    )

    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    betas = linear_beta_schedule(num_timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader)
        epoch_loss = 0

        for images, conditions in pbar:
            images = images.to(device)
            conditions = conditions.to(device)
            t = torch.randint(0, num_timesteps, (images.size(0),), device=device)

            drop_mask = (torch.rand(conditions.size(0), device=device) < 0.1).float().unsqueeze(1)
            conditions = conditions * (1 - drop_mask)

            noise = torch.randn_like(images)
            sqrt_alpha_cumprod = extract(torch.sqrt(alphas_cumprod).to(device), t, images.shape)
            sqrt_one_minus_alpha_cumprod = extract(torch.sqrt(1 - alphas_cumprod).to(device), t, images.shape)
            x_t = sqrt_alpha_cumprod * images + sqrt_one_minus_alpha_cumprod * noise

            predicted_noise = model(x_t, t, conditions)
            mse = nn.MSELoss()(predicted_noise, noise)
            l1 = nn.L1Loss()(predicted_noise, noise)
            cosine_sim = nn.CosineSimilarity(dim=1)(predicted_noise.view(predicted_noise.size(0), -1), noise.view(noise.size(0), -1)).clamp(-1, 1)
            cosine_loss = 1 - cosine_sim.mean()

            loss = mse + 0.1 * l1 + 0.01 * cosine_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
            pbar.set_description(f"Epoch {epoch+1} Loss: {loss.detach().item():.4f}")

        avg_loss = epoch_loss / len(pbar)
        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{checkpoint_path}/best.pth')
            print(f"Best model saved to {checkpoint_path}/best.pth")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{checkpoint_path}/epoch{epoch+1}.pth')
            print(f"Model saved to {checkpoint_path}/epoch{epoch+1}.pth")

        if (epoch + 1) % 20 == 0:
            model.eval()
            with open(test_json_path, 'r') as f:
                data = json.load(f)
            with open(object_json_path, 'r') as f:
                obj2idx = json.load(f)

            with torch.no_grad():
                for i, label_list in enumerate(data):
                    onehot = torch.zeros(1, 24).to(device)
                    for obj in label_list:
                        onehot[0, obj2idx[obj]] = 1
                    save_path = os.path.join(output_dir, f"{i:06d}.png")
                    sample_ddpm(model, cond=onehot, num_timesteps=num_timesteps, image_size=64,
                            save_path=save_path, device=device, guidance_scale=guidance_scale)
    
                acc = evaluate_images(output_dir, test_json_path, object_json_path, device)
                print(f"[Epoch {epoch+1}] Final Eval Accuracy: {acc:.4f}")
                wandb.log({"eval_accuracy": acc, "epoch": epoch + 1})

    wandb.finish()
    return model
