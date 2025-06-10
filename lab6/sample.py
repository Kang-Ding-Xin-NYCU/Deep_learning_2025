# sample.py
import torch
import os
import json
from torchvision.utils import save_image, make_grid
from ddpm_model import extract, linear_beta_schedule

@torch.no_grad()
def sample_ddpm(model, cond, num_timesteps, image_size, device, guidance_scale=2.0):
    model.eval()
    batch_size = cond.size(0)
    img = torch.randn((batch_size, 3, image_size, image_size), device=device)

    betas = linear_beta_schedule(num_timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), alphas_cumprod[:-1]], dim=0)

    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod + 1e-8)

    for t in reversed(range(num_timesteps)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        pred_noise_cond = model(img, t_tensor, cond)
        pred_noise_uncond = model(img, t_tensor, torch.zeros_like(cond))
        pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

        coeff1 = extract(sqrt_recip_alphas, t_tensor, img.shape)
        coeff2 = extract(betas / sqrt_one_minus_alphas_cumprod, t_tensor, img.shape)
        model_mean = coeff1 * (img - coeff2 * pred_noise)

        if t > 0:
            noise = torch.randn_like(img)
            variance = extract(posterior_variance, t_tensor, img.shape)
            img = model_mean + torch.sqrt(variance) * noise
        else:
            img = model_mean

    img = (img.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]
    return img

def save_grid_and_individual_images(img, save_dir, grid_path):
    os.makedirs(save_dir, exist_ok=True)
    grid_img = make_grid(img, nrow=8, padding=2)
    save_image(grid_img, grid_path)
    for i in range(img.size(0)):
        save_image(img[i], os.path.join(save_dir, f"{i}.png"))
    print(f"Saved grid to {grid_path} and images to {save_dir}")

@torch.no_grad()
def sample_denoising_grid(model, label_list, obj2idx, save_path, device, timesteps=1000, image_size=64, guidance_scale=2.0):
    model.eval()
    cond_dim = 24
    cond = torch.zeros(1, cond_dim).to(device)
    for obj in label_list:
        cond[0, obj2idx[obj]] = 1

    img = torch.randn((1, 3, image_size, image_size), device=device)

    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), alphas_cumprod[:-1]], dim=0)

    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    steps = []
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

        pred_noise_cond = model(img, t_tensor, cond)
        pred_noise_uncond = model(img, t_tensor, torch.zeros_like(cond))
        pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

        coeff1 = sqrt_recip_alphas[t]
        coeff2 = betas[t] / sqrt_one_minus_alphas_cumprod[t]
        model_mean = coeff1 * (img - coeff2 * pred_noise)

        if t > 0:
            noise = torch.randn_like(img)
            variance = posterior_variance[t]
            img = model_mean + torch.sqrt(variance) * noise
        else:
            img = model_mean

        if t % (timesteps // 11) == 0:
            steps.append(img.clone().squeeze(0))

    imgs = torch.stack(steps)
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    grid = make_grid(imgs, nrow=6)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(grid, save_path)
    print(f"Denoising grid saved to {save_path}")

def generate_from_json(model, json_path, obj2idx, save_dir, grid_path, cond_dim, num_timesteps, image_size, device, guidance_scale):
    with open(json_path, 'r') as f:
        data = json.load(f)

    cond_list = []
    for label_list in data[:32]:
        onehot = torch.zeros(cond_dim).to(device)
        for obj in label_list:
            onehot[obj2idx[obj]] = 1
        cond_list.append(onehot)
    cond_tensor = torch.stack(cond_list)

    img = sample_ddpm(model, cond_tensor, num_timesteps, image_size, device, guidance_scale)
    save_grid_and_individual_images(img, save_dir, grid_path)

    denoise_path = os.path.join(save_dir, f"denoise_{os.path.basename(save_dir)}.png")
    sample_denoising_grid(model, data[0], obj2idx, denoise_path, device, num_timesteps, image_size, guidance_scale)