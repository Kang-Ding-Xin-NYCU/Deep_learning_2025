# main.py
import argparse
import torch
import os
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import ICLEVRDataset
from ddpm_model import SimpleConditionalUNet
from train import train_ddpm
from sample import generate_from_json, sample_ddpm, sample_denoising_grid, save_grid_and_individual_images
from evaluate import evaluate_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'sample', 'eval', 'denoise'])
    parser.add_argument('--image_dir', type=str, default='./images')
    parser.add_argument('--json_path', type=str, default='./test.json')
    parser.add_argument('--object_path', type=str, default='./objects.json')
    parser.add_argument('--save_path', type=str, default='./output/sample.png')
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--checkpoint', type=str, default='./ckpt/model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--guidance_scale', type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cond_dim = 24
    model = SimpleConditionalUNet(cond_dim=cond_dim, time_dim=256).to(device)

    if args.mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ICLEVRDataset(image_dir=args.image_dir, json_file=args.json_path, object_json=args.object_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        model = train_ddpm(model, dataloader, num_timesteps=args.timesteps, epochs=args.epochs, device=device, guidance_scale=args.guidance_scale, checkpoint_path=os.path.dirname(args.checkpoint))
        torch.save(model.state_dict(), args.checkpoint)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        with open(args.object_path, 'r') as f:
            obj2idx = json.load(f)
        generate_from_json(model, './test.json', obj2idx, './output/test', './output/test_grid.png', cond_dim, args.timesteps, args.image_size, device, args.guidance_scale)
        generate_from_json(model, './new_test.json', obj2idx, './output/new_test', './output/new_test_grid.png', cond_dim, args.timesteps, args.image_size, device, args.guidance_scale)

    elif args.mode == 'eval':
        acc1 = evaluate_images('./output/test', './test.json', args.object_path, device)
        acc2 = evaluate_images('./output/new_test', './new_test.json', args.object_path, device)
        print(f"Evaluation accuracy: test = {acc1:.4f}, new_test = {acc2:.4f}")

    elif args.mode == 'denoise':
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        with open(args.object_path, 'r') as f:
            obj2idx = json.load(f)
        save_path = os.path.join(args.save_dir, "denoising_grid.png")
        label_list = ["red sphere", "cyan cylinder", "cyan cube"]
        sample_denoising_grid(model, label_list, obj2idx, save_path, device, args.timesteps, args.image_size, args.guidance_scale)

if __name__ == '__main__':
    main()
