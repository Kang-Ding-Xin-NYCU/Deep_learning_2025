import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2)
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.args = args
        self.beta = 0
        self.current_epoch = current_epoch
        self.total_epoch = args.num_epoch

        if self.args.kl_anneal_type == 'Cyclical':
            self.cycle = self.frange_cycle_linear(self.total_epoch, start=0.0, stop=self.args.kl_anneal_ratio,
                                                  n_cycle=self.args.kl_anneal_cycle,
                                                  ratio=self.args.kl_anneal_ratio)
        elif self.args.kl_anneal_type == 'Monotonic':
            self.cycle = np.linspace(0.0, self.args.kl_anneal_ratio, self.total_epoch)
        else:
            self.cycle = [self.args.kl_anneal_ratio] * self.total_epoch

    def update(self):
        self.current_epoch += 1
    
    def get_beta(self):
        return self.cycle[min(self.current_epoch, len(self.cycle)-1)]

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = int(period * ratio)
        for c in range(n_cycle):
            v = np.linspace(start, stop, step)
            L[int(c*period):int(c*period)+step] = v
        return L
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[50, 100, 150], gamma=0.75)
        self.kl_annealing = kl_annealing(args, current_epoch=1)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 1
        
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        self.best_psnr = float('-inf')
        self.tfr_record = []
        self.loss_record = []
 
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

        import matplotlib.pyplot as plt

        plt.figure()
        epochs = list(range(1, len(self.tfr_record)+1))
        plt.plot(epochs, self.loss_record, label="Validation Loss", color='blue')
        plt.plot(epochs, self.tfr_record, label="Teacher Forcing Ratio", color='red')

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Teacher Forcing Ratio vs Validation Loss")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.args.save_root, f"{self.args.kl_anneal_type}_tfr_loss_curve.png")
        plt.savefig(save_path)

        plt.figure()
        plt.plot(epochs, self.loss_record, color='orange', label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss Curve")
        plt.grid(True)
        plt.legend()
        loss_curve_path = os.path.join(self.args.save_root, f"{self.args.kl_anneal_type}_loss_curve.png")
        plt.savefig(loss_curve_path)


    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        total_loss = 0
        total_loss = 0
        total_psnr = 0
        count = 0

        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, decoded_frames, gt_frames = self.val_one_step(img, label)
            total_loss += loss.item()
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

            pred = torch.stack(decoded_frames, dim=1)
            gt   = torch.stack(gt_frames, dim=1)

            psnr = Generate_PSNR(pred, gt)
            total_psnr += psnr.item()
            count += 1
    
        avg_loss = total_loss / len(val_loader)
        avg_psnr = total_psnr / count
        self.tfr_record.append(self.tfr)
        self.loss_record.append(avg_loss)

        print(f"Validation Loss @ Epoch {self.current_epoch}: {avg_loss:.4f}")
        print(f"Validation PSNR  @ Epoch {self.current_epoch}: {avg_psnr:.2f} dB")

        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            best_path = os.path.join(self.args.save_root, f"BEST_epoch={self.current_epoch}.ckpt")
            self.save(best_path)

    def training_one_step(self, img, label, adapt_TeacherForcing):
        img = img.to(self.args.device)
        label = label.to(self.args.device)
        self.optim.zero_grad()
        B = img.shape[0]

        total_mse = 0
        total_kld = 0
        decoded_frames = [img[:, 0]]

        for t in range(1, self.train_vi_len):
            prev_frame = decoded_frames[-1] if not adapt_TeacherForcing else img[:, t-1]
            frame_feat = self.frame_transformation(prev_frame)
            label_feat = self.label_transformation(label[:, t])
            z, mu, logvar = self.Gaussian_Predictor(frame_feat, label_feat)
            decoder_input = self.Decoder_Fusion(frame_feat, label_feat, z)
            output = self.Generator(decoder_input)

            decoded_frames.append(output)
            total_mse += self.mse_criterion(output, img[:, t])
            total_kld += kl_criterion(mu, logvar, B)

        beta = self.kl_annealing.get_beta()
        loss = total_mse + beta * total_kld
        loss.backward()
        self.optimizer_step()
        return loss
    
    def val_one_step(self, img, label):
        B = img.shape[0]
        total_mse = 0
        decoded_frames = [img[:, 0]]
        gt_frames = [img[:, 0]]

        for t in range(1, self.val_vi_len):
            prev_frame = decoded_frames[-1]
            frame_feat = self.frame_transformation(prev_frame)
            label_feat = self.label_transformation(label[:, t])
            z, mu, logvar = self.Gaussian_Predictor(frame_feat, label_feat)
            decoder_input = self.Decoder_Fusion(frame_feat, label_feat, z)
            output = self.Generator(decoder_input)

            decoded_frames.append(output)
            gt_frames.append(img[:, t])
            total_mse += self.mse_criterion(output, img[:, t])

        return total_mse, decoded_frames, gt_frames
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(0, self.tfr - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)

            user_lr = self.args.lr

            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)

            if user_lr != checkpoint['lr']:
                for param_group in self.optim.param_groups:
                    param_group['lr'] = user_lr
                self.args.lr = user_lr

            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10, 20], gamma=0.75)
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=1)
    parser.add_argument('--lr',            type=float,  default=5e-5,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=8)
    parser.add_argument('--num_epoch',     type=int, default=200,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=5,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=30,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=16,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=256,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.05,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.5,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=50,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
