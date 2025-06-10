import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        z = self.vqgan.encoder(x)
        z = self.vqgan.quant_conv(z)

        _, indices, _ = self.vqgan.codebook(z)

        return indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        def linear(x):
            return 1.0 - x

        def cosine(x):
            return np.cos((x * np.pi) / 2)

        def square(x):
            return 1.0 - x**2

        if mode == "linear":
            return linear
        elif mode == "cosine":
            return cosine
        elif mode == "square":
            return square
        else:
            raise NotImplementedError(f"Unknown gamma mode: {mode}")

##TODO2 step1-3:            
    def forward(self, x):
        z_indices = self.encode_to_z(x)
        z_indices = z_indices.view(x.size(0), -1)
    
        mask = torch.rand_like(z_indices.float()) < 0.3
        masked_input = z_indices.clone()
        masked_input[mask] = self.mask_token_id
    
        logits = self.transformer(masked_input)
        return logits, z_indices, mask
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, ratio):
        """
        一次 iteration 的 inpainting 解碼步驟（修正版）
        確保：原本非 mask 的 token 保留不動，從 masked 中選信心高的保留

        Args:
            z_indices: torch.LongTensor, shape (B, 256)，初始預測 token
            mask: torch.BoolTensor, shape (B, 256)，True 表示還是 mask 狀態
            ratio: float, 當前 iteration 的 ratio (例如 t/T)，用於溫度調整
        Returns:
            updated_z: torch.LongTensor, 更新後的 token 預測
            new_mask: torch.BoolTensor, 下一輪的 mask 狀態
        """
        B, N = z_indices.shape

        masked_input = z_indices.clone()
        masked_input[mask] = self.mask_token_id

        logits = self.transformer(masked_input)
        probs = torch.softmax(logits, dim=-1)

        z_indices_predict_prob, z_indices_predict = torch.max(probs, dim=-1)

        gumbel_noise = -torch.empty_like(z_indices_predict_prob).exponential_().log()
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * gumbel_noise

        keep_ratio = self.gamma(1 - ratio)
        keep_num = int(keep_ratio * N)

        keep_mask = ~mask.clone()

        masked_confidence = confidence.clone()
        masked_confidence[~mask] = float("inf")
        sorted_conf, sorted_idx = torch.sort(masked_confidence, dim=-1)

        for b in range(B):
            already_keep = keep_mask[b].sum().item()
            need_extra = max(keep_num - already_keep, 0)
            if need_extra > 0:
                keep_mask[b, sorted_idx[b, :need_extra]] = True

        new_mask = ~keep_mask

        updated_z = z_indices.clone()
        updated_z[mask] = z_indices_predict[mask]

        return updated_z, new_mask

    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
