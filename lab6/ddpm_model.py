# ddpm_model_with_cross_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GEGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * F.gelu(x2)

class CrossAttentionBlock2d(nn.Module):
    def __init__(self, channels, cond_dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(GEGLU(channels, channels))
        self.cond_proj = nn.Linear(cond_dim, channels)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x_norm = self.norm1(x)

        cond_kv = self.cond_proj(cond).unsqueeze(1)
        attn_out, _ = self.attn(query=x_norm, key=cond_kv, value=cond_kv)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

class SelfAttentionBlock2d(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(GEGLU(channels, channels))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x_norm = self.norm1(x)

        attn_out, _ = self.attn(query=x_norm, key=x_norm, value=x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

class LearnableTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=1000):
        super().__init__()
        self.embedding = nn.Embedding(max_period, dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, t):
        return self.embedding(t)

class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.fc = GEGLU(cond_dim, out_dim)

    def forward(self, cond):
        return self.fc(cond)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, use_dropout=False):
        super().__init__()
        self.time_mlp = GEGLU(embed_dim, out_channels)
        self.cond_mlp = GEGLU(embed_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU()
        )
        block2_layers = [
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels)
        ]
        if use_dropout:
            block2_layers.append(nn.Dropout2d(p=0.1))
        self.block2 = nn.Sequential(*block2_layers)

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.block1(x)
        time_feature = self.time_mlp(t_emb)[:, :, None, None]
        cond_feature = self.cond_mlp(c_emb)[:, :, None, None]
        h = h + time_feature + cond_feature
        h = self.block2(h)
        return F.gelu(h + self.res_conv(x))

class SimpleConditionalUNet(nn.Module):
    def __init__(self, cond_dim, time_dim=256):
        super().__init__()
        base_channels = 64
        self.time_dim = time_dim
        self.time_embed = LearnableTimeEmbedding(time_dim)
        self.cond_embed = ConditionEmbedding(cond_dim, time_dim)

        self.start = nn.Sequential(
            nn.Conv2d(3, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
        )

        self.enc1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.attn_e1 = CrossAttentionBlock2d(base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=2, stride=2)

        self.enc2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.attn_e2 = CrossAttentionBlock2d(base_channels * 2, time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=2, stride=2)

        self.enc3 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim, use_dropout=True)
        self.attn_e3 = CrossAttentionBlock2d(base_channels * 4, time_dim)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=2, stride=2)

        self.mid = ResidualBlock(base_channels * 8, base_channels * 8, time_dim, use_dropout=True)
        self.mid_self_attn = SelfAttentionBlock2d(base_channels * 8)
        self.mid_cross_attn = CrossAttentionBlock2d(base_channels * 8, time_dim)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4, time_dim, use_dropout=True)
        self.attn_d3 = CrossAttentionBlock2d(base_channels * 4, time_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2, time_dim)
        self.attn_d2 = CrossAttentionBlock2d(base_channels * 2, time_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, time_dim)
        self.attn_d1 = CrossAttentionBlock2d(base_channels, time_dim)

        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, 3, 3, padding=1),
        )

    def forward(self, x, t, cond):
        t_emb = self.time_embed(t.long())
        c_emb = self.cond_embed(cond)

        e = self.start(x)

        e1 = self.enc1(e, t_emb, c_emb)
        e1 = self.attn_e1(e1, c_emb)
        down1 = self.down1(e1)

        e2 = self.enc2(down1, t_emb, c_emb)
        e2 = self.attn_e2(e2, c_emb)
        down2 = self.down2(e2)

        e3 = self.enc3(down2, t_emb, c_emb)
        e3 = self.attn_e3(e3, c_emb)
        down3 = self.down3(e3)

        mid = self.mid(down3, t_emb, c_emb)
        mid = self.mid_self_attn(mid)
        mid = self.mid_cross_attn(mid, c_emb)

        d3 = self.up3(mid)
        d3 = torch.cat([d3, e3], dim=1)    
        d3 = self.dec3(d3, t_emb, c_emb)
        d3 = self.attn_d3(d3, c_emb)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)  
        d2 = self.dec2(d2, t_emb, c_emb)
        d2 = self.attn_d2(d2, c_emb)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)  
        d1 = self.dec1(d1, t_emb, c_emb)
        d1 = self.attn_d1(d1, c_emb)

        return self.final(d1)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    bs = t.shape[0]
    t = t.view(-1)
    out = a[t]
    return out.view(bs, *((1,) * (len(x_shape) - 1))).to(t.device)