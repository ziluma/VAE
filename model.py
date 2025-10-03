import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops.layers.torch import Rearrange

@dataclass
class VAEConfig:
    latent_dim = 128
    hidden_dim = 256
    ch_img = 1
    img_size = 28
    ch_encs = [32, 64, 128]
    ch_decs = [128, 64, 32]
    n_gp = 8
    dropout = 0.1
    act = nn.ReLU


def conv_block(in_ch, out_ch, kernel_size, stride, pad=1, transpose=False, act=nn.ReLU):
    conv = nn.ConvTranspose2d if transpose else nn.Conv2d
    return nn.Sequential(
        conv(in_ch, out_ch, kernel_size, stride, padding=pad),
        act()
    )


class Encoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            conv_block(config.ch_img, config.ch_encs[0], 4, 2, act=config.act),
            conv_block(config.ch_encs[0], config.ch_encs[1], 4, 2, act=config.act),
            conv_block(config.ch_encs[1], config.ch_encs[2], 3, 1, act=config.act),
            nn.Flatten(),
        )
        out_size = (config.img_size // 4)**2 * config.ch_encs[2]
        self.c_mean = nn.Linear(out_size, config.latent_dim)
        self.c_logvar = nn.Linear(out_size, config.latent_dim)
    
    def forward(self, x):
        h = self.net(x)
        mean = self.c_mean(h)
        logvar = self.c_logvar(h)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        C, W = config.ch_decs[0], config.img_size // 4
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, C * W * W),
            config.act(),
            Rearrange('b (c h w) -> b c h w', c=C, h=W, w=W),
            conv_block(config.ch_decs[0], config.ch_decs[1], 3, 1, act=config.act),
            conv_block(config.ch_decs[1], config.ch_decs[2], 4, 2, transpose=True, act=config.act),
            conv_block(config.ch_decs[2], config.ch_img, 4, 2, transpose=True, act=nn.Identity),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.net(z)
    

class VAE(nn.Module):
    def __init__(self, config: VAEConfig, device):
        super().__init__()
        self.encoder = Encoder(config).to(device)
        self.decoder = Decoder(config).to(device)
    
    def sample(self, mu, logvar):
        ''' q(z | x) = N(z | mu, var) '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sample(mu, logvar)
        x_recon = self.decoder(z)

        loss_recon = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.shape[0]
        loss_KL = .5 * torch.sum(mu**2 + logvar.exp() - logvar) / x.shape[0]

        return loss_recon + loss_KL

