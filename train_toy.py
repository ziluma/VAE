import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import os

from model import VAE, VAEConfig


config = VAEConfig()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
torch.manual_seed(666)

# IO
print_every = 100
sample_every = 5 # sample every ? epochs   

# training
lr = 1e-3
weight_decay = 1e-4
n_epochs = 20

transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Lambda(lambda x: x * 2.0 - 1.0),
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

vae = VAE(config, device).to(device)
optim = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)

check_dir = './ckpt'
sample_dir = './samples'

os.makedirs(sample_dir, exist_ok=True)
os.makedirs(check_dir, exist_ok=True)

global_step = 0
for epoch in range(1, n_epochs+1):
    vae.train()
    for x, _ in train_loader:
        x = x.to(device)
        loss = vae(x)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optim.step()

        if global_step % print_every == 0:
            print(f"epoch {epoch} step {global_step}: loss={loss.item():.4f}")
        global_step += 1

    with torch.no_grad():
        if epoch % sample_every != 0: continue
        samples = vae.sample()
        grid = vutils.make_grid(samples, nrow=4)
        out_path = os.path.join(sample_dir, f"samples_epoch_{epoch:03d}.png")
        vutils.save_image(grid, out_path)
        print(f"Saved {out_path}")

# final checkpoint
ckpt_path = os.path.join(check_dir, "vae_mnist.pt")
torch.save({
    "model": vae.state_dict()
}, ckpt_path)
print(f"Saved checkpoint to {ckpt_path}")


