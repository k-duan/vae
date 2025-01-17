from datetime import datetime
from typing import Iterator

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from model import VAE


def collate_mnist_fn(data: list[tuple[Image, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.zeros((0, 1, 28, 28), dtype=torch.float32)  # BxCxHxW, where H=28, W=28, C=1 for mnist
    labels = torch.zeros((0,), dtype=torch.int64)  # Bx1
    for image, label in data:
        images = torch.cat([images, torch.tensor(np.asarray(image)).view(1, 1, 28, 28)])
        labels = torch.cat([labels, torch.tensor(label).view(-1)])
    images /= 255
    return images, labels

def grad_norm(parameters: Iterator[torch.nn.Parameter]) -> float:
   total_norm = 0.0
   for p in parameters:
      if p.grad is not None:
         param_norm = p.grad.data.norm(2)
         total_norm += param_norm.item() ** 2
   return total_norm ** 0.5

def main():
    torch.manual_seed(123)
    log_name = f"mnist-vae-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
    writer = SummaryWriter(log_dir=f"runs/{log_name}")
    dataset = torchvision.datasets.MNIST("./", train=True, download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, collate_fn=collate_mnist_fn)
    vae = VAE(input_size=784, z_dim=32, dropout=0)
    optimizer = torch.optim.AdamW(params=vae.parameters(), lr=1e-3, weight_decay=1e-4)
    n_epochs = 50

    i = 0
    for _ in range(n_epochs):
        for x, _ in dataloader:
            optimizer.zero_grad()
            x_recon, recon_loss, kl_loss, total_loss, p_mu, p_log_sigma_2, q_mu, q_log_sigma_2 = vae(x)
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            writer.add_images("train/images/x", x, i)
            writer.add_images("train/images/x_recon", x_recon, i)
            writer.add_scalar("train/loss/recon", recon_loss.item(), i)
            writer.add_scalar("train/loss/kl", kl_loss.item(), i)
            writer.add_scalar("train/loss/total", total_loss.item(), i)
            writer.add_scalar("train/model/grad_norm", grad_norm(vae.parameters()), i)
            writer.add_scalar("train/p/mu", torch.mean(p_mu), i)
            writer.add_scalar("train/p/log_sigma_2", torch.mean(p_log_sigma_2), i)
            writer.add_scalar("train/q/mu", torch.mean(q_mu), i)
            writer.add_scalar("train/q/log_sigma_2", torch.mean(q_log_sigma_2), i)
            i += 1

if __name__ == "__main__":
    main()
