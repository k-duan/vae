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
    vae = VAE(input_size=784, z_dim=64)
    optimizer = torch.optim.AdamW(params=vae.parameters(), lr=1e-3)

    i = 0
    for images, _ in dataloader:
        optimizer.zero_grad()
        outputs, recon_loss, kl_loss = vae(images)
        total_loss = recon_loss + kl_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        writer.add_images("train/images", images, i)
        writer.add_images("train/outputs", outputs, i)
        writer.add_scalar("train/recon_loss", recon_loss.item(), i)
        writer.add_scalar("train/kl_loss", kl_loss.item(), i)
        writer.add_scalar("train/total_loss", total_loss.item(), i)
        writer.add_scalar("train/grad_norm", grad_norm(vae.parameters()), i)
        i += 1

if __name__ == "__main__":
    main()