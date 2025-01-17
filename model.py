import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self._linear = nn.Linear(in_dim, out_dim)
        self._bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self._linear(x)
        x = self._bn(x)
        return nn.functional.relu(x)

class FFN(nn.Module):
    def __init__(self, dims: list[int], dropout: float):
        super().__init__()
        self._blocks = nn.ModuleList([Block(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])])
        self._dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for block in self._blocks:
            x = block(x)
        return self._dropout(x)

class Encoder(nn.Module):
    def __init__(self, input_size: int, z_dim: int, dropout: float):
        super().__init__()
        self._ffn = FFN(dims=[input_size, z_dim * 4], dropout=dropout)
        self._mu = nn.Linear(z_dim * 4, z_dim) # \mu
        self._log_sigma2 = nn.Linear(z_dim * 4, z_dim) # log(\simga^2)

    def forward(self, x: torch.Tensor):
        x = self._ffn(x)
        return self._mu(x), torch.clamp(self._log_sigma2(x), min=-3, max=2)

class Decoder(nn.Module):
    def __init__(self, output_size: int, z_dim: int, dropout: float):
        super().__init__()
        self._ffn = FFN(dims=[z_dim, z_dim * 16], dropout=dropout)
        self._mu = nn.Linear(z_dim * 16, output_size) # \mu
        self._log_sigma2 = nn.Linear(z_dim * 16, output_size) # log(\simga^2)

    def forward(self, z: torch.Tensor):
        z = self._ffn(z)
        return self._mu(z), torch.clamp(self._log_sigma2(z), min=-3, max=2)

class VAE(nn.Module):
    def __init__(self, input_size: int, z_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(input_size=input_size, z_dim=z_dim, dropout=dropout)
        self.decoder = Decoder(output_size=input_size, z_dim=z_dim, dropout=dropout)
        self._input_size = input_size
        self._z_dim = z_dim

    def compute_loss(self, x: torch.Tensor, p_mu: torch.Tensor, p_log_sigma_2: torch.Tensor, z_mu: torch.Tensor, z_log_sigma_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Loss function is negative ELBO and contain two parts
        # 1) recon_loss: -log(p(x|z)).
        # In gaussian case (p(x|z)), this is the negative log likelihood of a gaussian, rather than MSE
        # Using MSE between (inputs, outputs) pairs is minimizing |x-x_recon|^2, which effectively eliminating
        # the variance term and thus only favoring the means. This cause very blurry, mean-image like, reconstructions.
        recon_loss = torch.mean(0.5 * torch.sum(p_log_sigma_2, dim=-1) + 0.5 * torch.sum((x - p_mu) ** 2 / torch.exp(p_log_sigma_2), dim=-1))
        # 2) kl_loss: KL(q(z|x) || p(z)), where the prior p(z) is assumed to be unit gaussian
        kl_loss = torch.mean(0.5 * (torch.sum(torch.exp(z_log_sigma_2) - 1 - z_log_sigma_2, dim=-1) + torch.sum(z_mu ** 2, dim=-1)))
        # total_loss: sum of the two loss terms
        total_loss = recon_loss + kl_loss
        return recon_loss, kl_loss, total_loss

    def forward(self, x: torch.Tensor):
        # inputs: BxCxHxW
        x = x.view(x.size(0), -1)  # BxCxHxW -> Bx1
        batch_size = x.size(0)
        # The forward pass of the VAE takes a real sample as input,
        # then generates its reconstructed version
        q_mu, q_log_sigma_2 = self.encoder(x)
        # Sample z from q(z|x)
        # Use torch.randn() (much faster) instead of torch.distributions.MultivariateNormal()
        z = torch.randn_like(q_mu) * torch.exp(0.5 * q_log_sigma_2) + q_mu
        p_mu, p_log_sigma_2 = self.decoder(z)
        x_recon = torch.randn_like(p_mu) * torch.exp(0.5 * p_log_sigma_2) + p_mu
        recon_loss, kl_loss, total_loss = self.compute_loss(x, p_mu, p_log_sigma_2, q_mu, q_log_sigma_2)
        return x_recon.view(batch_size, 1, 28, 28), recon_loss, kl_loss, total_loss, p_mu, p_log_sigma_2, q_mu, q_log_sigma_2
