import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size: int, z_dim):
        super().__init__()
        self._linear1 = nn.Linear(input_size, input_size // 2)
        self._linear2 = nn.Linear(input_size // 2, input_size // 4)
        self._mu = nn.Linear(input_size // 4, z_dim) # \mu
        self._log_sigma2 = nn.Linear(input_size // 4, z_dim) # log(\simga^2)

    def forward(self, inputs: torch.Tensor):
        outputs = self._linear1(inputs)
        outputs = nn.functional.relu(outputs)
        outputs = self._linear2(outputs)
        outputs = nn.functional.relu(outputs)
        return self._mu(outputs), self._log_sigma2(outputs)


class Decoder(nn.Module):
    def __init__(self, output_size: int, z_dim):
        super().__init__()
        self._linear1 = nn.Linear(z_dim, z_dim * 2)
        self._linear2 = nn.Linear(z_dim * 2, z_dim * 4)
        self._mu = nn.Linear(z_dim * 4, output_size)
        self._log_sigma2 = nn.Linear(z_dim * 4, output_size)

    def forward(self, inputs: torch.Tensor):
        outputs = self._linear1(inputs)  # Bxz_dim -> Bx(z_dim*2)
        outputs = nn.functional.relu(outputs)
        outputs = self._linear2(outputs)
        outputs = nn.functional.relu(outputs)
        return self._mu(outputs), self._log_sigma2(outputs)

class VAE(nn.Module):
    def __init__(self, input_size: int, z_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(input_size=input_size, z_dim=z_dim)
        self.decoder = Decoder(output_size=input_size, z_dim=z_dim)
        self._input_size = input_size
        self._z_dim = z_dim

    def compute_loss(self, inputs: torch.Tensor, outputs: torch.Tensor,
                     z_mu: torch.Tensor, z_log_sigma_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = torch.nn.functional.mse_loss(inputs, outputs)
        kl_loss = -torch.mean(torch.sum(1 + z_log_sigma_2 - z_mu ** 2 - torch.exp(z_log_sigma_2), dim=-1))
        return recon_loss, kl_loss

    def forward(self, inputs: torch.Tensor):
        # inputs: BxCxHxW
        inputs = inputs.view(inputs.size(0), -1)  # BxCxHxW -> Bx1
        # The forward pass of the VAE takes a real sample as input,
        # then generates its reconstructed version
        z_mu, z_log_sigma_2 = self.encoder(inputs)
        # Sample z from unit gaussian
        q = torch.distributions.MultivariateNormal(
            torch.zeros(inputs.size(0), self._z_dim),
            torch.eye(self._z_dim).repeat(inputs.size(0), 1, 1))
        z = q.sample()
        x_mu, x_log_sigma_2 = self.decoder(z * torch.sqrt(torch.exp(z_log_sigma_2)) + z_mu)
        # Sample x from unit gaussian
        p = torch.distributions.MultivariateNormal(
            torch.zeros(inputs.size(0), self._input_size),
            torch.eye(self._input_size).repeat(inputs.size(0), 1, 1))
        x = p.sample()
        outputs = x * torch.sqrt(torch.exp(x_log_sigma_2)) + x_mu
        recon_loss, kl_loss = self.compute_loss(inputs, outputs, z_mu, z_log_sigma_2)
        return outputs.view(inputs.size(0), 1, 28, 28), recon_loss, kl_loss

    @torch.no_grad()
    def encode(self, inputs: torch.Tensor):
        return self.encoder(inputs)

    @torch.no_grad()
    def decode(self, inputs: torch.Tensor):
        return self.decoder(inputs)
