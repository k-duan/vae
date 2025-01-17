# vae

Reproduce the VAE experiment on MNIST. Both encoder and decoder are assumed as Gaussian.

![Screenshot 2025-01-16 at 20.57.40.png](screenshots/Screenshot%202025-01-16%20at%2020.57.40.png)

Notes:

1. Reconstruction loss must match the assumed decoder probability distribution.
2. Use `torch.randn()` for multivariate unit Gaussian sampling (much faster than `torch.distributions.MultivariateNormal()`)
3. `log(\sigma^2)` from both encoder and decoder must be clipped.
4. Gradient norm clipping is not required as long as the norm isn't exploding.
5. KL loss is expected to increase in the beginning, but should stabilize and converge to lower values later.

Reference:

1. [The original VAE paper](https://arxiv.org/abs/1312.6114)
2. https://deepgenerativemodels.github.io/notes/vae/
