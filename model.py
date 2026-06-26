import torch
import torch.nn as nn


def _svd_init(n, scale):
    W = torch.randn(n, n)
    U, _, Vh = torch.linalg.svd(W)
    return (U @ Vh) * scale


class Encoder(nn.Module):
    def __init__(self, n_x, n_z, alpha):
        super().__init__()
        width = 16 * alpha
        self.net = nn.Sequential(
            nn.Linear(n_x, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, n_z),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, n_z, n_x, alpha):
        super().__init__()
        width = 16 * alpha
        self.net = nn.Sequential(
            nn.Linear(n_z, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, n_x),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, z):
        return self.net(z)


class LRAN(nn.Module):
    """
    Linearly Recurrent Autoencoder for controlled dynamical systems.
    Learns z_{k+1} = A z_k + B u_k in a nonlinear latent space.
    """

    def __init__(self, n_x, n_u, n_z, alpha, init_scale=0.99):
        super().__init__()
        self.encoder = Encoder(n_x, n_z, alpha)
        self.decoder = Decoder(n_z, n_x, alpha)

        # A
        self.A = nn.Linear(n_z, n_z, bias=False)
        self.A.weight.data = _svd_init(n_z, init_scale)

        # B
        self.B = nn.Linear(n_u, n_z, bias=False)
        nn.init.xavier_uniform_(self.B.weight)

    def rollout(self, z0, us):
        """
        Propagate latent state forward under control inputs.

        z0 : (batch, n_z)    initial latent state
        us : (batch, K, n_u) control sequence
        Returns list of K predicted latent states.
        """
        z = z0
        z_preds = []
        for k in range(us.shape[1]):
            z = self.A(z) + self.B(us[:, k])
            z_preds.append(z)
        return z_preds
