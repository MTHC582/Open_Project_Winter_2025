import torch
import torch.nn as nn
from src.utils import cholesky_to_density


class DensityMatrixReconstructor(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[64, 64], system_dim=2):
        super().__init__()
        self.system_dim = system_dim

        # TODO: Calculate output dimension.
        # We need 4 real numbers to build a 2x2 complex lower triangular matrix.
        # L = [[a, 0], [b+jc, d]] -> variables are a, b, c, d, Hence,,
        self.output_dim = 4

        # TODO: Define the MLP layers
        # Use nn.Linear and nn.ReLU
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        # Final output layer
        layers.append(nn.Linear(in_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Batch of Pauli measurements (batch_size, 3)
        Returns:
            L (torch.Tensor): Lower triangular complex matrices (batch_size, 2, 2)
        """
        batch_size = x.size(0)

        # 1. Get raw real-valued output (batch, 4)
        raw = self.net(x)

        # 2. Reshape into Complex Matrix form L
        # Initialize zero matrix (complex)
        L = torch.zeros(
            (batch_size, self.system_dim, self.system_dim),
            dtype=torch.complex64,
            device=x.device,
        )

        # TODO: Map raw outputs to L elements
        # L[0,0] is real -> use raw[:, 0]
        # L[1,1] is real -> use raw[:, 1]
        # L[1,0] is complex -> use raw[:, 2] (real part) + 1j * raw[:, 3] (imag part)

        # Mappin logic of mine, type casting with complex64 just to be sure of no errors

        L[:, 0, 0] = raw[:, 0]
        L[:, 1, 1] = raw[:, 1]
        L[:, 1, 0] = 0.1 * (raw[:, 2] + 1j * raw[:, 3])  # just scaling nothong muchhh
        # Making the system like somethoing close to the diagonal matrix

        rho = cholesky_to_density(L)

        return rho
