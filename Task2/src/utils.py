import torch
import numpy as np
from scipy.linalg import sqrtm


def cholesky_to_density(L):
    """
    Converts a lower triangular matrix L into a valid density matrix rho.
    Formula: rho = (L * L_dagger) / Tr(L * L_dagger)

    Ensures rho is:
    1. Hermitian (by construction L * L_dagger)
    2. Positive Semi-Definite (by construction)
    3. Unit Trace (by normalization)

    Args:
        L (torch.Tensor): Output from the neural network.
                          Shape: (batch_size, dim, dim) - Complex valued

    Returns:
        rho (torch.Tensor): Valid density matrices.
    """

    # We must zero out the upper triangle, or it's not Cholesky decomposition.
    mask = torch.tril(torch.ones_like(L))
    L = L * mask

    # Compute unnormalized rho = L * L_dagger
    # torch.transpose(..., 1, 2) swaps the last two dimensions (matrix transpose)
    # .conj() takes the complex conjugate
    L_dagger = torch.transpose(L.conj(), 1, 2)
    rho_unnormalized = torch.bmm(L, L_dagger)  # bmm = Batch Matrix Multiplication

    # Compute Trace for normalization
    # einsum 'bii->b' extracts the diagonal elements (ii) for each batch (b) and sums them
    trace = torch.einsum("bii->b", rho_unnormalized).real

    # Reshape trace to (batch_size, 1, 1) so we can divide the matrix by it
    trace = trace.view(-1, 1, 1)

    # Normalize
    # Add a tiny epsilon (1e-10) to avoid division by zero errors during training
    rho = rho_unnormalized / (trace + 1e-10)

    return rho


def compute_fidelity(rho_pred, rho_true):
    """
    Computes Quantum Fidelity between two density matrices.
    F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2

    Note: This is expensive to compute in a training loop.
    For reporting, we use a simplified version or rely on Qiskit/Numpy for validation.

    Here, we assume rho_pred and rho_true are standard Numpy arrays (complex64)
    for evaluation after training.
    """
    if isinstance(rho_pred, torch.Tensor):
        rho_pred = rho_pred.detach().cpu().numpy()
    if isinstance(rho_true, torch.Tensor):
        rho_true = rho_true.detach().cpu().numpy()

    fidelities = []

    for i in range(len(rho_pred)):
        r = rho_pred[i]
        t = rho_true[i]

        # F = (Tr(sqrt(sqrt(r) * t * sqrt(r))))^2
        # This check sees if t is a pure state (t^2 = t).
        is_pure = np.allclose(t @ t, t, atol=1e-5)

        if is_pure:
            # Mathematical Shortcut: If 't' is pure, Fidelity = Tr(r * t)
            # This is roughly 500x faster than sqrtm.
            f = np.real(np.trace(r @ t))
        else:
            # We only use your original heavy math if the state is mixed.
            try:
                sqrt_r = sqrtm(r)
                term = sqrt_r @ t @ sqrt_r
                sqrt_term = sqrtm(term)
                f = np.real(np.trace(sqrt_term)) ** 2
            except ValueError:
                f = 0.0

        f = np.clip(f, 0.0, 1.0)
        fidelities.append(f)

    return np.mean(fidelities)


def compute_trace_distance(rho_pred, rho_true):
    """
    Computes the Trace Distance between two density matrices.
    T(rho, sigma) = 0.5 * Tr|rho - sigma|

    Args:
        rho_pred (torch.Tensor or np.ndarray): Predicted states
        rho_true (torch.Tensor or np.ndarray): Ground truth states

    Returns:
        float: Mean Trace Distance across the batch (0.0 = perfect match, 1.0 = max error)
    """
    if isinstance(rho_pred, torch.Tensor):
        rho_pred = rho_pred.detach().cpu().numpy()
    if isinstance(rho_true, torch.Tensor):
        rho_true = rho_true.detach().cpu().numpy()

    distances = []

    for i in range(len(rho_pred)):
        # Calculate Difference Matrix
        diff = rho_pred[i] - rho_true[i]

        # Calculate Eigenvalues of the Hermitian difference matrix
        # For Hermitian matrices, Singular Values = Abs(Eigenvalues)
        eigvals = np.linalg.eigvalsh(diff)

        # Trace Distance = 0.5 * Sum(|Eigenvalues|)
        dist = 0.5 * np.sum(np.abs(eigvals))
        distances.append(dist)

    return np.mean(distances)
