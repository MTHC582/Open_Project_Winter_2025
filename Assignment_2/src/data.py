import numpy as np
import os


def generate_random_density_matrix(dim=2):
    """
    Generates a valid random density matrix rho.
    Must be Hermitian, Positive Semi-Definite, and Trace=1.
    """
    # TODO: Generate a random complex matrix G (Ginibre ensemble)
    # G = ...
    # Generate random real and imaginary parts
    G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)

    # TODO: Construct rho = G @ G.H (G times G conjugate transpose)
    # rho = ...
    rho = G @ G.conj().T

    # TODO: Normalize rho so that trace(rho) == 1
    rho = rho / np.trace(rho)

    return rho.astype(np.complex64)  # Should be shape (dim, dim), complex64


def get_pauli_measurements(rho):
    """
    Simulates the measurement process.
    Input: Density matrix rho.
    Output: Expectation values for Pauli X, Y, Z.
    """
    # Define Pauli Matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # TODO: Calculate expectation value <O> = Real(Trace(rho @ O))
    # exp_x = ...
    # exp_y = ...
    # exp_z = ...
    exp_x = np.real(np.trace(rho @ sigma_x))
    exp_y = np.real(np.trace(rho @ sigma_y))
    exp_z = np.real(np.trace(rho @ sigma_z))

    return np.array([exp_x, exp_y, exp_z], dtype=np.float32)


def generate_dataset(num_samples=5000, save_path="data"):
    """
    Main loop to create and save the dataset.
    """
    X_data = []  # Stores measurements (Inputs)
    y_data = []  # Stores density matrices (Targets)

    print(f"Generating {num_samples} samples...")

    for _ in range(num_samples):
        # TODO: Call your helper functions
        # rho = ...
        # measurements = ...
        rho = generate_random_density_matrix(dim=2)
        measurements = get_pauli_measurements(rho)

        X_data.append(measurements)
        y_data.append(rho)

    # TODO: Save X_data and y_data as .npy files in 'save_path'
    # np.save(..., ...)
    # Converting to numpy arrays before saving for efficiency
    np.save(os.path.join(save_path, "measurements.npy"), np.array(X_data))
    np.save(os.path.join(save_path, "density_matrices.npy"), np.array(y_data))

    print("Dataset saved!")


if __name__ == "__main__":
    # Ensure the directory exists

    os.makedirs("data", exist_ok=True)

    generate_dataset()
