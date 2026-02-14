import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# Import your model module
from src.model import DensityMatrixReconstructor
from src.utils import compute_fidelity


def train_model(data_path="data", output_path="outputs", epochs=50, lr=1e-3):
    # --- SETUP DEVICE (CUDA Support) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Since im using python 3.12.10 , cuda can be used
    # Would suggest to use python versions likely as 3.8 < python --version <= 3.12

    # --- STEP 1: LOAD DATA ---
    print(f"Loading data from {data_path}...")

    # Define paths to your specific numpy files
    X_path = os.path.join(data_path, "measurements.npy")
    y_path = os.path.join(data_path, "density_matrices.npy")

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Files not found in {data_path}. Run data.py first.")

    # Load the numpy arrays
    X_numpy = np.load(X_path)
    y_numpy = np.load(y_path)

    # TODO: Create TensorDataset
    # Hint: X_numpy should be float(), y_numpy should be cfloat() (complex)
    # dataset = ...

    # We convert numpy arrays to Tensors and move them to the GPU (device) immediately
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    y_tensor = torch.tensor(y_numpy, dtype=torch.complex64)

    # Create Split (80% Train, 20% Test) for validation
    split_idx = int(0.8 * len(X_tensor))
    train_ds = TensorDataset(X_tensor[:split_idx], y_tensor[:split_idx])
    test_ds = TensorDataset(X_tensor[split_idx:], y_tensor[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(
        test_ds, batch_size=1000, shuffle=False
    )  # Large batch for fast validation

    # --- STEP 2: SETUP MODEL ---
    model = DensityMatrixReconstructor().to(device)

    # TODO: Initialize Optimizer (Adam is recommended)
    # optimizer = ...
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Helper for loss (MSE works on complex numbers in PyTorch, but we define it here)
    loss_fn = nn.MSELoss()

    # --- STEP 3: TRAINING LOOP ---
    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y_true in train_loader:
            # Moving batches to GPU here.
            batch_X = batch_X.to(device)
            batch_y_true = batch_y_true.to(device)

            # TODO: Zero gradients
            optimizer.zero_grad()

            # TODO: Forward pass
            # Remember your model now returns 'rho' directly!
            # rho_pred = ...
            rho_pred = model(batch_X)

            # TODO: Calculate Loss
            # Loss = MSE(Real Parts) + MSE(Imaginary Parts)
            # loss = ...
            # We split into Real and Imaginary parts and sum the losses.
            loss = loss_fn(rho_pred.real, batch_y_true.real) + loss_fn(
                rho_pred.imag, batch_y_true.imag
            )

            # TODO: Backward pass and Optimizer step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # --- VALIDATION STEP (Important for Assignment Metrics) ---
        if (epoch + 1) % 10 == 0:
            model.eval()  # Switch to eval mode (disable dropout etc.)
            avg_loss = total_loss / len(train_loader)

            # Check Fidelity on one test batch
            with torch.no_grad():
                test_X, test_y = next(iter(test_loader))

                # --- CORRECTION 3: Move validation batch to GPU ---
                test_X = test_X.to(device)
                test_y = test_y.to(device)

                test_pred = model(test_X)
                val_fidelity = compute_fidelity(test_pred, test_y)

            print(
                f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Val Fidelity: {val_fidelity:.4f}"
            )

    # --- STEP 4: SAVE ---
    os.makedirs(output_path, exist_ok=True)
    save_file = os.path.join(output_path, "model_weights.pt")

    # TODO: Save the model's state_dict
    # torch.save(...)
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")


if __name__ == "__main__":
    train_model(data_path="data")
