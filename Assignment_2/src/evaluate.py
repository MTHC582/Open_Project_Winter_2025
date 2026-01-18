import torch
import numpy as np
import time
import os
from torch.utils.data import DataLoader, TensorDataset

# Import math functions
from src.model import DensityMatrixReconstructor
from src.utils import compute_fidelity, compute_trace_distance


def evaluate_model(data_path="data", model_path="outputs/model_weights.pt"):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    X_numpy = np.load(os.path.join(data_path, "measurements.npy"))
    y_numpy = np.load(os.path.join(data_path, "density_matrices.npy"))

    # Convert to Tensor
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    y_tensor = torch.tensor(y_numpy, dtype=torch.complex64)

    # Use the last 20% as test set
    split_idx = int(0.8 * len(X_tensor))
    test_X = X_tensor[split_idx:]
    test_y = y_tensor[split_idx:]

    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=1, shuffle=False)

    # --- Load Model ---
    model = DensityMatrixReconstructor().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Run train.py first.")
        return

    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Evaluating on {len(test_loader)} test samples...")

    # --- Run Inference & Collect Metrics ---
    fidelities = []
    trace_dists = []
    latencies = []

    with torch.no_grad():
        for i, (bx, by) in enumerate(test_loader):
            bx = bx.to(device)
            by = by.to(device)

            # Start Timer
            start_time = time.time()

            # Inference
            pred_rho = model(bx)

            # End Timer
            end_time = time.time()

            # 1. Fidelity (Higher is better, max 1.0)
            fid = compute_fidelity(pred_rho, by)
            fidelities.append(fid)

            # 2. Trace Distance (Lower is better, min 0.0)
            td = compute_trace_distance(pred_rho, by)
            trace_dists.append(td)

            # 3. Latency
            latencies.append((end_time - start_time) * 1000)

    # --- 4. Report Final Results ---
    mean_fidelity = np.mean(fidelities)
    mean_trace_dist = np.mean(trace_dists)
    avg_latency = np.mean(latencies)

    print("\n" + "=" * 40)
    print(f"   FINAL EVALUATION REPORT")
    print("=" * 40)
    print(f"Samples Tested:       {len(test_loader)}")
    print("-" * 40)
    print(f"Mean Fidelity:        {mean_fidelity:.5f}  (Target: > 0.99)")
    print(f"Mean Trace Distance:  {mean_trace_dist:.5f}  (Target: < 0.01)")
    print(f"Avg Inference Time:   {avg_latency:.4f} ms")
    print("=" * 40)


if __name__ == "__main__":
    evaluate_model()
