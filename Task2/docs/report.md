# Assignment 2 Report: Quantum State Tomography

## 1. Model Working

**Architecture (Track 2: Hardware-Centric MLP):**
We implemented a Multi-Layer Perceptron (MLP) designed for efficient hardware deployment. The network maps measuring data (Pauli expectation values) to a valid quantum state representation.

- **Input:** 3 continuous values corresponding to $\langle X \rangle, \langle Y \rangle, \langle Z \rangle$.
- **Hidden Layers:** Two dense layers (64 neurons each) with ReLU activation.
- **Output:** 4 real-valued parameters representing the Cholesky factor $L$.

**Physical Constraints Enforcement:**  
To ensure the output is a valid density matrix (Hermitian, Positive Semi-Definite, Unit Trace)[cite: 5], we do not predict $\rho$ directly. Instead, the network predicts a lower-triangular matrix $L$.  
The density matrix is reconstructed via Cholesky decomposition:
$$\rho = \frac{LL^{\dagger}}{\text{Tr}(LL^{\dagger})}$$
This mathematical enforcement guarantees that every prediction satisfies quantum physical laws by construction.

## 2. Replication Guide

This guide details how to reproduce the results from scratch.

**Dependencies:**

- Python 3.12 or less (least 3.8-9)
- `torch` (PyTorch)
- `numpy`

**Step-by-Step Execution:**

1.  **Data Generation:**
    Run the following command to generate 5,000 synthetic quantum states and their measurements:

    ```bash
    python src/data.py
    ```

    _Output:_ Creates `data/measurements.npy` and `data/density_matrices.npy`.

2.  **Model Training:**
    Train the MLP for 50 epochs:

    ```bash
    python -m src.train
    ```

    _Output:_ Saves the best model weights to `outputs/model_weights.pt`.

3.  **Evaluation:**
    To verify metrics on the test set:
    ```bash
    python -m src.evaluate
    ```

## 3. Final Metrics

The model was evaluated on a held-out test set of 1,000 samples.

- **Mean Fidelity:** 0.99964 (Target: >0.99)
- **Mean Trace Distance:** 0.00727 (Target: <0.01)
- **Avg Inference Latency:** ~0.39 ms per stat
