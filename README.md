# Machine Learning for Quantum State Tomography

**Open Project Winter 2025**

## 📌 Project Overview

This repository contains the complete implementation of a **Scalable Quantum State Tomography (QST) Pipeline**. The goal was to develop software tools capable of reconstructing quantum states and benchmarking performance as the system size ($N$ qubits) scales.

Traditional tomography scales exponentially with Hilbert space dimension ($d=2^N$). This project explores **modular software design**, **surrogate modeling**, and **ablation studies** to benchmark classical limits.

---

## 🚀 Key Features

- **Multi-Mode State Representation:**
  - **Density Matrices ($\rho$):** Full support for mixed states, purity calculations, and Bloch sphere visualization (Task 1 & 2).
  - **Statevectors ($\psi$):** Scalable complex vector representations for pure states up to $N=12+$ (Task 3).
- **Metric System:** Automated calculation of Fidelity ($F$), Trace Distance ($D$), and Purity ($\gamma$).
- **Serialization Engine:** Custom `pickle`-based I/O system to save/load model states (`.pkl`) and checkpoints.
- **Ablation Framework:** Tools to test model depth vs. initialization time and fidelity.

---

## 📂 Repository Structure

```text
Open_Project_Winter_2025
│
├── Task1
│   ├── data
│   └── Task1.ipynb
│
├── Task2
│   ├── data
│   ├── docs
│   ├── outputs
│   ├── src
│   └── AI_USAGE.md
│
├── Task3
│   ├── docs
│   ├── models
│   ├── scalability_results.csv
│   └── Task3.ipynb
│
├── .gitignore
├── LICENSE
└── README.md
```
