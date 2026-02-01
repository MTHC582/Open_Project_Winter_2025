# Report (Scalability & Software Design)

## 1. Serialization Strategy

For this project, we implemented a serialization system using Python's native `pickle` module.

**Why Pickle?**

- **Object Preservation:** Pickle allows us to serialize the entire `QuantumModel` class instance, including its internal state (parameters, configuration) and metadata, without manually mapping every variable to a file schema.
- **Simplicity:** It requires minimal boilerplate code compared to JSON or XML, making it ideal for saving experimental checkpoints (`.pkl` files).

**When to use HDF5?**
While `pickle` is excellent for Python objects, we would switch to **HDF5** (Hierarchical Data Format) if:

1.  **Data Volume:** We needed to store massive numerical datasets (TB scale) where loading the entire file into RAM is impossible. HDF5 supports partial I/O (reading slices from disk).
2.  **Interoperability:** We needed to share data with non-Python environments (e.g., C++, MATLAB), as `pickle` is Python-specific and not secure for untrusted data.

## 2. Ablation Study Plan

**Hypothesis:** Increasing the number of layers (`n_layers`) in the model architecture will likely increase the initialization runtime but should not affect the random fidelity baseline (since the model is untrained).

**Metrics:**

- **Independent Variable:** Model Depth (Layers: 1, 5, 10, 20)
- **Dependent Variables:** Mean Fidelity, Runtime (s)
- **Control:** Fixed Qubit count (N=4)
