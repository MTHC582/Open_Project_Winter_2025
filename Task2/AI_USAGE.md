# AI Usage Report

**Tool Used:**

- **Google Gemini** (Generative AI / LLM)

**1. Disclosure of Usage**
In compliance with the assignment's AI Attribution Policy, I declare that this project was developed with the assistance of an AI thought partner. The AI was used for:

- **Architectural Planning:** Designing the repository structure to strictly match assignment requirements (Section 3.1).
- **Mathematical Modeling:** Deriving the Cholesky decomposition logic to enforce physical constraints ($\rho = LL^\dagger / \text{Tr}$) within the neural network.
- **Code Scaffolding:** Generating initial Python templates for `data.py`, `model.py`, and `train.py`.
- **Debugging:** identifying shape mismatch errors in the tensor operations for the Fidelity metric calculation in `utils.py`.

**2. Prompt Logs**

- **Full Chat Transcript:** [My AI-CHAT](docs/Google%20Gemini.html)
  - _Description:_ The transcript documents the entire development process, from initial project setup to final metric evaluation.
  - _Key Prompts:_ "What is the final src file structure?", "Give me the template for model.py", "Debug the training loop loss calculation."

**3. Verification of AI Outputs**
To ensure academic integrity and correctness, all AI-generated content was verified via the following methods:

- **Mathematical Verification:** The Cholesky formula provided by the AI was cross-referenced with quantum state tomography literature to ensure it guarantees Positive Semi-Definiteness.
- **Metric Validation:** The "Fidelity" function was manually checked against standard cases (e.g., pure states) to ensure it returns 1.0 for identical states.
- **Constraint Checks:** The final model output was tested to confirm that $\text{Tr}(\rho) \approx 1.0$ and eigenvalues $\ge 0$ for every sample in the test set.
