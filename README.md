# Per-Gate Noise Vulnerability of Quantum Algorithms

**A Systematic Benchmarking Study Under Realistic Error Models**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hardware Validated](https://img.shields.io/badge/IBM%20Quantum-Validated-blueviolet.svg)](#hardware-validation)

## Author

**Tanvir Hassan**
Department of Physics, Jagannath University, Dhaka, Bangladesh
📧 [tanvir6307@gmail.com](mailto:tanvir6307@gmail.com)

---

## Overview

This repository contains the complete codebase, data, and figures for a systematic noise-aware benchmarking framework that evaluates how different quantum noise channels degrade the performance of quantum algorithms. The study introduces the **per-gate decay rate** (λ/G), a novel structure-aware metric for comparing algorithmic noise vulnerability across circuits of vastly different depths and gate counts.

### Key Contributions

1. **Systematic multi-channel benchmarking** — Quantum Teleportation, Grover's Search (3q, 5q), and VQE-optimized QAOA for MaxCut (4q, 6q) evaluated under four physically motivated noise channels (depolarizing, amplitude damping, phase damping, thermal relaxation with readout error).

2. **Per-gate decay rate (λ/G)** — A novel metric that normalizes the exponential fidelity decay constant by total gate count, enabling fair comparison between algorithms with different circuit structures.

3. **Hardware validation** — Simulation predictions validated on IBM's 156-qubit `ibm_marrakesh` processor (Heron architecture), achieving Pearson correlation r = 0.981 across 13 circuits.

4. **11,500 data points** with 25-seed ensemble averaging, bootstrap 95% confidence intervals, and analytical fitting (R² > 0.92 for all 20 algorithm–noise combinations).

---

## Repository Structure

```
.
├── noise_aware.py              # Main benchmarking framework (simulation + analysis + figures)
├── hardware_validation.py      # IBM Quantum hardware validation pipeline
├── results/
│   ├── benchmark_data_v2.json  # Complete benchmark data (raw + aggregated + fits + scaling)
│   ├── hardware_validation.json# Hardware validation results (13 circuits)
│   ├── aggregated_results.csv  # Aggregated fidelity data (460 points)
│   ├── fit_results.csv         # Exponential fit parameters (20 fits)
│   ├── fig1_fidelity_vs_noise_CI.{png,pdf}   # Fidelity decay with CI bands
│   ├── fig2_depth_normalized.{png,pdf}        # Depth-normalized fidelity F^{1/d}
│   ├── fig3_robustness_heatmap.{png,pdf}      # AUC robustness heatmap
│   ├── fig4_per_gate_decay.{png,pdf}          # Per-gate decay rate curves
│   ├── fig5_error_thresholds.{png,pdf}        # Error threshold bar chart
│   ├── fig6_depth_scaling.{png,pdf}           # Scaling study (qubits & depth)
│   ├── fig7_combined_overlay.{png,pdf}        # Cross-algorithm comparison
│   ├── fig8_per_gate_decay_rates.{png,pdf}    # Fitted λ/G bar chart
│   ├── fig9_success_probability.{png,pdf}     # Success probability with CI
│   ├── fig10_ci_convergence.{png,pdf}         # CI convergence validation
│   └── fig11_hardware_validation.{png,pdf}    # Hardware validation scatter + bars
└── README.md
```

---

## Main Results

### Algorithms & Noise Channels

| Algorithm | Qubits | Circuit Depth | Gate Count | Description |
|-----------|--------|---------------|------------|-------------|
| Teleportation | 3 | 8 | 11 | Deferred-measurement protocol |
| Grover-3q | 3 | 25 | 51 | 2 oracle-diffuser iterations |
| Grover-5q | 5 | 49 | 141 | 4 oracle-diffuser iterations |
| QAOA-p1-4q | 4 | 14 | 23 | VQE-optimized MaxCut, p=1 |
| QAOA-p1-6q | 6 | 26 | 48 | VQE-optimized MaxCut, p=1 |

**Noise channels:** Depolarizing, Amplitude Damping, Phase Damping, Thermal Relaxation (T₁/T₂ + readout error)

### Per-Gate Decay Rate (λ/G) — Novel Metric

| Algorithm | Depolarizing | Amp. Damping | Phase Damping | Thermal Relax. |
|-----------|-------------|-------------|--------------|----------------|
| Teleportation | 0.152 | 0.170 | 0.042 | 0.148 |
| Grover-3q | 0.342 | 0.217 | 0.113 | 0.127 |
| Grover-5q | 2.488 | 2.451 | 0.965 | 3.546 |
| QAOA-p1-4q | 0.495 | 0.338 | 0.274 | 0.326 |
| QAOA-p1-6q | 0.541 | 0.390 | 0.292 | 0.435 |

**Key findings:**
- **Teleportation** exhibits the lowest per-gate vulnerability (λ/G = 0.04–0.17)
- **Grover-5q** shows 14–24× higher per-gate decay than Teleportation, with super-linear growth from 3 to 5 qubits
- **QAOA** demonstrates moderate, stable per-gate decay (λ/G ≈ 0.3–0.5) across qubit counts
- **Depolarizing noise** is the most destructive channel per gate for search/optimization algorithms

### Hardware Validation

Validated on **IBM `ibm_marrakesh`** (156-qubit Heron processor):

| Metric | Value |
|--------|-------|
| Pearson correlation (sim vs hw) | r = 0.981 |
| Mean Absolute Error | 0.019 |
| RMSE | 0.029 |
| Circuits tested | 13 (5 Teleportation + 5 Grover + 3 QAOA) |
| Shots per circuit | 8,192 |

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/<username>/quantum-algorithm-benchmarking.git
cd quantum-algorithm-benchmarking

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install qiskit qiskit-aer numpy scipy matplotlib
```

For hardware validation (optional):
```bash
pip install qiskit-ibm-runtime
```

For GPU acceleration (optional):
```bash
pip install qiskit-aer-gpu
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `qiskit` ≥ 1.0 | Quantum circuit construction and transpilation |
| `qiskit-aer` | Density-matrix and shot-based simulation |
| `numpy` | Numerical computation |
| `scipy` | Curve fitting (exponential decay) and bootstrap CI |
| `matplotlib` | Publication-quality figure generation |
| `qiskit-ibm-runtime` | IBM Quantum hardware access (optional) |

---

## Usage

### Running the Full Benchmark

```bash
python noise_aware.py
```

This executes the complete pipeline:

1. **Backend detection:**  Attempts GPU acceleration, falls back to CPU density-matrix
2. **Noise sweep:** 4 noise types × 23 strengths × 5 algorithms × 25 seeds ≈ 11,500 simulations
3. **Aggregation:** Bootstrap 95% CI for all 460 data points
4. **Fitting:** Exponential decay model F(p) = A·exp(−λp) + (1−A)/2ⁿ
5. **Scaling study:** Grover 2→8 qubits, QAOA p=1→6 layers
6. **Visualization:** 10 publication figures (PNG + PDF at 400 DPI)
7. **Data export:** JSON and CSV files


### Running Hardware Validation

```bash
# Requires IBM Quantum account and API token
python hardware_validation.py
```

This submits 13 circuits to IBM Quantum hardware, runs calibration-based noise simulation, and generates the validation figure.

**Note:** Requires an IBM Quantum Platform account. Update the `IBM_TOKEN` variable in `hardware_validation.py` with your API token.

---

## Figures

| Figure | Description |
|--------|-------------|
| **Fig. 1** | Fidelity vs noise strength with 95% CI bands and exponential fit overlays |
| **Fig. 2** | Depth-normalized fidelity F^{1/d} for fair cross-algorithm comparison |
| **Fig. 3** | Robustness heatmap (AUC of fidelity curves) |
| **Fig. 4** | Per-gate decay rate −ln(F)/G as a function of noise strength |
| **Fig. 5** | Error threshold p* where fidelity drops below 0.5 |
| **Fig. 6** | Scaling study: Grover (qubit scaling) and QAOA (depth scaling) |
| **Fig. 7** | Cross-algorithm overlay by noise channel |
| **Fig. 8** | Fitted per-gate decay rates λ/G (bar chart) |
| **Fig. 9** | Task-specific success probability with CI |
| **Fig. 10** | Bootstrap CI convergence vs ensemble size |
| **Fig. 11** | Hardware validation: predicted vs measured success probabilities |

---

## Data Format

### `benchmark_data_v2.json`
```json
{
  "raw_results": [...],        // 11,500 individual simulation results
  "aggregated_results": [...], // 460 mean values with CI
  "fit_results": [...],        // 20 exponential fit parameters
  "scaling_results": [...],    // 52 scaling study data points
  "metadata": {
    "shots": 8192,
    "n_ensemble": 25,
    "bootstrap_ci": 0.95,
    "noise_strengths": [0.0, 0.005, ..., 0.5]
  }
}
```

### `hardware_validation.json`
```json
{
  "backend": "ibm_marrakesh",
  "job_id": "...",
  "shots": 8192,
  "n_circuits": 13,
  "statistics": {
    "pearson_r": 0.9808,
    "mae": 0.0191,
    "rmse": 0.0288
  },
  "results": [...]  // Per-circuit hardware vs simulation comparison
}
```

---

## Methodology

### Analytical Decay Model

Fidelity decay is modeled as:

$$\mathcal{F}(p) = A \exp(-\lambda p) + \frac{1-A}{2^n}$$

where:
- *A* ∈ [0, 1.5]: amplitude
- *λ* ≥ 0: decay rate
- *n*: number of qubits
- (1−A)/2ⁿ: asymptotic fidelity (maximally mixed state)

### Per-Gate Decay Rate

$$\lambda/G = \frac{\lambda}{\text{total gate count}}$$

This metric captures how efficiently each gate contributes to fidelity loss, independent of circuit size.

### Statistical Protocol

- **Ensemble size:** N = 25 independent random seeds per data point
- **Randomization:** Random Bloch sphere states (Teleportation), random marked items (Grover), random Erdős–Rényi graphs (QAOA)
- **Confidence intervals:** Percentile bootstrap with 2,000 resamples
- **Shots:** 8,192 per circuit for success probability estimation

---

## Citation

If you use this framework or data in your research, please cite:

```bibtex
@article{hassan2026pergate,
  title={Per-Gate Noise Vulnerability of Quantum Algorithms: A Systematic Benchmarking Study Across Multiple Error Channels},
  author={Hassan, Tanvir},
  year={2026},
  note={Menuscript is being processing}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Hardware validation was performed on IBM Quantum's `ibm_marrakesh` processor (156-qubit Heron architecture) via the IBM Quantum Platform. We thank IBM Quantum for providing cloud access to their quantum computing resources.
