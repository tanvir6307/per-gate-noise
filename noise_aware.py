#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       NOISE-AWARE QUANTUM ALGORITHM BENCHMARKING FRAMEWORK                   ║
║                                                                              ║
║  Systematic study of how depolarizing, amplitude-damping, phase-damping,     ║
║  and thermal-relaxation noise affect Quantum Teleportation, Grover's         ║
║  Search, and QAOA for MaxCut.                                                ║
║                                                                              ║
║  Key features:                                                               ║
║    1. Statistical rigor   – N_ENSEMBLE seeds, bootstrap 95% CI               ║
║    2. Depth-normalized F  – F^{1/d} per-layer fidelity                       ║
║    3. Realistic noise     – T1/T2 thermal relaxation + readout error         ║
║    4. Analytical fitting  – F(p) = A·exp(-λp) + (1-A)/2^n                    ║
║    5. Ensemble averaging  – random states / marked items / graphs            ║
║    6. Multi-scale study   – Grover 2-8q, QAOA 4-8q, p=1-6                    ║
║    7. Publication figures – shaded CI, fit overlays, math labels             ║
║    8. Per-gate decay rate – λ/G structure-aware vulnerability metric         ║
║                                                                              ║
║  GPU-accelerated via Qiskit Aer when available.                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import time
import json
import warnings
import itertools
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Callable, Any
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import bootstrap as scipy_bootstrap

# ──────────────────────────────────────────────────────────────────────
# Qiskit imports
# ──────────────────────────────────────────────────────────────────────
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import (
    Statevector,
    DensityMatrix,
    state_fidelity,
    Operator,
    partial_trace,
)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
    ReadoutError,
)

# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════════════
# §0  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Dense at low noise for steep curves (Grover ≥5q), coarser at high noise
_fine   = np.round(np.arange(0.0, 0.055, 0.005), 4)   # 0..0.05  (11 pts)
_coarse = np.round(np.arange(0.06, 0.52, 0.04), 4)     # 0.06..0.50 (12 pts)
NOISE_STRENGTHS = sorted(set(_fine.tolist() + _coarse.tolist()))

SHOTS     = 8192
BASE_SEED = 42

# Ensemble and statistical configuration
N_ENSEMBLE   = 25        # independent random seeds per data point
BOOTSTRAP_CI = 0.95      # confidence level
N_BOOTSTRAP  = 2000      # bootstrap resamples

# Multi-scale study configuration
GROVER_QUBITS         = [2, 3, 4, 5, 6, 7, 8]         # scaling study
GROVER_BENCH_QUBITS   = [3, 5]                         # main noise sweep
QAOA_BENCH_CONFIGS    = [(4, 1), (6, 1)]               # (n_qubits, p)
QAOA_SCALE_DEPTHS     = [1, 2, 3, 4, 5, 6]             # scaling study
QAOA_SCALE_QUBITS     = [4, 5, 6, 7, 8]                # scaling study

# Thermal relaxation parameters (calibration-realistic values)
T1_US          = 50.0     # T1 relaxation time (microseconds)
T2_US          = 70.0     # T2 dephasing time (microseconds)
GATE_1Q_US     = 0.035    # single-qubit gate time (microseconds)
GATE_2Q_US     = 0.300    # two-qubit gate time (microseconds)
READOUT_ERROR  = 0.02     # base readout bit-flip probability


# ══════════════════════════════════════════════════════════════════════
# §1  GPU / BACKEND DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_gpu_backend() -> Tuple[AerSimulator, str]:
    """Try GPU-accelerated Aer first, fall back to CPU."""
    for method, label in [
        ("statevector_gpu", "GPU-cuStateVec"),
        ("density_matrix_gpu", "GPU-DensityMatrix"),
    ]:
        try:
            sim = AerSimulator(method=method)
            qc = QuantumCircuit(1); qc.h(0); qc.measure_all()
            sim.run(transpile(qc, sim), shots=10).result()
            print(f"  ✓ Backend: {label}")
            return sim, label
        except Exception:
            continue
    sim = AerSimulator(method="density_matrix")
    print("  ✓ Backend: CPU-DensityMatrix")
    return sim, "CPU-DensityMatrix"


# ══════════════════════════════════════════════════════════════════════
# §2  NOISE MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════

NOISE_TYPES = [
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "thermal_relaxation",
]

NOISE_LABELS = {
    "depolarizing":       "Depolarizing",
    "amplitude_damping":  "Amplitude Damping",
    "phase_damping":      "Phase Damping",
    "thermal_relaxation": "Thermal Relaxation (T1/T2)",
}


def build_noise_model(
    noise_type: str,
    strength: float,
    n_qubits: int,
    add_readout: bool = True,
) -> Optional[NoiseModel]:
    """
    Construct a NoiseModel for the given noise channel and strength.

    Parameters
    ----------
    noise_type  : one of NOISE_TYPES
    strength    : error probability / gamma in [0, 1)
    n_qubits    : number of qubits in the circuit
    add_readout : whether to add measurement readout error
    """
    if strength <= 0.0:
        return None

    noise_model = NoiseModel()
    strength = min(strength, 0.999)

    single_q_gates = ["u1", "u2", "u3", "x", "y", "z", "h", "s",
                      "t", "rx", "ry", "rz", "id", "sx"]
    two_q_gates    = ["cx", "cz", "cy", "swap", "cp", "ecr", "rxx"]

    if noise_type == "depolarizing":
        err_1q = depolarizing_error(strength, 1)
        err_2q = depolarizing_error(min(strength, 15 / 16 - 1e-9), 2)

    elif noise_type == "amplitude_damping":
        err_1q = amplitude_damping_error(strength)
        err_2q = err_1q.tensor(amplitude_damping_error(strength))

    elif noise_type == "phase_damping":
        err_1q = phase_damping_error(strength)
        err_2q = err_1q.tensor(phase_damping_error(strength))

    # Thermal relaxation channel (T1/T2 decoherence)
    elif noise_type == "thermal_relaxation":
        t1 = T1_US
        t2 = min(T2_US, 2 * t1)
        # Scale gate times by `strength` so larger strength → more decoherence
        scaled_1q = GATE_1Q_US + strength * t1 * 0.5
        scaled_2q = GATE_2Q_US + strength * t1 * 1.0
        err_1q = thermal_relaxation_error(t1, t2, scaled_1q)
        err_2q = thermal_relaxation_error(t1, t2, scaled_2q).tensor(
            thermal_relaxation_error(t1, t2, scaled_2q)
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    for g in single_q_gates:
        noise_model.add_all_qubit_quantum_error(err_1q, g)
    for g in two_q_gates:
        noise_model.add_all_qubit_quantum_error(err_2q, g)

    # Readout error (measurement bit-flip)
    if add_readout and strength > 0:
        p_ro = min(READOUT_ERROR * strength * 5, 0.45)
        ro_err = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
        for q in range(n_qubits):
            noise_model.add_readout_error(ro_err, [q])

    return noise_model


# ══════════════════════════════════════════════════════════════════════
# §3  QUANTUM CIRCUIT BUILDERS
# ══════════════════════════════════════════════════════════════════════

# ── 3A  TELEPORTATION ─────────────────────────────────────────────

def build_teleportation_circuit(
    state_angles: Tuple[float, float] = (np.pi / 4, np.pi / 6),
) -> QuantumCircuit:
    """
    3-qubit teleportation with deferred measurement (unitary corrections).
    Prepares |ψ⟩ = Ry(θ)Rz(φ)|0⟩ on q0, teleports to q2.
    """
    theta, phi = state_angles
    qc = QuantumCircuit(3, name="Teleportation")
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.barrier()
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier()
    qc.cx(1, 2)
    qc.cz(0, 2)
    return qc


def teleportation_ideal_dm(
    state_angles: Tuple[float, float] = (np.pi / 4, np.pi / 6),
) -> DensityMatrix:
    """Single-qubit density matrix of the state to be teleported."""
    theta, phi = state_angles
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return DensityMatrix.from_instruction(qc)


# ── 3B  GROVER ────────────────────────────────────────────────────

def _grover_oracle(n_qubits: int, marked: int) -> QuantumCircuit:
    """Phase oracle: flips sign of |marked⟩."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for i in range(n_qubits):
        if not (marked >> i) & 1:
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i in range(n_qubits):
        if not (marked >> i) & 1:
            qc.x(i)
    return qc


def _grover_diffuser(n_qubits: int) -> QuantumCircuit:
    """Grover diffusion operator."""
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(
    n_qubits: int = 3, marked: int = 5, n_iterations: Optional[int] = None,
) -> QuantumCircuit:
    """Grover's algorithm. Default iterations ≈ π/4·√N."""
    N = 2**n_qubits
    if n_iterations is None:
        n_iterations = max(1, int(np.round(np.pi / 4 * np.sqrt(N))))
    oracle   = _grover_oracle(n_qubits, marked)
    diffuser = _grover_diffuser(n_qubits)
    qc = QuantumCircuit(n_qubits, name=f"Grover-{n_qubits}q")
    qc.h(range(n_qubits))
    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    return qc


def grover_ideal_sv(
    n_qubits: int = 3, marked: int = 5, n_iterations: Optional[int] = None,
) -> Statevector:
    """Ideal Grover output statevector."""
    return Statevector.from_instruction(
        build_grover_circuit(n_qubits, marked, n_iterations)
    )


# ── 3C  QAOA (MaxCut) ────────────────────────────────────────────

def _maxcut_qaoa_circuit(
    edges: List[Tuple[int, int]], n_qubits: int,
    gamma: float, beta: float, p: int = 1,
) -> QuantumCircuit:
    """Build a p-layer QAOA circuit for MaxCut."""
    qc = QuantumCircuit(n_qubits, name=f"QAOA-p{p}")
    qc.h(range(n_qubits))
    for _layer in range(p):
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)
        for i in range(n_qubits):
            qc.rx(2 * beta, i)
    return qc


def _generate_graph(n_qubits: int, seed: int) -> List[Tuple[int, int]]:
    """Generate random graph edges for MaxCut."""
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(n_qubits)
             for j in range(i + 1, n_qubits)
             if rng.random() > 0.4]
    if not edges:
        edges = [(0, 1)]
    return edges


def _maxcut_value(bitstring: str, edges: List[Tuple[int, int]]) -> int:
    """Count edges cut by a bitstring assignment."""
    return sum(1 for i, j in edges
               if int(bitstring[-(i + 1)]) != int(bitstring[-(j + 1)]))


_QAOA_OPT_CACHE: Dict[tuple, Tuple[float, float]] = {}


def optimize_qaoa_params(
    edges: List[Tuple[int, int]], n_qubits: int, p: int,
) -> Tuple[float, float]:
    """
    Optimize QAOA angles (γ, β) for MaxCut on given graph.
    Coarse grid search + Nelder-Mead refinement.  Cached per (graph, n, p).
    Uses shared angles across all p layers (uniform-angle QAOA).
    """
    key = (tuple(edges), n_qubits, p)
    if key in _QAOA_OPT_CACHE:
        return _QAOA_OPT_CACHE[key]

    opt_cut = max(_maxcut_value(format(b, f"0{n_qubits}b"), edges)
                  for b in range(2**n_qubits))

    def _neg_approx_ratio(params):
        g, b = float(params[0]), float(params[1])
        qc = _maxcut_qaoa_circuit(edges, n_qubits, g, b, p)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()
        exp_cut = sum(_maxcut_value(bs, edges) * prob
                      for bs, prob in probs.items())
        return -exp_cut / max(opt_cut, 1)

    # Coarse grid search
    best_val, best_params = 0.0, (np.pi / 4, np.pi / 8)
    for g in np.linspace(0.05, np.pi, 10):
        for b in np.linspace(0.05, np.pi / 2, 10):
            val = _neg_approx_ratio([g, b])
            if val < best_val:
                best_val = val
                best_params = (g, b)

    # Refine with Nelder-Mead
    res = minimize(_neg_approx_ratio, best_params, method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4})
    gamma_opt, beta_opt = float(res.x[0]), float(res.x[1])
    _QAOA_OPT_CACHE[key] = (gamma_opt, beta_opt)
    return gamma_opt, beta_opt


def build_qaoa_circuit(
    n_qubits: int = 4, p: int = 1, seed: int = BASE_SEED,
) -> Tuple[QuantumCircuit, List[Tuple[int, int]]]:
    """Build QAOA for MaxCut with VQE-optimized angles. Returns (circuit, edges)."""
    edges = _generate_graph(n_qubits, seed)
    gamma, beta = optimize_qaoa_params(edges, n_qubits, p)
    qc = _maxcut_qaoa_circuit(edges, n_qubits, gamma, beta, p)
    return qc, edges


def qaoa_ideal_sv(
    n_qubits: int = 4, p: int = 1, seed: int = BASE_SEED,
) -> Statevector:
    """Ideal QAOA output statevector (with optimized angles)."""
    qc, _ = build_qaoa_circuit(n_qubits, p, seed)
    return Statevector.from_instruction(qc)


# ══════════════════════════════════════════════════════════════════════
# §4  SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Single data point (algorithm, noise, strength, ensemble_seed)."""
    algorithm: str
    noise_type: str
    noise_strength: float
    fidelity: float
    success_probability: float
    circuit_depth: int
    gate_count: int
    two_qubit_gate_count: int
    n_qubits: int
    ensemble_seed: int
    wall_time_s: float


@dataclass
class AggregatedResult:
    """Aggregated over ensemble seeds with bootstrap CI."""
    algorithm: str
    noise_type: str
    noise_strength: float
    fidelity_mean: float
    fidelity_ci_lo: float
    fidelity_ci_hi: float
    fidelity_std: float
    succ_prob_mean: float
    succ_prob_ci_lo: float
    succ_prob_ci_hi: float
    depth_norm_fidelity: float         # F^{1/d}
    per_gate_decay_rate: float         # -ln(F) / gate_count
    circuit_depth: int
    gate_count: int
    n_qubits: int
    n_samples: int


def simulate_density_matrix(
    qc: QuantumCircuit,
    noise_model: Optional[NoiseModel],
    backend: AerSimulator,
) -> DensityMatrix:
    """Run circuit on density-matrix simulator, return full DM."""
    qc_copy = qc.copy()
    qc_copy.save_density_matrix()
    tqc = transpile(qc_copy, backend, optimization_level=0)
    result = backend.run(tqc, noise_model=noise_model, shots=1).result()
    return result.data()["density_matrix"]


def simulate_counts(
    qc: QuantumCircuit,
    noise_model: Optional[NoiseModel],
    backend: AerSimulator,
    shots: int = SHOTS,
    seed: int = BASE_SEED,
) -> dict:
    """Run with measurements and return counts."""
    qc_meas = qc.copy()
    qc_meas.measure_all()
    tqc = transpile(qc_meas, backend, optimization_level=0)
    result = backend.run(tqc, noise_model=noise_model, shots=shots,
                         seed_simulator=seed).result()
    return result.get_counts()


def _circuit_stats(qc: QuantumCircuit) -> dict:
    """Extract circuit depth, gate count, 2q gate count."""
    ops = qc.count_ops()
    two_q = sum(v for k, v in ops.items()
                if k in {"cx", "cz", "cy", "swap", "cp", "ecr", "mcx", "rxx"})
    return {
        "circuit_depth": qc.depth(),
        "gate_count": sum(ops.values()),
        "two_qubit_gate_count": two_q,
    }


# ══════════════════════════════════════════════════════════════════════
# §5  BENCHMARK RUNNERS (ensemble-averaged over random instances)
# ══════════════════════════════════════════════════════════════════════

def _random_bloch_angles(rng: np.random.Generator) -> Tuple[float, float]:
    """Sample a uniformly random point on the Bloch sphere."""
    theta = float(rng.uniform(0, np.pi))
    phi   = float(rng.uniform(0, 2 * np.pi))
    return (theta, phi)


def run_teleportation_single(
    noise_type: str, strength: float, backend: AerSimulator, seed: int,
) -> BenchmarkResult:
    """Single teleportation run with a random input state."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    angles = _random_bloch_angles(rng)
    qc = build_teleportation_circuit(angles)
    ideal_dm = teleportation_ideal_dm(angles)
    n_qubits = qc.num_qubits
    nm = build_noise_model(noise_type, strength, n_qubits)
    full_dm = simulate_density_matrix(qc, nm, backend)
    noisy_dm = partial_trace(DensityMatrix(full_dm), [0, 1])
    fid = float(state_fidelity(ideal_dm, noisy_dm))
    stats = _circuit_stats(qc)
    return BenchmarkResult(
        algorithm="Teleportation", noise_type=noise_type,
        noise_strength=strength, fidelity=fid, success_probability=fid,
        n_qubits=n_qubits, ensemble_seed=seed,
        wall_time_s=time.perf_counter() - t0, **stats,
    )


def run_grover_single(
    noise_type: str, strength: float, backend: AerSimulator,
    n_qubits: int, seed: int,
) -> BenchmarkResult:
    """Single Grover run with a random marked item."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    marked = int(rng.integers(0, 2**n_qubits))
    qc = build_grover_circuit(n_qubits, marked)
    ideal_sv = grover_ideal_sv(n_qubits, marked)
    ideal_dm = DensityMatrix(ideal_sv)
    nm = build_noise_model(noise_type, strength, n_qubits)
    noisy_dm = simulate_density_matrix(qc, nm, backend)
    fid = float(state_fidelity(ideal_dm, DensityMatrix(noisy_dm)))
    # Success probability from counts
    counts = simulate_counts(qc, nm, backend, seed=seed)
    target = format(marked, f"0{n_qubits}b")
    target_rev = target[::-1]
    succ = max(counts.get(target, 0), counts.get(target_rev, 0)) / SHOTS
    stats = _circuit_stats(qc)
    return BenchmarkResult(
        algorithm=f"Grover-{n_qubits}q", noise_type=noise_type,
        noise_strength=strength, fidelity=fid, success_probability=succ,
        n_qubits=n_qubits, ensemble_seed=seed,
        wall_time_s=time.perf_counter() - t0, **stats,
    )


def run_qaoa_single(
    noise_type: str, strength: float, backend: AerSimulator,
    n_qubits: int, p: int, seed: int,
) -> BenchmarkResult:
    """Single QAOA run with a random graph instance."""
    t0 = time.perf_counter()
    qc, edges = build_qaoa_circuit(n_qubits, p, seed)
    ideal_sv = qaoa_ideal_sv(n_qubits, p, seed)
    ideal_dm = DensityMatrix(ideal_sv)
    nm = build_noise_model(noise_type, strength, n_qubits)
    noisy_dm = simulate_density_matrix(qc, nm, backend)
    fid = float(state_fidelity(ideal_dm, DensityMatrix(noisy_dm)))
    # Approximation ratio
    counts = simulate_counts(qc, nm, backend, seed=seed)
    opt_cut = max(_maxcut_value(format(b, f"0{n_qubits}b"), edges)
                  for b in range(2**n_qubits))
    total = sum(counts.values())
    exp_cut = sum(_maxcut_value(bs, edges) * c
                  for bs, c in counts.items()) / total
    approx = exp_cut / max(opt_cut, 1)
    stats = _circuit_stats(qc)
    return BenchmarkResult(
        algorithm=f"QAOA-p{p}-{n_qubits}q", noise_type=noise_type,
        noise_strength=strength, fidelity=fid, success_probability=approx,
        n_qubits=n_qubits, ensemble_seed=seed,
        wall_time_s=time.perf_counter() - t0, **stats,
    )


# ══════════════════════════════════════════════════════════════════════
# §6  STATISTICAL AGGREGATION
# ══════════════════════════════════════════════════════════════════════

def _bootstrap_ci(data: np.ndarray, confidence: float = BOOTSTRAP_CI) -> Tuple[float, float]:
    """Bootstrap 95% confidence interval for the mean."""
    if len(data) < 3:
        return float(np.min(data)), float(np.max(data))
    try:
        res = scipy_bootstrap(
            (data,), np.mean, n_resamples=N_BOOTSTRAP,
            confidence_level=confidence, method="percentile",
            random_state=np.random.default_rng(0),
        )
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        m, s = np.mean(data), np.std(data, ddof=1)
        return float(m - 1.96 * s), float(m + 1.96 * s)


def aggregate_results(raw: List[BenchmarkResult]) -> List[AggregatedResult]:
    """
    Group raw results by (algorithm, noise_type, noise_strength) and compute:
      - mean, std, and bootstrap 95% CI
      - depth-normalized fidelity F^{1/d}
      - per-gate decay rate  -ln(F) / gate_count
    """
    groups: Dict[tuple, List[BenchmarkResult]] = defaultdict(list)
    for r in raw:
        groups[(r.algorithm, r.noise_type, r.noise_strength)].append(r)

    agg = []
    for (algo, nt, ns), items in groups.items():
        fids  = np.array([r.fidelity for r in items])
        succs = np.array([r.success_probability for r in items])

        f_mean = float(np.mean(fids))
        f_ci   = _bootstrap_ci(fids)
        f_std  = float(np.std(fids, ddof=1)) if len(fids) > 1 else 0.0

        s_mean = float(np.mean(succs))
        s_ci   = _bootstrap_ci(succs)

        depth = items[0].circuit_depth
        gc    = items[0].gate_count

        # Depth-normalized fidelity: F^{1/d}
        dnf = f_mean ** (1.0 / max(depth, 1)) if f_mean > 0 else 0.0

        # Per-gate decay rate: -ln(F) / G
        pgdr = -np.log(max(f_mean, 1e-15)) / max(gc, 1)

        agg.append(AggregatedResult(
            algorithm=algo, noise_type=nt, noise_strength=ns,
            fidelity_mean=f_mean, fidelity_ci_lo=f_ci[0], fidelity_ci_hi=f_ci[1],
            fidelity_std=f_std,
            succ_prob_mean=s_mean, succ_prob_ci_lo=s_ci[0], succ_prob_ci_hi=s_ci[1],
            depth_norm_fidelity=dnf, per_gate_decay_rate=pgdr,
            circuit_depth=depth, gate_count=gc,
            n_qubits=items[0].n_qubits, n_samples=len(items),
        ))

    return sorted(agg, key=lambda a: (a.algorithm, a.noise_type, a.noise_strength))


# ══════════════════════════════════════════════════════════════════════
# §7  ANALYTICAL FITTING  —  F(p) = A·exp(-λp) + (1-A)/2^n
# ══════════════════════════════════════════════════════════════════════

def _fidelity_model(p: np.ndarray, A: float, lam: float, n_qubits: int) -> np.ndarray:
    """Exponential decay model: F(p) = A·exp(-λ·p) + (1-A)/2^n."""
    return A * np.exp(-lam * p) + (1 - A) / 2**n_qubits


@dataclass
class FitResult:
    """Result of fitting the exponential decay model."""
    algorithm: str
    noise_type: str
    A: float
    lam: float              # overall decay rate λ
    lam_per_gate: float     # λ / gate_count
    lam_per_depth: float    # λ / circuit_depth
    n_qubits: int
    gate_count: int
    circuit_depth: int
    r_squared: float
    converged: bool


def fit_fidelity_decay(agg_results: List[AggregatedResult]) -> List[FitResult]:
    """Fit exponential decay to each (algorithm, noise_type) fidelity curve."""
    groups: Dict[tuple, List[AggregatedResult]] = defaultdict(list)
    for a in agg_results:
        groups[(a.algorithm, a.noise_type)].append(a)

    fits = []
    for (algo, nt), items in groups.items():
        items = sorted(items, key=lambda a: a.noise_strength)
        xs = np.array([a.noise_strength for a in items])
        ys = np.array([a.fidelity_mean for a in items])
        nq = items[0].n_qubits
        gc = items[0].gate_count
        cd = items[0].circuit_depth

        if len(xs) < 4:
            continue
        try:
            popt, _ = curve_fit(
                lambda p, A, lam: _fidelity_model(p, A, lam, nq),
                xs, ys, p0=[1.0, 5.0], bounds=([0, 0], [1.5, 500]),
                maxfev=10000,
            )
            y_pred = _fidelity_model(xs, *popt, nq)
            ss_res = np.sum((ys - y_pred) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-15)
            fits.append(FitResult(
                algorithm=algo, noise_type=nt,
                A=float(popt[0]), lam=float(popt[1]),
                lam_per_gate=float(popt[1]) / max(gc, 1),
                lam_per_depth=float(popt[1]) / max(cd, 1),
                n_qubits=nq, gate_count=gc, circuit_depth=cd,
                r_squared=float(r2), converged=True,
            ))
        except Exception:
            fits.append(FitResult(
                algorithm=algo, noise_type=nt,
                A=0, lam=0, lam_per_gate=0, lam_per_depth=0,
                n_qubits=nq, gate_count=gc, circuit_depth=cd,
                r_squared=0, converged=False,
            ))
    return fits


# ══════════════════════════════════════════════════════════════════════
# §8  ERROR THRESHOLD DETECTION
# ══════════════════════════════════════════════════════════════════════

def find_error_threshold(
    fidelities: List[float], noise_strengths: List[float],
    threshold_fidelity: float = 0.5,
) -> Optional[float]:
    """Find noise strength where fidelity crosses threshold (interpolated)."""
    for i in range(1, len(fidelities)):
        if fidelities[i] <= threshold_fidelity <= fidelities[i - 1]:
            f0, f1 = fidelities[i - 1], fidelities[i]
            p0, p1 = noise_strengths[i - 1], noise_strengths[i]
            if abs(f0 - f1) < 1e-12:
                return p0
            return float(p0 + (threshold_fidelity - f0) * (p1 - p0) / (f1 - f0))
    return None


# ══════════════════════════════════════════════════════════════════════
# §9  SCALING STUDY  (Grover 2→8 qubits, QAOA p=1→6)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ScalingResult:
    algorithm: str
    noise_type: str
    noise_strength: float
    parameter: str
    parameter_value: int
    fidelity_mean: float
    fidelity_ci_lo: float
    fidelity_ci_hi: float
    depth_norm_fidelity: float
    circuit_depth: int


def run_scaling_study(backend: AerSimulator) -> List[ScalingResult]:
    """Scale Grover (qubits 2→8) and QAOA (p=1→6) with ensemble CI."""
    results = []
    fixed_p = 0.05

    for nt in NOISE_TYPES:
        # ── Grover qubit scaling ──
        for nq in GROVER_QUBITS:
            fids = []
            depth = 0
            for s in range(N_ENSEMBLE):
                seed = BASE_SEED + s * 1000
                rng = np.random.default_rng(seed)
                marked = int(rng.integers(0, 2**nq))
                qc = build_grover_circuit(nq, marked)
                ideal_dm = DensityMatrix(grover_ideal_sv(nq, marked))
                nm = build_noise_model(nt, fixed_p, nq)
                noisy_dm = simulate_density_matrix(qc, nm, backend)
                fid = float(state_fidelity(ideal_dm, DensityMatrix(noisy_dm)))
                fids.append(fid)
                depth = qc.depth()
            farr = np.array(fids)
            ci = _bootstrap_ci(farr)
            fm = float(np.mean(farr))
            dnf = fm ** (1.0 / max(depth, 1)) if fm > 0 else 0.0
            results.append(ScalingResult(
                algorithm="Grover", noise_type=nt, noise_strength=fixed_p,
                parameter="n_qubits", parameter_value=nq,
                fidelity_mean=fm, fidelity_ci_lo=ci[0], fidelity_ci_hi=ci[1],
                depth_norm_fidelity=dnf, circuit_depth=depth,
            ))
        print(f"      Grover qubit scaling ({NOISE_LABELS[nt]}): "
              f"{len(GROVER_QUBITS)} pts")

        # ── QAOA depth scaling ──
        for p_depth in QAOA_SCALE_DEPTHS:
            nq = 4
            fids = []
            depth = 0
            for s in range(N_ENSEMBLE):
                seed = BASE_SEED + s * 1000
                qc, _ = build_qaoa_circuit(nq, p_depth, seed)
                ideal_dm = DensityMatrix(qaoa_ideal_sv(nq, p_depth, seed))
                nm = build_noise_model(nt, fixed_p, nq)
                noisy_dm = simulate_density_matrix(qc, nm, backend)
                fid = float(state_fidelity(ideal_dm, DensityMatrix(noisy_dm)))
                fids.append(fid)
                depth = qc.depth()
            farr = np.array(fids)
            ci = _bootstrap_ci(farr)
            fm = float(np.mean(farr))
            dnf = fm ** (1.0 / max(depth, 1)) if fm > 0 else 0.0
            results.append(ScalingResult(
                algorithm="QAOA", noise_type=nt, noise_strength=fixed_p,
                parameter="p_depth", parameter_value=p_depth,
                fidelity_mean=fm, fidelity_ci_lo=ci[0], fidelity_ci_hi=ci[1],
                depth_norm_fidelity=dnf, circuit_depth=depth,
            ))
        print(f"      QAOA depth scaling ({NOISE_LABELS[nt]}): "
              f"{len(QAOA_SCALE_DEPTHS)} pts")

    return results


# ══════════════════════════════════════════════════════════════════════
# §10  VISUALIZATION  (publication-quality figures with CI bands)
# ══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif":  ["Computer Modern", "DejaVu Serif", "Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ALGO_COLORS = {
    "Teleportation": "#1565C0",
    "Grover-3q":     "#2E7D32",  "Grover-5q": "#43A047",
    "Grover-6q":     "#66BB6A",  "Grover-7q": "#81C784",
    "Grover-8q":     "#A5D6A7",  "Grover":    "#2E7D32",
    "QAOA-p1-4q":    "#E65100",  "QAOA-p1-6q":"#FF9800",
    "QAOA":          "#E65100",
}

NOISE_MARKERS = {
    "depolarizing": "o", "amplitude_damping": "s",
    "phase_damping": "^", "thermal_relaxation": "D",
}
NOISE_LINES = {
    "depolarizing": "-", "amplitude_damping": "--",
    "phase_damping": "-.", "thermal_relaxation": ":",
}
NOISE_COLORS = {
    "depolarizing": "#D32F2F", "amplitude_damping": "#7B1FA2",
    "phase_damping": "#1565C0", "thermal_relaxation": "#F57F17",
}


def _save_fig(fig, name: str):
    """Helper: save as both PNG and PDF."""
    fig.savefig(RESULTS_DIR / f"{name}.png")
    fig.savefig(RESULTS_DIR / f"{name}.pdf")
    print(f"  📊 {name}.png/pdf")


# ── FIG 1: Fidelity vs noise with CI + fit overlay ───────────────

def plot_fidelity_vs_noise_ci(agg: List[AggregatedResult],
                              fits: List[FitResult], save=True):
    algorithms = sorted(set(a.algorithm for a in agg))
    n = len(algorithms)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fit_map = {(f.algorithm, f.noise_type): f for f in fits}

    for ax, algo in zip(axes, algorithms):
        for nt in NOISE_TYPES:
            sub = sorted([a for a in agg
                          if a.algorithm == algo and a.noise_type == nt],
                         key=lambda a: a.noise_strength)
            if not sub:
                continue
            xs = np.array([a.noise_strength for a in sub])
            ys = np.array([a.fidelity_mean for a in sub])
            lo = np.array([a.fidelity_ci_lo for a in sub])
            hi = np.array([a.fidelity_ci_hi for a in sub])
            c = NOISE_COLORS[nt]
            ax.plot(xs, ys, marker=NOISE_MARKERS[nt], linestyle=NOISE_LINES[nt],
                    color=c, label=NOISE_LABELS[nt], markersize=3.5, linewidth=1.3)
            ax.fill_between(xs, lo, hi, alpha=0.15, color=c)
            # Fit overlay
            fr = fit_map.get((algo, nt))
            if fr and fr.converged:
                xf = np.linspace(0, xs.max(), 200)
                yf = _fidelity_model(xf, fr.A, fr.lam, fr.n_qubits)
                ax.plot(xf, yf, color=c, linewidth=0.8, alpha=0.5, linestyle="--")
            # Threshold
            thresh = find_error_threshold(ys.tolist(), xs.tolist(), 0.5)
            if thresh is not None:
                ax.axvline(thresh, color=c, alpha=0.2, linewidth=0.7, linestyle=":")
        ax.set_xlabel(r"Noise Strength $p$")
        ax.set_title(algo, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel(r"State Fidelity $\mathcal{F}$")
    # Shared legend outside plot area — avoids obscuring curves
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(NOISE_TYPES),
               bbox_to_anchor=(0.5, -0.10), fontsize=8, framealpha=0.9,
               edgecolor="gray")
    fig.suptitle(r"Fidelity vs Noise (shaded: 95% CI, dashed: $Ae^{-\lambda p}$ fit)",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    if save:
        _save_fig(fig, "fig1_fidelity_vs_noise_CI")
    plt.close(fig)


# ── FIG 2: Depth-normalized fidelity F^{1/d} ─────────────────────

def plot_depth_normalized(agg: List[AggregatedResult], save=True):
    algorithms = sorted(set(a.algorithm for a in agg))
    fig, axes = plt.subplots(1, len(NOISE_TYPES),
                             figsize=(4.8 * len(NOISE_TYPES), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, nt in zip(axes, NOISE_TYPES):
        for algo in algorithms:
            sub = sorted([a for a in agg
                          if a.algorithm == algo and a.noise_type == nt],
                         key=lambda a: a.noise_strength)
            if not sub:
                continue
            xs = [a.noise_strength for a in sub]
            ys = [a.depth_norm_fidelity for a in sub]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.3,
                    label=algo, color=ALGO_COLORS.get(algo))
        ax.set_xlabel(r"Noise Strength $p$")
        ax.set_title(NOISE_LABELS[nt])
        ax.set_ylim(0.5, 1.02)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
    axes[0].set_ylabel(r"Depth-Normalized Fidelity $\mathcal{F}^{1/d}$")
    fig.suptitle(r"Per-Layer Fidelity $\mathcal{F}^{1/d}$ — Fair Comparison",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save:
        _save_fig(fig, "fig2_depth_normalized")
    plt.close(fig)


# ── FIG 3: Robustness heatmap (AUC) ──────────────────────────────

def plot_robustness_heatmap(agg: List[AggregatedResult], save=True):
    algorithms = sorted(set(a.algorithm for a in agg))
    mat = np.zeros((len(algorithms), len(NOISE_TYPES)))
    for i, algo in enumerate(algorithms):
        for j, nt in enumerate(NOISE_TYPES):
            sub = sorted([a for a in agg
                          if a.algorithm == algo and a.noise_type == nt],
                         key=lambda a: a.noise_strength)
            if len(sub) < 2:
                continue
            xs = np.array([a.noise_strength for a in sub])
            ys = np.array([a.fidelity_mean for a in sub])
            mat[i, j] = float(np.trapezoid(ys, xs))

    fig, ax = plt.subplots(figsize=(7, max(3, len(algorithms) * 0.7 + 1)))
    cmap = LinearSegmentedColormap.from_list("r",
                                             ["#EF5350", "#FFEE58", "#66BB6A"])
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(NOISE_TYPES)))
    ax.set_xticklabels([NOISE_LABELS[nt] for nt in NOISE_TYPES],
                       rotation=25, ha="right")
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms)
    for i in range(len(algorithms)):
        for j in range(len(NOISE_TYPES)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if mat[i, j] < 0.12 else "black")
    plt.colorbar(im, ax=ax, label=r"AUC($\mathcal{F}$) — Robustness")
    ax.set_title("Algorithm Robustness to Noise (AUC)", fontweight="bold")
    plt.tight_layout()
    if save:
        _save_fig(fig, "fig3_robustness_heatmap")
    plt.close(fig)


# ── FIG 4: Per-gate decay rate λ/G ────────────────────────────────

def plot_per_gate_decay(agg: List[AggregatedResult], save=True):
    algorithms = sorted(set(a.algorithm for a in agg))
    fig, axes = plt.subplots(1, len(NOISE_TYPES),
                             figsize=(4.8 * len(NOISE_TYPES), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, nt in zip(axes, NOISE_TYPES):
        for algo in algorithms:
            sub = sorted([a for a in agg
                          if a.algorithm == algo and a.noise_type == nt
                          and a.noise_strength > 0.01],
                         key=lambda a: a.noise_strength)
            if not sub:
                continue
            xs = [a.noise_strength for a in sub]
            ys = [a.per_gate_decay_rate for a in sub]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.3,
                    label=algo, color=ALGO_COLORS.get(algo))
        ax.set_xlabel(r"Noise Strength $p$")
        ax.set_title(NOISE_LABELS[nt])
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
    axes[0].set_ylabel(r"Per-Gate Decay $-\ln(\mathcal{F})/G$")
    fig.suptitle(r"Per-Gate Decay Rate — Structure-Dependent Vulnerability",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save:
        _save_fig(fig, "fig4_per_gate_decay")
    plt.close(fig)


# ── FIG 5: Error threshold bar chart ─────────────────────────────

def plot_error_thresholds(agg: List[AggregatedResult], save=True):
    algorithms = sorted(set(a.algorithm for a in agg))
    thresholds = {}
    for algo in algorithms:
        for nt in NOISE_TYPES:
            sub = sorted([a for a in agg
                          if a.algorithm == algo and a.noise_type == nt],
                         key=lambda a: a.noise_strength)
            if len(sub) < 2:
                continue
            ys = [a.fidelity_mean for a in sub]
            xs = [a.noise_strength for a in sub]
            t = find_error_threshold(ys, xs, 0.5)
            thresholds[(algo, nt)] = t if t is not None else 0.0
    if not thresholds:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(algorithms) * 2.5), 5))
    x = np.arange(len(algorithms))
    w = 0.8 / len(NOISE_TYPES)
    for k, nt in enumerate(NOISE_TYPES):
        vals = [thresholds.get((algo, nt), 0.0) for algo in algorithms]
        bars = ax.bar(x + k * w, vals, w, label=NOISE_LABELS[nt],
                      color=NOISE_COLORS[nt], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(r"Noise Threshold $p^*$ at $\mathcal{F}=0.5$")
    ax.set_title(r"Error Threshold Where $\mathcal{F}$ Drops Below 0.5",
                 fontweight="bold")
    ax.set_xticks(x + w * len(NOISE_TYPES) / 2)
    ax.set_xticklabels(algorithms, rotation=20, ha="right")
    ax.legend(fontsize=7)
    plt.tight_layout()
    if save:
        _save_fig(fig, "fig5_error_thresholds")
    plt.close(fig)


# ── FIG 6: Scaling study ─────────────────────────────────────────

def plot_scaling(scaling: List[ScalingResult], save=True):
    groups = defaultdict(list)
    for r in scaling:
        groups[(r.algorithm, r.parameter)].append(r)
    n = len(groups)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, ((algo, param), items) in zip(axes, groups.items()):
        for nt in NOISE_TYPES:
            sub = sorted([r for r in items if r.noise_type == nt],
                         key=lambda r: r.circuit_depth)
            if not sub:
                continue
            xs = [r.circuit_depth for r in sub]
            ys = [r.fidelity_mean for r in sub]
            lo = [r.fidelity_ci_lo for r in sub]
            hi = [r.fidelity_ci_hi for r in sub]
            c = NOISE_COLORS[nt]
            ax.plot(xs, ys, marker=NOISE_MARKERS[nt], linestyle=NOISE_LINES[nt],
                    color=c, label=NOISE_LABELS[nt], markersize=4, linewidth=1.3)
            ax.fill_between(xs, lo, hi, alpha=0.12, color=c)
        ax.set_xlabel("Circuit Depth")
        ax.set_title(f"{algo} ({param} scaling)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, framealpha=0.8)
    axes[0].set_ylabel(r"State Fidelity $\mathcal{F}$")
    fig.suptitle(r"Fidelity vs Circuit Depth — Scaling (shaded: 95% CI)",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save:
        _save_fig(fig, "fig6_depth_scaling")
    plt.close(fig)


# ── FIG 7: Combined overlay by noise type ────────────────────────

def plot_combined_overlay(agg: List[AggregatedResult], save=True):
    algorithms = sorted(set(a.algorithm for a in agg))
    fig, axes = plt.subplots(1, len(NOISE_TYPES),
                             figsize=(4.8 * len(NOISE_TYPES), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, nt in zip(axes, NOISE_TYPES):
        for algo in algorithms:
            sub = sorted([a for a in agg
                          if a.algorithm == algo and a.noise_type == nt],
                         key=lambda a: a.noise_strength)
            if not sub:
                continue
            xs = np.array([a.noise_strength for a in sub])
            ys = np.array([a.fidelity_mean for a in sub])
            lo = np.array([a.fidelity_ci_lo for a in sub])
            hi = np.array([a.fidelity_ci_hi for a in sub])
            c = ALGO_COLORS.get(algo)
            ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.2,
                    label=algo, color=c)
            ax.fill_between(xs, lo, hi, alpha=0.1, color=c)
        ax.set_xlabel(r"Noise Strength $p$")
        ax.set_title(NOISE_LABELS[nt])
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel(r"State Fidelity $\mathcal{F}$")
    # Shared legend outside plot area — avoids obscuring curves
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               bbox_to_anchor=(0.5, -0.10), fontsize=8, framealpha=0.9,
               edgecolor="gray")
    fig.suptitle(r"Cross-Algorithm Comparison by Noise Type (95% CI)",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    if save:
        _save_fig(fig, "fig7_combined_overlay")
    plt.close(fig)


# ── FIG 8: Fit summary — per-gate decay bar chart ────────────────

def plot_fit_summary(fits: List[FitResult], save=True):
    converged = [f for f in fits if f.converged]
    if not converged:
        return
    algorithms = sorted(set(f.algorithm for f in converged))
    nts = sorted(set(f.noise_type for f in converged))

    fig, ax = plt.subplots(figsize=(max(8, len(algorithms) * 2.5), 5))
    x = np.arange(len(algorithms))
    w = 0.8 / max(len(nts), 1)
    for k, nt in enumerate(nts):
        vals = []
        for algo in algorithms:
            m = [f for f in converged if f.algorithm == algo and f.noise_type == nt]
            vals.append(m[0].lam_per_gate if m else 0.0)
        bars = ax.bar(x + k * w, vals, w, label=NOISE_LABELS.get(nt, nt),
                      color=NOISE_COLORS.get(nt, "gray"), alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{v:.4f}", ha="center", va="bottom",
                        fontsize=6, rotation=45)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(r"Per-Gate Decay Rate $\lambda / G$")
    ax.set_title(r"Per-Gate Decay from Fit $\mathcal{F}(p)=Ae^{-\lambda p}+(1-A)/2^n$",
                 fontweight="bold")
    ax.set_xticks(x + w * len(nts) / 2)
    ax.set_xticklabels(algorithms, rotation=20, ha="right")
    ax.legend(fontsize=7)
    plt.tight_layout()
    if save:
        _save_fig(fig, "fig8_per_gate_decay_rates")
    plt.close(fig)


# ── FIG 9: Success probability with CI ───────────────────────────

def plot_success_probability_ci(agg: List[AggregatedResult], save=True):
    algorithms = sorted(set(a.algorithm for a in agg))
    n = len(algorithms)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, algo in zip(axes, algorithms):
        for nt in NOISE_TYPES:
            sub = sorted([a for a in agg
                          if a.algorithm == algo and a.noise_type == nt],
                         key=lambda a: a.noise_strength)
            if not sub:
                continue
            xs = np.array([a.noise_strength for a in sub])
            ys = np.array([a.succ_prob_mean for a in sub])
            lo = np.array([a.succ_prob_ci_lo for a in sub])
            hi = np.array([a.succ_prob_ci_hi for a in sub])
            c = NOISE_COLORS[nt]
            ax.plot(xs, ys, marker=NOISE_MARKERS[nt], linestyle=NOISE_LINES[nt],
                    color=c, label=NOISE_LABELS[nt], markersize=3.5, linewidth=1.3)
            ax.fill_between(xs, lo, hi, alpha=0.15, color=c)
        ax.set_xlabel(r"Noise Strength $p$")
        ax.set_title(algo)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Success Prob. / Approx. Ratio")
    # Shared legend outside plot area — avoids obscuring curves
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(NOISE_TYPES),
               bbox_to_anchor=(0.5, -0.10), fontsize=8, framealpha=0.9,
               edgecolor="gray")
    fig.suptitle(r"Success Probability vs Noise (95% CI)",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    if save:
        _save_fig(fig, "fig9_success_probability")
    plt.close(fig)


# ── FIG 10: CI convergence vs ensemble size ──────────────────────

def plot_ci_convergence(raw: List[BenchmarkResult], save=True):
    """Show CI width stabilises as N_ensemble grows (validates sample size)."""
    test_configs = [
        ("Teleportation", "depolarizing", 0.1),
        ("Grover-3q", "depolarizing", 0.1),
        ("QAOA-p1-4q", "depolarizing", 0.1),
    ]
    ensemble_sizes = list(range(3, N_ENSEMBLE + 1))

    fig, ax = plt.subplots(figsize=(6, 4))
    for algo, nt, ns in test_configs:
        points = [r for r in raw
                  if r.algorithm == algo and r.noise_type == nt
                  and abs(r.noise_strength - ns) < 0.01]
        if len(points) < 5:
            continue
        fids_all = np.array([r.fidelity for r in points[:N_ENSEMBLE]])
        widths = []
        for n_sub in ensemble_sizes:
            if n_sub > len(fids_all):
                break
            sub = fids_all[:n_sub]
            ci = _bootstrap_ci(sub)
            widths.append(ci[1] - ci[0])
        ax.plot(ensemble_sizes[:len(widths)], widths,
                marker="o", markersize=3, linewidth=1.3, label=algo)
    ax.set_xlabel(r"Ensemble Size $N$")
    ax.set_ylabel("95% CI Width")
    ax.set_title("CI Convergence vs Ensemble Size", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    plt.tight_layout()
    if save:
        _save_fig(fig, "fig10_ci_convergence")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# §11  DATA EXPORT
# ══════════════════════════════════════════════════════════════════════

def export_all(raw: List[BenchmarkResult], agg: List[AggregatedResult],
               fits: List[FitResult], scaling: List[ScalingResult]):
    """Save all data to JSON and CSV files."""
    data = {
        "raw_results": [asdict(r) for r in raw],
        "aggregated_results": [asdict(a) for a in agg],
        "fit_results": [asdict(f) for f in fits],
        "scaling_results": [asdict(s) for s in scaling],
        "metadata": {
            "shots": SHOTS, "n_ensemble": N_ENSEMBLE,
            "bootstrap_ci": BOOTSTRAP_CI,
            "noise_strengths": NOISE_STRENGTHS,
            "base_seed": BASE_SEED,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    jp = RESULTS_DIR / "benchmark_data_v2.json"
    with open(jp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  💾 {jp}")

    # CSV — aggregated
    cp = RESULTS_DIR / "aggregated_results.csv"
    with open(cp, "w") as f:
        hdr = list(asdict(agg[0]).keys())
        f.write(",".join(hdr) + "\n")
        for a in agg:
            d = asdict(a)
            f.write(",".join(str(d[k]) for k in hdr) + "\n")
    print(f"  💾 {cp}")

    # CSV — fit results
    fp = RESULTS_DIR / "fit_results.csv"
    with open(fp, "w") as f:
        hdr = list(asdict(fits[0]).keys())
        f.write(",".join(hdr) + "\n")
        for fr in fits:
            d = asdict(fr)
            f.write(",".join(str(d[k]) for k in hdr) + "\n")
    print(f"  💾 {fp}")


# ══════════════════════════════════════════════════════════════════════
# §12  ANALYSIS SUMMARY
# ══════════════════════════════════════════════════════════════════════

def print_analysis(agg: List[AggregatedResult], fits: List[FitResult]):
    """Comprehensive analysis with the novel per-gate decay angle."""
    algorithms = sorted(set(a.algorithm for a in agg))

    print("\n" + "═" * 78)
    print("  ANALYSIS SUMMARY")
    print("═" * 78)

    # Error thresholds
    print(f"\n  Error Thresholds (noise strength p* at F = 0.5):")
    hdr = f"  {'Algorithm':<18}"
    for nt in NOISE_TYPES:
        hdr += f"  {NOISE_LABELS[nt]:>18}"
    print(hdr)
    print("  " + "─" * 78)

    threshold_data = {}
    for algo in algorithms:
        row = f"  {algo:<18}"
        for nt in NOISE_TYPES:
            sub = sorted([a for a in agg
                          if a.algorithm == algo and a.noise_type == nt],
                         key=lambda a: a.noise_strength)
            ys = [a.fidelity_mean for a in sub]
            xs = [a.noise_strength for a in sub]
            t = find_error_threshold(ys, xs, 0.5)
            threshold_data[(algo, nt)] = t
            row += f"  {t:>18.4f}" if t else f"  {'> 0.50':>18}"
        print(row)

    # Fit summary
    print(f"\n  Exponential Fit: F(p) = A·exp(-λp) + (1-A)/2^n")
    print(f"  {'Algorithm':<18} {'Noise':<22} {'A':>6} {'λ':>8} "
          f"{'λ/G':>8} {'λ/d':>8} {'R²':>8}")
    print("  " + "─" * 78)
    for f in sorted(fits, key=lambda f: (f.algorithm, f.noise_type)):
        if f.converged:
            print(f"  {f.algorithm:<18} {f.noise_type:<22} "
                  f"{f.A:>6.3f} {f.lam:>8.3f} "
                  f"{f.lam_per_gate:>8.5f} {f.lam_per_depth:>8.5f} "
                  f"{f.r_squared:>8.4f}")

    # Per-gate decay rate analysis
    print(f"\n  {'═' * 78}")
    print("  Per-Gate Decay Rate Analysis (λ/G)")
    print(f"  {'═' * 78}")

    for nt in NOISE_TYPES:
        nt_fits = {f.algorithm: f for f in fits
                   if f.noise_type == nt and f.converged}
        if not nt_fits:
            continue
        print(f"\n  Under {NOISE_LABELS[nt]} (per-gate λ/G):")
        for algo, f in sorted(nt_fits.items()):
            print(f"    {algo:<18}: λ/G = {f.lam_per_gate:.5f}  "
                  f"(depth={f.circuit_depth}, gates={f.gate_count})")

        tele = nt_fits.get("Teleportation")
        if tele:
            for algo, f in nt_fits.items():
                if algo == "Teleportation":
                    continue
                ratio = f.lam_per_gate / max(tele.lam_per_gate, 1e-15)
                word = "MORE" if ratio > 1 else "LESS"
                print(f"  → {algo} has {ratio:.1f}× {word} per-gate decay "
                      f"than Teleportation")

    # Most vulnerable noise channel per algorithm (by per-gate λ/G)
    print(f"\n  Most Vulnerable Noise Channel (per-gate):")
    for algo in algorithms:
        worst_nt, worst_rate = None, 0
        for f in fits:
            if f.algorithm == algo and f.converged and f.lam_per_gate > worst_rate:
                worst_rate = f.lam_per_gate
                worst_nt = f.noise_type
        if worst_nt:
            print(f"    {algo:<18}: {NOISE_LABELS[worst_nt]} "
                  f"(λ/G = {worst_rate:.5f})")

    print("\n" + "═" * 78)


# ══════════════════════════════════════════════════════════════════════
# §13  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()

    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║    NOISE-AWARE QUANTUM ALGORITHM BENCHMARKING FRAMEWORK               ║")
    print("╠═══════════════════════════════════════════════════════════════════════╣")
    print("║  Algorithms : Teleportation · Grover (3q,5q) · QAOA-opt (4q,6q)       ║")
    print("║  Noise      : Depolarizing · Amp.Damp · Phase.Damp · Thermal Relax    ║")
    print("║  Features   : VQE-optimized QAOA · N=25 ensemble · bootstrap CI       ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Ensemble size : {N_ENSEMBLE} seeds/point")
    print(f"  Shots/run     : {SHOTS}")
    print(f"  Noise points  : {len(NOISE_STRENGTHS)} "
          f"({NOISE_STRENGTHS[0]}..{NOISE_STRENGTHS[-1]})")
    print()

    # [1/7] Backend
    print("[1/7] Detecting backend...")
    backend, backend_label = detect_gpu_backend()
    print()

    # [2/7] Main noise sweep with ensemble
    n_algo_configs = 1 + len(GROVER_BENCH_QUBITS) + len(QAOA_BENCH_CONFIGS)
    n_configs = len(NOISE_TYPES) * len(NOISE_STRENGTHS) * n_algo_configs
    total_sims = n_configs * N_ENSEMBLE
    print(f"[2/7] Noise sweep "
          f"({len(NOISE_TYPES)} noise × {len(NOISE_STRENGTHS)} strengths × "
          f"{n_algo_configs} algos × {N_ENSEMBLE} seeds = ~{total_sims} sims)...")

    raw_results: List[BenchmarkResult] = []
    done = 0

    for nt in NOISE_TYPES:
        for strength in NOISE_STRENGTHS:
            for s in range(N_ENSEMBLE):
                seed = BASE_SEED + s * 1000

                # Teleportation
                raw_results.append(
                    run_teleportation_single(nt, strength, backend, seed))

                # Grover at multiple qubit counts
                for nq in GROVER_BENCH_QUBITS:
                    raw_results.append(
                        run_grover_single(nt, strength, backend, nq, seed))

                # QAOA at multiple configs
                for nq, p in QAOA_BENCH_CONFIGS:
                    raw_results.append(
                        run_qaoa_single(nt, strength, backend, nq, p, seed))

                done += n_algo_configs

            pct = done / (total_sims) * 100
            print(f"      {nt:<22} p={strength:.2f}  "
                  f"({done}/{total_sims}, {pct:.0f}%)")

    print(f"  ✓ {len(raw_results)} raw data points\n")

    # [3/7] Aggregate
    print("[3/7] Aggregating with bootstrap CI...")
    agg = aggregate_results(raw_results)
    print(f"  ✓ {len(agg)} aggregated points\n")

    # [4/7] Fit
    print("[4/7] Fitting exponential decay model...")
    fits = fit_fidelity_decay(agg)
    n_conv = sum(1 for f in fits if f.converged)
    print(f"  ✓ {n_conv}/{len(fits)} fits converged\n")

    # [5/7] Scaling
    print("[5/7] Running scaling studies (Grover 2→8q, QAOA p=1→6)...")
    scaling = run_scaling_study(backend)
    print(f"  ✓ {len(scaling)} scaling points\n")

    # [6/7] Figures
    print("[6/7] Generating 10 publication figures (PNG + PDF)...")
    plot_fidelity_vs_noise_ci(agg, fits)
    plot_depth_normalized(agg)
    plot_robustness_heatmap(agg)
    plot_per_gate_decay(agg)
    plot_error_thresholds(agg)
    plot_scaling(scaling)
    plot_combined_overlay(agg)
    plot_fit_summary(fits)
    plot_success_probability_ci(agg)
    plot_ci_convergence(raw_results)
    print()

    # [7/7] Export
    print("[7/7] Exporting data...")
    export_all(raw_results, agg, fits, scaling)
    print()

    # Analysis
    print_analysis(agg, fits)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Total time  : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Backend     : {backend_label}")
    print(f"  Results     : {RESULTS_DIR.resolve()}")
    print(f"  Raw points  : {len(raw_results)}")
    print(f"  Figures     : 10 (PNG + PDF)")
    print("\n  Done! ✓\n")


if __name__ == "__main__":
    main()
