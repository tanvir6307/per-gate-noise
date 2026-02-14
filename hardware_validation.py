#!/usr/bin/env python3
"""
Hardware Validation for Noise-Aware Quantum Algorithm Benchmarking
══════════════════════════════════════════════════════════════════

Runs Teleportation-3q, Grover-3q, and QAOA-4q on IBM Quantum hardware
(ibm_marrakesh, 156-qubit Heron processor) and compares measured
success probabilities with calibration-based noise simulation.

Produces:
  - results/hardware_validation.json   (raw data)
  - results/fig11_hardware_validation.png / .pdf  (validation figure)
"""

import numpy as np
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# ── Qiskit imports ────────────────────────────────────────────────
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
IBM_TOKEN = "RhKu4mxf3N8yA0RRABDWyTxYAia2Z4faY5Cp9rtWE54p"
CHANNEL   = "ibm_quantum_platform"
BACKEND   = "ibm_marrakesh"
SHOTS     = 8192
RESULTS_DIR = Path("results")
BASE_SEED = 42


# ══════════════════════════════════════════════════════════════════
# §1  CIRCUIT BUILDERS  (match noise_aware.py exactly)
# ══════════════════════════════════════════════════════════════════

def build_teleportation_measured(theta: float, phi: float) -> QuantumCircuit:
    """
    Deferred-measurement teleportation circuit with final measurements.
    Input state: Ry(θ)Rz(φ)|0⟩ on q0.
    Bell pair on q1-q2.  Deferred CX/CZ corrections.
    All 3 qubits measured at the end.
    """
    qc = QuantumCircuit(3, 3, name=f"Teleport(θ={theta:.2f},φ={phi:.2f})")
    # Prepare input state on q0
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    # Bell pair: q1-q2
    qc.h(1)
    qc.cx(1, 2)
    # Bell measurement (unitary part): CX q0→q1, H q0
    qc.cx(0, 1)
    qc.h(0)
    # Deferred corrections (replace classically-controlled gates)
    qc.cx(1, 2)
    qc.cz(0, 2)
    # Measure all
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def teleport_ideal_probs(theta: float, phi: float) -> Dict[str, float]:
    """Ideal output distribution for deferred-measurement teleportation.
    
    With deferred measurement, the output qubit (q2) is unconditionally
    in the target state. We compute the full 3-qubit output distribution.
    """
    qc = QuantumCircuit(3, name="ideal")
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(1, 2)
    qc.cz(0, 2)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()
    return probs


def teleport_success_prob(counts: Dict[str, int], theta: float, phi: float) -> float:
    """Success probability for teleportation: overlap of q2 marginal
    with ideal output state.
    
    The ideal output state on q2 is |ψ⟩ = Ry(θ)Rz(φ)|0⟩.
    P(q2=0) for ideal = cos²(θ/2), P(q2=1) = sin²(θ/2).
    We compute the q2 marginal from counts and compute fidelity.
    """
    total = sum(counts.values())
    # q2 marginal: in Qiskit bit ordering, q2 is the leftmost bit of a 3-bit string
    # Qiskit uses little-endian: bit string "abc" means q0=c, q1=b, q2=a
    p_q2_0 = sum(v for k, v in counts.items() if k[0] == '0') / total
    p_q2_1 = 1.0 - p_q2_0
    # Ideal q2 state: Ry(θ)Rz(φ)|0⟩ → cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
    # P(0) = cos²(θ/2), P(1) = sin²(θ/2)
    ideal_p0 = np.cos(theta / 2) ** 2
    ideal_p1 = np.sin(theta / 2) ** 2
    # Classical fidelity: F = (√p0·√q0 + √p1·√q1)²  (Bhattacharyya)
    fidelity = (np.sqrt(p_q2_0 * ideal_p0) + np.sqrt(p_q2_1 * ideal_p1)) ** 2
    return float(fidelity)


# ── Grover ────────────────────────────────────────────────────────

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


def build_grover_measured(n_qubits: int = 3, marked: int = 5,
                          n_iterations: Optional[int] = None) -> QuantumCircuit:
    """Grover circuit with measurements. Same construction as noise_aware.py."""
    N = 2 ** n_qubits
    if n_iterations is None:
        n_iterations = max(1, int(np.round(np.pi / 4 * np.sqrt(N))))
    oracle   = _grover_oracle(n_qubits, marked)
    diffuser = _grover_diffuser(n_qubits)
    qc = QuantumCircuit(n_qubits, n_qubits, name=f"Grover-{n_qubits}q(t={marked})")
    qc.h(range(n_qubits))
    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def grover_success_prob(counts: Dict[str, int], n_qubits: int, marked: int) -> float:
    """P(measuring the marked state)."""
    total = sum(counts.values())
    target_str = format(marked, f"0{n_qubits}b")[::-1]  # little-endian
    # Actually, Qiskit uses little-endian internally. The counts keys are
    # in big-endian by convention: key "101" means q2=1, q1=0, q0=1
    # So target_str should be big-endian representation
    target_str = format(marked, f"0{n_qubits}b")  # big-endian
    return counts.get(target_str, 0) / total


# ── QAOA (MaxCut) ────────────────────────────────────────────────

def _generate_graph(n_qubits: int, seed: int) -> List[Tuple[int, int]]:
    """Generate random graph edges for MaxCut (matches noise_aware.py)."""
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(n_qubits)
             for j in range(i + 1, n_qubits)
             if rng.random() > 0.4]
    return edges


def _maxcut_value(bitstring: str, edges: List[Tuple[int, int]]) -> int:
    """Number of edges cut by bitstring."""
    return sum(1 for i, j in edges if bitstring[i] != bitstring[j])


def _maxcut_qaoa_circuit(edges, n_qubits, gamma, beta, p=1) -> QuantumCircuit:
    """Build p-layer QAOA circuit for MaxCut (matches noise_aware.py)."""
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


def optimize_qaoa_params(edges, n_qubits, p=1):
    """Grid search + Nelder-Mead optimization (matches noise_aware.py)."""
    from scipy.optimize import minimize

    # Optimal cut for normalization
    n = n_qubits
    opt_cut = 0
    for s in range(2**n):
        bs = format(s, f"0{n}b")
        cut = _maxcut_value(bs, edges)
        opt_cut = max(opt_cut, cut)

    def _neg_approx_ratio(params):
        g, b = float(params[0]), float(params[1])
        qc = _maxcut_qaoa_circuit(edges, n_qubits, g, b, p)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()
        exp_cut = sum(_maxcut_value(bs, edges) * prob
                      for bs, prob in probs.items())
        return -exp_cut / max(opt_cut, 1)

    best_val, best_params = 0.0, (np.pi / 4, np.pi / 8)
    for g in np.linspace(0.05, np.pi, 10):
        for b in np.linspace(0.05, np.pi / 2, 10):
            val = _neg_approx_ratio([g, b])
            if val < best_val:
                best_val = val
                best_params = (g, b)

    res = minimize(_neg_approx_ratio, best_params, method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4})
    return float(res.x[0]), float(res.x[1])


def build_qaoa_measured(n_qubits: int = 4, p: int = 1,
                        seed: int = BASE_SEED) -> Tuple[QuantumCircuit, List, int]:
    """Build QAOA with measurements. Returns (circuit, edges, optimal_cut)."""
    edges = _generate_graph(n_qubits, seed)
    gamma, beta = optimize_qaoa_params(edges, n_qubits, p)
    qc = _maxcut_qaoa_circuit(edges, n_qubits, gamma, beta, p)
    # Add measurements
    qc.add_register(qc._create_creg(n_qubits, "c"))
    qc.measure(range(n_qubits), range(n_qubits))
    # Compute optimal cut
    opt_cut = 0
    for s in range(2**n_qubits):
        bs = format(s, f"0{n_qubits}b")
        cut = _maxcut_value(bs, edges)
        opt_cut = max(opt_cut, cut)
    return qc, edges, opt_cut


def qaoa_approx_ratio(counts: Dict[str, int],
                      edges: List[Tuple[int, int]],
                      opt_cut: int) -> float:
    """Compute approximation ratio from measurement counts."""
    total = sum(counts.values())
    n = max(max(i, j) for i, j in edges) + 1 if edges else 0
    exp_cut = 0.0
    for bs, cnt in counts.items():
        # Qiskit returns big-endian bitstrings
        # Need to reverse for our _maxcut_value which indexes by position
        bs_rev = bs[::-1]  # to position-indexed
        cut = _maxcut_value(bs_rev, edges)
        exp_cut += cut * cnt
    exp_cut /= total
    return exp_cut / max(opt_cut, 1)


# ══════════════════════════════════════════════════════════════════
# §2  MAIN VALIDATION PIPELINE
# ══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print("  HARDWARE VALIDATION — Noise-Aware Quantum Benchmarking")
    print("=" * 70)

    # ── Step 1: Connect to IBM Quantum ────────────────────────────
    print("\n[1/7] Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel=CHANNEL, token=IBM_TOKEN)
    backend = service.backend(BACKEND)
    print(f"  Backend: {backend.name}")
    print(f"  Qubits:  {backend.num_qubits}")
    print(f"  Status:  operational={backend.status().operational}, "
          f"pending_jobs={backend.status().pending_jobs}")

    # ── Step 2: Build circuits ────────────────────────────────────
    print("\n[2/7] Building circuits...")
    circuits = []
    labels = []
    metadata = []

    # 2a. Teleportation circuits (5 different input states)
    teleport_states = [
        (0.0,      0.0,     "|0⟩"),
        (np.pi,    0.0,     "|1⟩"),
        (np.pi/2,  0.0,     "|+⟩"),
        (np.pi/2,  np.pi/2, "|+i⟩"),
        (np.pi/3,  np.pi/4, "Ry(π/3)Rz(π/4)|0⟩"),
    ]
    for theta, phi, label in teleport_states:
        qc = build_teleportation_measured(theta, phi)
        circuits.append(qc)
        labels.append(f"Teleport: {label}")
        metadata.append({
            "algorithm": "Teleportation",
            "theta": theta, "phi": phi,
            "input_label": label,
        })
        print(f"    Teleport {label}: {qc.num_qubits}q, "
              f"depth={qc.depth()}, gates={qc.size()}")

    # 2b. Grover-3q circuits (5 different marked items)
    grover_targets = [0, 3, 5, 6, 7]
    for target in grover_targets:
        qc = build_grover_measured(3, target)
        circuits.append(qc)
        target_str = format(target, "03b")
        labels.append(f"Grover: |{target_str}⟩")
        metadata.append({
            "algorithm": "Grover-3q",
            "marked": target,
            "target_bitstring": target_str,
            "n_iterations": max(1, int(np.round(np.pi / 4 * np.sqrt(8)))),
        })
        print(f"    Grover |{target_str}⟩: {qc.num_qubits}q, "
              f"depth={qc.depth()}, gates={qc.size()}")

    # 2c. QAOA-4q circuits (3 different random graphs)
    qaoa_seeds = [42, 123, 256]
    for seed in qaoa_seeds:
        qc, edges, opt_cut = build_qaoa_measured(4, 1, seed)
        circuits.append(qc)
        labels.append(f"QAOA: seed={seed}")
        metadata.append({
            "algorithm": "QAOA-4q",
            "seed": seed,
            "edges": edges,
            "optimal_cut": opt_cut,
        })
        print(f"    QAOA seed={seed}: {qc.num_qubits}q, "
              f"depth={qc.depth()}, gates={qc.size()}, "
              f"edges={len(edges)}, opt_cut={opt_cut}")

    n_circuits = len(circuits)
    print(f"\n  Total circuits: {n_circuits}")

    # ── Step 3: Transpile for backend ─────────────────────────────
    print("\n[3/7] Transpiling to hardware basis gates...")
    transpiled = transpile(
        circuits, backend=backend,
        optimization_level=3,
        seed_transpiler=42,
    )
    for i, tc in enumerate(transpiled):
        depth = tc.depth()
        ops = dict(tc.count_ops())
        n_2q = ops.get("cz", 0) + ops.get("ecr", 0) + ops.get("cx", 0)
        print(f"    {labels[i]}: depth={depth}, "
              f"2q_gates={n_2q}, total_gates={tc.size()}")
        metadata[i]["transpiled_depth"] = depth
        metadata[i]["transpiled_2q_gates"] = n_2q
        metadata[i]["transpiled_total_gates"] = tc.size()

    # ── Step 4: Submit to hardware ────────────────────────────────
    print("\n[4/7] Submitting to IBM Quantum hardware...")
    print(f"  {n_circuits} circuits × {SHOTS} shots = "
          f"{n_circuits * SHOTS:,} total shots")
    t_submit = time.time()

    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=SHOTS)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    print(f"  Waiting for results...")

    # Poll for completion
    while True:
        status = job.status()
        print(f"    Status: {status}  [{time.time()-t_submit:.0f}s elapsed]")
        if status in ("DONE", "ERROR", "CANCELLED"):
            break
        time.sleep(10)

    if job.status() != "DONE":
        print(f"  *** Job failed with status: {job.status()} ***")
        sys.exit(1)

    hw_result = job.result()
    t_hw = time.time() - t_submit
    print(f"  Hardware execution completed in {t_hw:.1f}s")

    # Extract counts from hardware results
    hw_counts_list = []
    for i in range(n_circuits):
        pub_result = hw_result[i]
        # SamplerV2 returns BitArray in pub_result.data
        # Get the classical register - try common names
        creg_data = None
        for attr in ["c", "meas", "c0"]:
            if hasattr(pub_result.data, attr):
                creg_data = getattr(pub_result.data, attr)
                break
        if creg_data is None:
            # Try to get any available attribute
            data_attrs = [a for a in dir(pub_result.data)
                          if not a.startswith("_")]
            if data_attrs:
                creg_data = getattr(pub_result.data, data_attrs[0])
        counts = creg_data.get_counts()
        hw_counts_list.append(counts)
        top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
        print(f"    {labels[i]}: top outcomes = "
              f"{', '.join(f'{k}:{v}' for k,v in top3)}")

    # ── Step 5: Simulate with calibration-based noise model ──────
    print("\n[5/7] Simulating with calibration-based noise model...")
    t_sim = time.time()
    # Build noisy simulator that mirrors the backend topology exactly
    noisy_sim = AerSimulator.from_backend(backend)
    # Run the already-transpiled circuits directly (no re-transpilation)
    sim_job = noisy_sim.run(transpiled, shots=SHOTS, seed_simulator=42)
    sim_result = sim_job.result()
    sim_counts_list = [sim_result.get_counts(i) for i in range(n_circuits)]
    t_sim_elapsed = time.time() - t_sim
    print(f"  Simulation completed in {t_sim_elapsed:.1f}s")

    # ── Step 6: Also run ideal (noiseless) simulation ─────────────
    print("\n[6/7] Running ideal (noiseless) simulation...")
    ideal_sim = AerSimulator()
    ideal_transpiled = transpile(circuits, backend=ideal_sim, seed_transpiler=42)
    ideal_job = ideal_sim.run(ideal_transpiled, shots=SHOTS, seed_simulator=42)
    ideal_result = ideal_job.result()
    ideal_counts_list = [ideal_result.get_counts(i) for i in range(n_circuits)]

    # ── Step 7: Compute success probabilities and compare ─────────
    print("\n[7/7] Computing success metrics...")
    results = []

    for i in range(n_circuits):
        algo = metadata[i]["algorithm"]

        if algo == "Teleportation":
            theta, phi = metadata[i]["theta"], metadata[i]["phi"]
            sp_hw    = teleport_success_prob(hw_counts_list[i], theta, phi)
            sp_sim   = teleport_success_prob(sim_counts_list[i], theta, phi)
            sp_ideal = teleport_success_prob(ideal_counts_list[i], theta, phi)
        elif algo == "Grover-3q":
            marked = metadata[i]["marked"]
            sp_hw    = grover_success_prob(hw_counts_list[i], 3, marked)
            sp_sim   = grover_success_prob(sim_counts_list[i], 3, marked)
            sp_ideal = grover_success_prob(ideal_counts_list[i], 3, marked)
        elif algo == "QAOA-4q":
            edges = metadata[i]["edges"]
            opt_cut = metadata[i]["optimal_cut"]
            sp_hw    = qaoa_approx_ratio(hw_counts_list[i], edges, opt_cut)
            sp_sim   = qaoa_approx_ratio(sim_counts_list[i], edges, opt_cut)
            sp_ideal = qaoa_approx_ratio(ideal_counts_list[i], edges, opt_cut)
        else:
            continue

        results.append({
            "label": labels[i],
            "algorithm": algo,
            "metadata": {k: v for k, v in metadata[i].items()
                         if k not in ("edges",)},
            "success_prob_hardware": round(sp_hw, 6),
            "success_prob_simulated": round(sp_sim, 6),
            "success_prob_ideal": round(sp_ideal, 6),
            "hw_counts_top5": dict(sorted(hw_counts_list[i].items(),
                                          key=lambda x: -x[1])[:5]),
            "sim_counts_top5": dict(sorted(sim_counts_list[i].items(),
                                           key=lambda x: -x[1])[:5]),
        })

        print(f"  {labels[i]:30s} │ Ideal={sp_ideal:.4f} │ "
              f"Sim={sp_sim:.4f} │ HW={sp_hw:.4f} │ "
              f"Δ(sim-hw)={sp_sim-sp_hw:+.4f}")

    # ── Aggregate statistics ──────────────────────────────────────
    hw_vals  = np.array([r["success_prob_hardware"] for r in results])
    sim_vals = np.array([r["success_prob_simulated"] for r in results])

    # Pearson correlation
    if len(hw_vals) > 2:
        corr = np.corrcoef(sim_vals, hw_vals)[0, 1]
    else:
        corr = float("nan")
    mae = np.mean(np.abs(sim_vals - hw_vals))
    rmse = np.sqrt(np.mean((sim_vals - hw_vals) ** 2))

    print(f"\n  ── Summary Statistics ──")
    print(f"  Pearson r (sim vs hw):   {corr:.4f}")
    print(f"  MAE (sim vs hw):         {mae:.4f}")
    print(f"  RMSE (sim vs hw):        {rmse:.4f}")
    print(f"  Mean HW success:         {np.mean(hw_vals):.4f}")
    print(f"  Mean simulated success:  {np.mean(sim_vals):.4f}")

    # ── Save results ──────────────────────────────────────────────
    output = {
        "timestamp": datetime.now().isoformat(),
        "backend": BACKEND,
        "job_id": job_id,
        "shots": SHOTS,
        "n_circuits": n_circuits,
        "hw_execution_time_s": round(t_hw, 1),
        "statistics": {
            "pearson_r": round(corr, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        },
        "calibration_summary": {
            "note": "Retrieved at runtime from backend.target",
        },
        "results": results,
    }
    out_path = RESULTS_DIR / "hardware_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # ══════════════════════════════════════════════════════════════
    # §3  VALIDATION FIGURE
    # ══════════════════════════════════════════════════════════════
    print("\n  Generating validation figure...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel (a): Scatter plot — Predicted vs Measured ───────────
    ax = axes[0]
    algo_colors = {
        "Teleportation": "#2196F3",  # blue
        "Grover-3q":     "#FF9800",  # orange
        "QAOA-4q":       "#4CAF50",  # green
    }
    algo_markers = {
        "Teleportation": "o",
        "Grover-3q":     "s",
        "QAOA-4q":       "D",
    }

    for r in results:
        algo = r["algorithm"]
        ax.scatter(
            r["success_prob_simulated"],
            r["success_prob_hardware"],
            c=algo_colors[algo], marker=algo_markers[algo],
            s=120, edgecolors="black", linewidths=0.8, zorder=5,
        )

    # Diagonal line (perfect agreement)
    lims = [0, 1.05]
    ax.plot(lims, lims, "k--", lw=1.0, alpha=0.5, label="Perfect agreement")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Simulated Success Probability", fontsize=12)
    ax.set_ylabel("Hardware Success Probability", fontsize=12)
    ax.set_title(f"(a) Predicted vs Measured  (r = {corr:.3f})", fontsize=13)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markeredgecolor="k", markersize=10, label="Teleportation"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF9800",
               markeredgecolor="k", markersize=10, label="Grover-3q"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#4CAF50",
               markeredgecolor="k", markersize=10, label="QAOA-4q"),
        Line2D([0], [0], linestyle="--", color="k", alpha=0.5,
               label="Perfect agreement"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    # Annotation: Pearson r, MAE
    ax.text(0.05, 0.92, f"MAE = {mae:.4f}\nRMSE = {rmse:.4f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # ── Panel (b): Grouped bar chart ──────────────────────────────
    ax = axes[1]
    x = np.arange(n_circuits)
    width = 0.25

    ideal_vals = [r["success_prob_ideal"] for r in results]
    sim_vals_plot = [r["success_prob_simulated"] for r in results]
    hw_vals_plot = [r["success_prob_hardware"] for r in results]

    bars1 = ax.bar(x - width, ideal_vals, width, label="Ideal",
                   color="#E3F2FD", edgecolor="#1565C0", linewidth=0.8)
    bars2 = ax.bar(x, sim_vals_plot, width, label="Simulated (noise model)",
                   color="#FFF3E0", edgecolor="#E65100", linewidth=0.8)
    bars3 = ax.bar(x + width, hw_vals_plot, width, label="Hardware",
                   color="#E8F5E9", edgecolor="#2E7D32", linewidth=0.8)

    ax.set_ylabel("Success Probability / Approx. Ratio", fontsize=12)
    ax.set_title("(b) Per-Circuit Comparison", fontsize=13)
    ax.set_xticks(x)
    # Short labels for x-axis
    short_labels = []
    for r in results:
        algo = r["algorithm"]
        if algo == "Teleportation":
            short_labels.append(r["metadata"]["input_label"])
        elif algo == "Grover-3q":
            short_labels.append(f"|{r['metadata']['target_bitstring']}⟩")
        elif algo == "QAOA-4q":
            short_labels.append(f"s={r['metadata']['seed']}")
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    # Add algorithm group labels
    # Teleportation: indices 0-4, Grover: 5-9, QAOA: 10-12
    n_tel = len(teleport_states)
    n_gro = len(grover_targets)
    n_qao = len(qaoa_seeds)
    for start, count, name in [
        (0, n_tel, "Teleportation"),
        (n_tel, n_gro, "Grover-3q"),
        (n_tel + n_gro, n_qao, "QAOA-4q"),
    ]:
        mid = start + count / 2 - 0.5
        ax.annotate(name, xy=(mid, -0.08), xycoords=("data", "axes fraction"),
                    fontsize=10, fontweight="bold", ha="center",
                    color=algo_colors[name.replace("-3q", "-3q").replace("-4q", "-4q")])

    fig.suptitle(
        f"Hardware Validation on IBM {BACKEND.replace('ibm_', '').capitalize()} "
        f"(Heron, 156 qubits)\n"
        f"{SHOTS:,} shots per circuit  •  {n_circuits} circuits  •  "
        f"Job: {job_id[:12]}...",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig_path = RESULTS_DIR / f"fig11_hardware_validation.{ext}"
        fig.savefig(fig_path, dpi=400, bbox_inches="tight")
        print(f"  Saved {fig_path}")
    plt.close()

    # ── Final summary ─────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  VALIDATION COMPLETE")
    print(f"  Total time:    {elapsed:.1f}s")
    print(f"  Pearson r:     {corr:.4f}")
    print(f"  MAE:           {mae:.4f}")
    print(f"  Data circuits: {n_circuits}")
    print(f"  Output files:  hardware_validation.json, "
          f"fig11_hardware_validation.png/pdf")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
