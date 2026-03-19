#!/usr/bin/env python3
"""
Phase Transitions in Gated State Space Models on the Binary Parity Task.

Studies grokking / sharp phase transitions in a GRU-style gated SSM trained
on the running parity task. Varies hidden dimension and sequence length,
logging detailed metrics and producing publication-quality plots.
"""

import itertools
import os
import pickle
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ─────────────────────────── Constants ───────────────────────────

MAIN_HIDDEN_DIMS = [1, 2, 3, 4, 6, 8, 12, 16]
MAIN_SEQ_LEN = 8
MAIN_SEEDS = 1

SEC_SEQ_LENS = [4, 6, 8, 10, 12, 16]
SEC_HIDDEN_DIM = 4
SEC_SEEDS = 1

NUM_EPOCHS = 3000
BATCH_SIZE = 64
LR = 0.001

TEST_LEN = 16
TEST_SIZE = 500
LARGE_N_THRESHOLD = 12
LARGE_N_SAMPLE = 4096

LOG_INTERVAL = 10  # compute full metrics every N epochs (keeps plots smooth, ~5x faster)

DPI = 200
DEVICE = "cpu"
FIG_DIR = "figures"

# ─────────────────────────── Model ───────────────────────────────


class GatedSSM(nn.Module):
    """Gated recurrent SSM (GRU-style) for scalar binary input."""

    def __init__(self, hidden_dim):
        super().__init__()
        D = hidden_dim
        # Gate parameters
        self.W_z = nn.Parameter(torch.empty(D, D))
        self.U_z = nn.Parameter(torch.empty(D))
        self.b_z = nn.Parameter(torch.zeros(D))
        # Candidate parameters
        self.W_h = nn.Parameter(torch.empty(D, D))
        self.U_h = nn.Parameter(torch.empty(D))
        self.b_h = nn.Parameter(torch.zeros(D))
        # Output
        self.C = nn.Parameter(torch.empty(D))
        self.d = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "W_" in name:
                nn.init.xavier_normal_(p)
            elif "U_" in name or name == "C":
                nn.init.normal_(p, std=0.5)
            # biases stay at zero

    def forward(self, x, return_gates=False):
        """
        Args:
            x: (batch, seq_len) float tensor of 0s and 1s
            return_gates: if True, also return gate activations
        Returns:
            outputs: (batch, seq_len) predicted probabilities
            gates (optional): (batch, seq_len, D) gate activations
        """
        batch, T = x.shape
        D = self.W_z.shape[0]
        h = torch.zeros(batch, D, device=x.device)

        outputs = []
        gates = [] if return_gates else None

        for t in range(T):
            x_t = x[:, t : t + 1]  # (batch, 1)
            z = torch.sigmoid(h @ self.W_z.T + x_t * self.U_z + self.b_z)
            h_tilde = torch.tanh(h @ self.W_h.T + x_t * self.U_h + self.b_h)
            h = (1 - z) * h + z * h_tilde
            y_t = torch.sigmoid(h @ self.C + self.d)  # (batch,)
            outputs.append(y_t)
            if return_gates:
                gates.append(z.detach())

        outputs = torch.stack(outputs, dim=1)  # (batch, T)
        outputs = outputs.clamp(1e-7, 1 - 1e-7)

        if return_gates:
            gates = torch.stack(gates, dim=1)  # (batch, T, D)
            return outputs, gates
        return outputs


# ─────────────────────────── Data ────────────────────────────────


def generate_parity_dataset(seq_len, max_samples=None):
    """Generate binary sequences and their running parity labels.

    If 2^seq_len <= max_samples (or max_samples is None), returns ALL sequences.
    Otherwise, samples max_samples random sequences.
    """
    n_total = 2**seq_len
    if max_samples is None or n_total <= max_samples:
        # Generate all sequences
        seqs = list(itertools.product([0, 1], repeat=seq_len))
        x = torch.tensor(seqs, dtype=torch.float32)
    else:
        x = torch.randint(0, 2, (max_samples, seq_len), dtype=torch.float32)

    targets = torch.cumsum(x, dim=1) % 2
    return x, targets


def generate_test_set(seq_len=16, size=500):
    """Generate a fixed test set (same across all runs)."""
    rng = torch.Generator()
    rng.manual_seed(9999)
    x = torch.randint(0, 2, (size, seq_len), dtype=torch.float32, generator=rng)
    targets = torch.cumsum(x, dim=1) % 2
    return x, targets


# ─────────────────────────── Metrics ─────────────────────────────


def compute_metrics(model, train_x, train_y, test_x, test_y):
    """Compute all logged metrics for one epoch."""
    with torch.no_grad():
        # Training metrics
        train_out, train_gates = model(train_x, return_gates=True)
        train_loss = F.binary_cross_entropy(train_out, train_y).item()
        train_acc = ((train_out > 0.5).float() == train_y).float().mean().item()

        # Test metrics
        test_out = model(test_x)
        test_acc = ((test_out > 0.5).float() == test_y).float().mean().item()

        # Gate analysis
        # train_gates: (batch, T, D), train_x: (batch, T)
        mask_1 = train_x.unsqueeze(-1).expand_as(train_gates)  # (batch, T, D)
        mask_0 = (1 - train_x).unsqueeze(-1).expand_as(train_gates)

        n_ones = mask_1.sum()
        n_zeros = mask_0.sum()

        gate_mean_1 = (train_gates * mask_1).sum() / n_ones if n_ones > 0 else 0.0
        gate_mean_0 = (train_gates * mask_0).sum() / n_zeros if n_zeros > 0 else 0.0

        gate_mean_1 = gate_mean_1.item() if isinstance(gate_mean_1, torch.Tensor) else gate_mean_1
        gate_mean_0 = gate_mean_0.item() if isinstance(gate_mean_0, torch.Tensor) else gate_mean_0
        gate_selectivity = gate_mean_1 - gate_mean_0

        # Spectral analysis of W_h
        eigvals = torch.linalg.eigvals(model.W_h.data)
        spectral_radius = eigvals.abs().max().item()
        eigenvalues = eigvals.tolist()

        # Output weight analysis
        c_norm = torch.norm(model.C.data).item()
        c_values = model.C.data.tolist()

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gate_mean_1": gate_mean_1,
        "gate_mean_0": gate_mean_0,
        "gate_selectivity": gate_selectivity,
        "spectral_radius": spectral_radius,
        "eigenvalues": eigenvalues,
        "c_norm": c_norm,
        "c_values": c_values,
    }


# ─────────────────────────── Training ────────────────────────────


def train_single_run(hidden_dim, seq_len, seed, num_epochs, batch_size, lr, test_x, test_y):
    """Train one model configuration and return full metric history."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    max_samples = LARGE_N_SAMPLE if seq_len > LARGE_N_THRESHOLD else None
    train_x, train_y = generate_parity_dataset(seq_len, max_samples)

    # Model + optimizer
    model = GatedSSM(hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    history = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(train_x.size(0))

        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i : i + batch_size]
            bx, by = train_x[batch_idx], train_y[batch_idx]

            outputs = model(bx)
            loss = F.binary_cross_entropy(outputs, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log metrics every LOG_INTERVAL epochs (the expensive part)
        if epoch % LOG_INTERVAL == 0 or epoch == num_epochs - 1:
            model.eval()
            metrics = compute_metrics(model, train_x, train_y, test_x, test_y)

            for k, v in metrics.items():
                history[k].append(v)
            history["logged_epochs"].append(epoch)

            if epoch % 100 == 0 or epoch == num_epochs - 1:
                print(
                    f"  epoch {epoch:4d} | loss {metrics['train_loss']:.4f} | "
                    f"train_acc {metrics['train_acc']:.4f} | test_acc {metrics['test_acc']:.4f} | "
                    f"gate_sel {metrics['gate_selectivity']:.4f}"
                )

    return dict(history)


# ─────────────────────────── Experiments ─────────────────────────


def run_main_experiment():
    """Main experiment: vary hidden_dim with seq_len=8."""
    results = {}
    test_x, test_y = generate_test_set(TEST_LEN, TEST_SIZE)

    for hidden_dim in MAIN_HIDDEN_DIMS:
        results[hidden_dim] = {}
        n_params = 2 * (hidden_dim**2 + hidden_dim + hidden_dim) + hidden_dim + 1
        print(f"\n=== hidden_dim={hidden_dim} ({n_params} params) ===")

        for seed in range(MAIN_SEEDS):
            print(f"\n--- seed={seed} ---")
            t0 = time.time()
            history = train_single_run(
                hidden_dim, MAIN_SEQ_LEN, seed, NUM_EPOCHS, BATCH_SIZE, LR, test_x, test_y
            )
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s")
            results[hidden_dim][seed] = history

    return results


def run_secondary_experiment():
    """Secondary experiment: vary seq_len with hidden_dim=4."""
    results = {}
    test_x, test_y = generate_test_set(TEST_LEN, TEST_SIZE)

    for seq_len in SEC_SEQ_LENS:
        results[seq_len] = {}
        n_seqs = min(2**seq_len, LARGE_N_SAMPLE) if seq_len > LARGE_N_THRESHOLD else 2**seq_len
        print(f"\n=== seq_len={seq_len} ({n_seqs} sequences) ===")

        for seed in range(SEC_SEEDS):
            print(f"\n--- seed={seed} ---")
            t0 = time.time()
            history = train_single_run(
                SEC_HIDDEN_DIM, seq_len, seed, NUM_EPOCHS, BATCH_SIZE, LR, test_x, test_y
            )
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s")
            results[seq_len][seed] = history

    return results


# ─────────────────────────── Helpers ─────────────────────────────


def get_transition_epoch(data, threshold=0.95):
    """First logged epoch where accuracy exceeds threshold. Returns None if never."""
    acc_curve = data["train_acc"]
    logged_epochs = data["logged_epochs"]
    for i, a in enumerate(acc_curve):
        if a >= threshold:
            return logged_epochs[i]
    return None


def setup_plot_style():
    """Set up matplotlib for publication-quality plots."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 10,
            "figure.figsize": (10, 6),
            "lines.linewidth": 1.5,
        }
    )


def get_colors(n):
    """Get n colorblind-friendly colors."""
    cmap = plt.cm.tab10
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def get_logged_epochs(results):
    """Get the epoch indices that were logged (from any seed)."""
    first_seed = sorted(results.keys())[0]
    return np.array(results[first_seed]["logged_epochs"])


def aggregate_metric(results, metric):
    """Aggregate a metric across seeds: returns (mean, std) arrays."""
    seeds = sorted(results.keys())
    data = np.array([results[s][metric] for s in seeds])
    return data.mean(axis=0), data.std(axis=0)


# ─────────────────────────── Plots ───────────────────────────────


def plot1_learning_curves(main_results):
    """Training accuracy vs epoch for all hidden_dims."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_colors(len(MAIN_HIDDEN_DIMS))

    for idx, hd in enumerate(MAIN_HIDDEN_DIMS):
        epochs = get_logged_epochs(main_results[hd])
        mean, std = aggregate_metric(main_results[hd], "train_acc")
        ax.plot(epochs, mean, color=colors[idx], label=f"D={hd}")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=colors[idx])

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlim(0, NUM_EPOCHS)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Accuracy")
    ax.set_title("Phase transition sharpness depends on hidden dimension")
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plot1_learning_curves.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved plot1_learning_curves.png")


def plot2_loss_curves(main_results):
    """Training loss vs epoch (log-scale) for all hidden_dims."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_colors(len(MAIN_HIDDEN_DIMS))

    for idx, hd in enumerate(MAIN_HIDDEN_DIMS):
        epochs = get_logged_epochs(main_results[hd])
        mean, std = aggregate_metric(main_results[hd], "train_loss")
        ax.plot(epochs, mean, color=colors[idx], label=f"D={hd}")
        ax.fill_between(
            epochs,
            np.maximum(mean - std, 1e-6),
            mean + std,
            alpha=0.15,
            color=colors[idx],
        )

    ax.set_yscale("log")
    ax.set_xlim(0, NUM_EPOCHS)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (BCE)")
    ax.set_title("Training loss plateau and sharp drop")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plot2_loss_curves.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved plot2_loss_curves.png")


def plot3_transition_epoch(main_results):
    """Transition epoch vs hidden_dim."""
    fig, ax = plt.subplots(figsize=(8, 6))

    dims = []
    means = []
    stds = []

    for hd in MAIN_HIDDEN_DIMS:
        seed_epochs = []
        did_not_converge = []
        for seed in range(MAIN_SEEDS):
            te = get_transition_epoch(main_results[hd][seed])
            if te is not None:
                seed_epochs.append(te)
                ax.scatter(hd, te, color="steelblue", alpha=0.5, s=40, zorder=5)
            else:
                did_not_converge.append(seed)
                ax.annotate(
                    "",
                    xy=(hd, NUM_EPOCHS),
                    xytext=(hd, NUM_EPOCHS - 150),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                )

        if seed_epochs:
            m = np.mean(seed_epochs)
            s = np.std(seed_epochs)
            dims.append(hd)
            means.append(m)
            stds.append(s)

    if dims:
        ax.errorbar(dims, means, yerr=stds, fmt="o-", color="darkblue", capsize=5, markersize=8, zorder=10)

    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Transition Epoch (acc > 0.95)")
    ax.set_title("Transition epoch vs model capacity")
    ax.set_ylim(0, NUM_EPOCHS + 200)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plot3_transition_epoch.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved plot3_transition_epoch.png")


def plot4_gate_selectivity(main_results):
    """Gate selectivity over training for hidden_dim=4 (dual y-axis)."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Use seed 0 for a clean single-run view
    data = main_results[4][0]
    epochs = np.array(data["logged_epochs"])

    # Left axis: training accuracy
    ax1.plot(epochs, data["train_acc"], color="green", linewidth=2, label="Train accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Accuracy", color="green")
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.set_ylim(0.45, 1.02)

    # Right axis: gate metrics
    ax2 = ax1.twinx()
    ax2.plot(epochs, data["gate_selectivity"], color="darkorange", linewidth=2, label="Gate selectivity")
    ax2.plot(epochs, data["gate_mean_1"], color="darkorange", linestyle="--", alpha=0.6, label="mean(z|x=1)")
    ax2.plot(epochs, data["gate_mean_0"], color="brown", linestyle="--", alpha=0.6, label="mean(z|x=0)")
    ax2.set_ylabel("Gate Value", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Gate selectivity emerges at the phase transition (D=4)")
    ax1.set_xlim(0, NUM_EPOCHS)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plot4_gate_selectivity.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved plot4_gate_selectivity.png")


def plot5_spectral_radius(main_results):
    """Spectral radius of W_h over training for all hidden_dims."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_colors(len(MAIN_HIDDEN_DIMS))

    for idx, hd in enumerate(MAIN_HIDDEN_DIMS):
        epochs = get_logged_epochs(main_results[hd])
        mean, _ = aggregate_metric(main_results[hd], "spectral_radius")
        ax.plot(epochs, mean, color=colors[idx], label=f"D={hd}")

        # Vertical dashed line at transition epoch (mean across seeds)
        trans_epochs = []
        for seed in range(MAIN_SEEDS):
            te = get_transition_epoch(main_results[hd][seed])
            if te is not None:
                trans_epochs.append(te)
        if trans_epochs:
            te_mean = int(np.mean(trans_epochs))
            ax.axvline(x=te_mean, color=colors[idx], linestyle=":", alpha=0.5)

    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="ρ=1")
    ax.set_xlim(0, NUM_EPOCHS)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spectral Radius of W_h")
    ax.set_title("Spectral radius evolution during training")
    ax.legend(loc="best", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plot5_spectral_radius.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved plot5_spectral_radius.png")


def plot6_generalization_length(main_results):
    """Length generalization: accuracy on length-16 sequences."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_colors(len(MAIN_HIDDEN_DIMS))

    for idx, hd in enumerate(MAIN_HIDDEN_DIMS):
        epochs = get_logged_epochs(main_results[hd])
        mean, std = aggregate_metric(main_results[hd], "test_acc")
        ax.plot(epochs, mean, color=colors[idx], label=f"D={hd}")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.1, color=colors[idx])

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(0, NUM_EPOCHS)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (length-16 sequences)")
    ax.set_title("Length generalization: trained on length-8, tested on length-16")
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plot6_generalization_length.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved plot6_generalization_length.png")


def plot7_transition_vs_seqlen(sec_results):
    """Transition epoch vs seq_len from secondary experiment."""
    fig, ax = plt.subplots(figsize=(8, 6))

    seq_lens = []
    means = []
    stds = []

    for sl in SEC_SEQ_LENS:
        seed_epochs = []
        for seed in range(SEC_SEEDS):
            te = get_transition_epoch(sec_results[sl][seed])
            if te is not None:
                seed_epochs.append(te)
                ax.scatter(sl, te, color="steelblue", alpha=0.5, s=40, zorder=5)
            else:
                ax.annotate(
                    "",
                    xy=(sl, NUM_EPOCHS),
                    xytext=(sl, NUM_EPOCHS - 150),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                )

        if seed_epochs:
            m = np.mean(seed_epochs)
            s = np.std(seed_epochs)
            seq_lens.append(sl)
            means.append(m)
            stds.append(s)

    if seq_lens:
        ax.errorbar(seq_lens, means, yerr=stds, fmt="o-", color="darkblue", capsize=5, markersize=8, zorder=10)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Transition Epoch (acc > 0.95)")
    ax.set_title("Task difficulty vs phase transition onset (D=4)")
    ax.set_ylim(0, NUM_EPOCHS + 200)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plot7_transition_vs_seqlen.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved plot7_transition_vs_seqlen.png")


def plot8_phase_diagram(main_results):
    """Phase diagram: hidden_dim vs epoch, colored by accuracy."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Build 2D array: rows=hidden_dims, cols=logged_epochs
    first_hd = MAIN_HIDDEN_DIMS[0]
    n_logged = len(main_results[first_hd][sorted(main_results[first_hd].keys())[0]]["train_acc"])
    acc_matrix = np.zeros((len(MAIN_HIDDEN_DIMS), n_logged))
    for i, hd in enumerate(MAIN_HIDDEN_DIMS):
        mean, _ = aggregate_metric(main_results[hd], "train_acc")
        acc_matrix[i, :] = mean

    im = ax.imshow(
        acc_matrix,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        extent=[0, NUM_EPOCHS, -0.5, len(MAIN_HIDDEN_DIMS) - 0.5],
        interpolation="nearest",
    )

    ax.set_yticks(range(len(MAIN_HIDDEN_DIMS)))
    ax.set_yticklabels([str(d) for d in MAIN_HIDDEN_DIMS])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hidden Dimension")
    ax.set_title("Phase diagram: accuracy across hidden dimension and training time")

    cbar = fig.colorbar(im, ax=ax, label="Training Accuracy")
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plot8_phase_diagram.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved plot8_phase_diagram.png")


def plot9_critical_window(main_results):
    """Zoomed-in view of the phase transition for hidden_dim=4."""
    # Find transition epoch (use seed 0)
    data = main_results[4][0]
    logged_epochs = np.array(data["logged_epochs"])
    te = get_transition_epoch(data)
    if te is None:
        te = 1500  # fallback center

    # Select logged points within ±100 epochs of the transition
    window = 100
    mask = (logged_epochs >= te - window) & (logged_epochs <= te + window)
    epoch_range = logged_epochs[mask]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # 1. Training accuracy
    axes[0].plot(epoch_range, np.array(data["train_acc"])[mask], color="steelblue", linewidth=2)
    axes[0].axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Train Accuracy")
    axes[0].set_title(f"Anatomy of the phase transition (D=4, seed=0, transition≈epoch {te})")

    # 2. Loss
    axes[1].plot(epoch_range, np.array(data["train_loss"])[mask], color="crimson", linewidth=2)
    axes[1].set_ylabel("Training Loss")

    # 3. Gate selectivity
    axes[2].plot(epoch_range, np.array(data["gate_selectivity"])[mask], color="darkorange", linewidth=2)
    axes[2].set_ylabel("Gate Selectivity")

    # 4. Spectral radius
    axes[3].plot(epoch_range, np.array(data["spectral_radius"])[mask], color="purple", linewidth=2)
    axes[3].axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    axes[3].set_ylabel("Spectral Radius")
    axes[3].set_xlabel("Epoch")

    # Mark transition epoch
    for ax in axes:
        ax.axvline(x=te, color="black", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plot9_critical_window.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved plot9_critical_window.png")


def plot10_C_weights(main_results):
    """Evolution of output weights C over training for hidden_dim=4."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seed 0
    data = main_results[4][0]
    epochs = np.array(data["logged_epochs"])
    c_vals = np.array(data["c_values"])  # (n_logged, D)

    colors_c = get_colors(c_vals.shape[1])
    for d in range(c_vals.shape[1]):
        ax.plot(epochs, c_vals[:, d], color=colors_c[d], linewidth=2, label=f"C[{d}]")

    # Mark transition
    te = get_transition_epoch(data)
    if te is not None:
        ax.axvline(x=te, color="black", linestyle="--", alpha=0.5, label=f"Transition (epoch {te})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Value")
    ax.set_title("Evolution of output weights C during training (D=4, seed=0)")
    ax.legend()
    ax.set_xlim(0, NUM_EPOCHS)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plot10_C_weights.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved plot10_C_weights.png")


# ─────────────────────────── Summary ─────────────────────────────


def print_summary_table(main_results):
    """Print a summary table of main experiment results."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = (
        f"{'hidden_dim':>10} | {'n_params':>8} | {'transition_epoch':>20} | "
        f"{'final_train_acc':>15} | {'final_test_acc16':>16} | {'peak_gate_sel':>14}"
    )
    print(header)
    print("-" * len(header))

    for hd in MAIN_HIDDEN_DIMS:
        n_params = 2 * (hd**2 + hd + hd) + hd + 1

        trans_epochs = []
        final_train = []
        final_test = []
        peak_sel = []

        for seed in range(MAIN_SEEDS):
            data = main_results[hd][seed]
            te = get_transition_epoch(data)
            if te is not None:
                trans_epochs.append(te)
            final_train.append(data["train_acc"][-1])
            final_test.append(data["test_acc"][-1])
            peak_sel.append(max(data["gate_selectivity"]))

        if trans_epochs:
            te_str = f"{np.mean(trans_epochs):.0f} ± {np.std(trans_epochs):.0f}"
        else:
            te_str = "never"

        print(
            f"{hd:>10} | {n_params:>8} | {te_str:>20} | "
            f"{np.mean(final_train):>15.4f} | {np.mean(final_test):>16.4f} | "
            f"{np.mean(peak_sel):>14.4f}"
        )


# ─────────────────────────── Main ────────────────────────────────


if __name__ == "__main__":
    setup_plot_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("Grokking in Gated State Space Models — Parity Task")
    print("=" * 60)
    t_start = time.time()

    # ── Main experiment ──
    print("\n>>> MAIN EXPERIMENT: Varying Hidden Dimension <<<")
    main_results = run_main_experiment()

    # ── Secondary experiment ──
    print("\n>>> SECONDARY EXPERIMENT: Varying Sequence Length <<<")
    sec_results = run_secondary_experiment()

    # ── Save results ──
    all_results = {"main": main_results, "secondary": sec_results}
    with open("ssm_grokking_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("\nResults saved to ssm_grokking_results.pkl")

    # ── Generate plots ──
    print("\nGenerating plots...")
    plot1_learning_curves(main_results)
    plot2_loss_curves(main_results)
    plot3_transition_epoch(main_results)
    plot4_gate_selectivity(main_results)
    plot5_spectral_radius(main_results)
    plot6_generalization_length(main_results)
    plot7_transition_vs_seqlen(sec_results)
    plot8_phase_diagram(main_results)
    plot9_critical_window(main_results)
    plot10_C_weights(main_results)

    # ── Summary ──
    print_summary_table(main_results)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    print("Done.")
