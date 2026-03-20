import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────── Helpers ───────────────────────────


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


# ─────────────────────────── Plots ───────────────────────────


def plot1_learning_curves(main_results, cfg):
    """Training accuracy vs epoch for all hidden_dims."""
    fig, ax = plt.subplots(figsize=(10, 6))
    hidden_dims = cfg["main_hidden_dims"]
    colors = get_colors(len(hidden_dims))

    for idx, hd in enumerate(hidden_dims):
        epochs = get_logged_epochs(main_results[hd])
        mean, std = aggregate_metric(main_results[hd], "train_acc")
        ax.plot(epochs, mean, color=colors[idx], label=f"D={hd}")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=colors[idx])

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlim(0, cfg["num_epochs"])
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Accuracy")
    ax.set_title("Phase transition sharpness depends on hidden dimension")
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["fig_dir"], "plot1_learning_curves.png"), dpi=cfg["dpi"])
    plt.close(fig)
    print("  Saved plot1_learning_curves.png")


def plot2_loss_curves(main_results, cfg):
    """Training loss vs epoch (log-scale) for all hidden_dims."""
    fig, ax = plt.subplots(figsize=(10, 6))
    hidden_dims = cfg["main_hidden_dims"]
    colors = get_colors(len(hidden_dims))

    for idx, hd in enumerate(hidden_dims):
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
    ax.set_xlim(0, cfg["num_epochs"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (BCE)")
    ax.set_title("Training loss plateau and sharp drop")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["fig_dir"], "plot2_loss_curves.png"), dpi=cfg["dpi"])
    plt.close(fig)
    print("  Saved plot2_loss_curves.png")


def plot3_transition_epoch(main_results, cfg):
    """Transition epoch vs hidden_dim."""
    fig, ax = plt.subplots(figsize=(8, 6))
    hidden_dims = cfg["main_hidden_dims"]
    num_epochs = cfg["num_epochs"]

    dims = []
    means = []
    stds = []

    for hd in hidden_dims:
        seed_epochs = []
        for seed in range(cfg["main_seeds"]):
            te = get_transition_epoch(main_results[hd][seed])
            if te is not None:
                seed_epochs.append(te)
                ax.scatter(hd, te, color="steelblue", alpha=0.5, s=40, zorder=5)
            else:
                ax.annotate(
                    "",
                    xy=(hd, num_epochs),
                    xytext=(hd, num_epochs - 150),
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
    ax.set_ylim(0, num_epochs + 200)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["fig_dir"], "plot3_transition_epoch.png"), dpi=cfg["dpi"])
    plt.close(fig)
    print("  Saved plot3_transition_epoch.png")


def plot4_gate_selectivity(main_results, cfg):
    """Gate selectivity over training for hidden_dim=4 (dual y-axis)."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

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
    ax1.set_xlim(0, cfg["num_epochs"])
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["fig_dir"], "plot4_gate_selectivity.png"), dpi=cfg["dpi"])
    plt.close(fig)
    print("  Saved plot4_gate_selectivity.png")


def plot5_spectral_radius(main_results, cfg):
    """Spectral radius of W_h over training for all hidden_dims."""
    fig, ax = plt.subplots(figsize=(10, 6))
    hidden_dims = cfg["main_hidden_dims"]
    colors = get_colors(len(hidden_dims))

    for idx, hd in enumerate(hidden_dims):
        epochs = get_logged_epochs(main_results[hd])
        mean, _ = aggregate_metric(main_results[hd], "spectral_radius")
        ax.plot(epochs, mean, color=colors[idx], label=f"D={hd}")

        # Vertical dashed line at transition epoch (mean across seeds)
        trans_epochs = []
        for seed in range(cfg["main_seeds"]):
            te = get_transition_epoch(main_results[hd][seed])
            if te is not None:
                trans_epochs.append(te)
        if trans_epochs:
            te_mean = int(np.mean(trans_epochs))
            ax.axvline(x=te_mean, color=colors[idx], linestyle=":", alpha=0.5)

    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="ρ=1")
    ax.set_xlim(0, cfg["num_epochs"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spectral Radius of W_h")
    ax.set_title("Spectral radius evolution during training")
    ax.legend(loc="best", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["fig_dir"], "plot5_spectral_radius.png"), dpi=cfg["dpi"])
    plt.close(fig)
    print("  Saved plot5_spectral_radius.png")


def plot6_generalization_length(main_results, cfg):
    """Length generalization: accuracy on length-16 sequences."""
    fig, ax = plt.subplots(figsize=(10, 6))
    hidden_dims = cfg["main_hidden_dims"]
    colors = get_colors(len(hidden_dims))

    for idx, hd in enumerate(hidden_dims):
        epochs = get_logged_epochs(main_results[hd])
        mean, std = aggregate_metric(main_results[hd], "test_acc")
        ax.plot(epochs, mean, color=colors[idx], label=f"D={hd}")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.1, color=colors[idx])

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(0, cfg["num_epochs"])
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (length-16 sequences)")
    ax.set_title("Length generalization: trained on length-8, tested on length-16")
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["fig_dir"], "plot6_generalization_length.png"), dpi=cfg["dpi"])
    plt.close(fig)
    print("  Saved plot6_generalization_length.png")


def plot7_transition_vs_seqlen(sec_results, cfg):
    """Transition epoch vs seq_len from secondary experiment."""
    fig, ax = plt.subplots(figsize=(8, 6))
    num_epochs = cfg["num_epochs"]

    seq_lens = []
    means = []
    stds = []

    for sl in cfg["sec_seq_lens"]:
        seed_epochs = []
        for seed in range(cfg["sec_seeds"]):
            te = get_transition_epoch(sec_results[sl][seed])
            if te is not None:
                seed_epochs.append(te)
                ax.scatter(sl, te, color="steelblue", alpha=0.5, s=40, zorder=5)
            else:
                ax.annotate(
                    "",
                    xy=(sl, num_epochs),
                    xytext=(sl, num_epochs - 150),
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
    ax.set_ylim(0, num_epochs + 200)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["fig_dir"], "plot7_transition_vs_seqlen.png"), dpi=cfg["dpi"])
    plt.close(fig)
    print("  Saved plot7_transition_vs_seqlen.png")


def plot8_phase_diagram(main_results, cfg):
    """Phase diagram: hidden_dim vs epoch, colored by accuracy."""
    fig, ax = plt.subplots(figsize=(12, 5))
    hidden_dims = cfg["main_hidden_dims"]
    num_epochs = cfg["num_epochs"]

    # Build 2D array: rows=hidden_dims, cols=logged_epochs
    first_hd = hidden_dims[0]
    n_logged = len(main_results[first_hd][sorted(main_results[first_hd].keys())[0]]["train_acc"])
    acc_matrix = np.zeros((len(hidden_dims), n_logged))
    for i, hd in enumerate(hidden_dims):
        mean, _ = aggregate_metric(main_results[hd], "train_acc")
        acc_matrix[i, :] = mean

    im = ax.imshow(
        acc_matrix,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        extent=[0, num_epochs, -0.5, len(hidden_dims) - 0.5],
        interpolation="nearest",
    )

    ax.set_yticks(range(len(hidden_dims)))
    ax.set_yticklabels([str(d) for d in hidden_dims])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hidden Dimension")
    ax.set_title("Phase diagram: accuracy across hidden dimension and training time")

    cbar = fig.colorbar(im, ax=ax, label="Training Accuracy")
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    fig.tight_layout()
    fig.savefig(os.path.join(cfg["fig_dir"], "plot8_phase_diagram.png"), dpi=cfg["dpi"])
    plt.close(fig)
    print("  Saved plot8_phase_diagram.png")


def plot9_critical_window(main_results, cfg):
    """Zoomed-in view of the phase transition for hidden_dim=4."""
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
    fig.savefig(os.path.join(cfg["fig_dir"], "plot9_critical_window.png"), dpi=cfg["dpi"])
    plt.close(fig)
    print("  Saved plot9_critical_window.png")


def plot10_C_weights(main_results, cfg):
    """Evolution of output weights C over training for hidden_dim=4."""
    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.set_xlim(0, cfg["num_epochs"])
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["fig_dir"], "plot10_C_weights.png"), dpi=cfg["dpi"])
    plt.close(fig)
    print("  Saved plot10_C_weights.png")
