#!/usr/bin/env python3
"""
Phase Transitions in Gated State Space Models on the Binary Parity Task.

Studies sharp phase transitions in a GRU-style gated SSM trained on the
running parity task. Varies hidden dimension and sequence length, logging
detailed metrics and producing publication-quality plots.
"""

import argparse
import os
import pickle
import time

from experiments import run_main_experiment, run_secondary_experiment
from plotting import (
    setup_plot_style,
    plot1_learning_curves,
    plot2_loss_curves,
    plot3_transition_epoch,
    plot4_gate_selectivity,
    plot5_spectral_radius,
    plot6_generalization_length,
    plot7_transition_vs_seqlen,
    plot8_phase_diagram,
    plot9_critical_window,
    plot10_C_weights,
)
from summary import print_summary_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase transitions in gated SSMs on the parity task"
    )

    # Model
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "mamba"])

    # Main experiment
    parser.add_argument("--main-hidden-dims", type=str, default="1,2,3,4,6,8,12,16")
    parser.add_argument("--main-seq-len", type=int, default=8)
    parser.add_argument("--main-seeds", type=int, default=1)

    # Secondary experiment
    parser.add_argument("--sec-seq-lens", type=str, default="4,6,8,10,12,16")
    parser.add_argument("--sec-hidden-dim", type=int, default=4)
    parser.add_argument("--sec-seeds", type=int, default=1)

    # Training
    parser.add_argument("--num-epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    # Testing
    parser.add_argument("--test-len", type=int, default=16)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--large-n-threshold", type=int, default=12)
    parser.add_argument("--large-n-sample", type=int, default=4096)

    # Logging & output
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fig-dir", type=str, default="figures")
    parser.add_argument("--results-file", type=str, default="ssm_grokking_results.pkl")

    return parser.parse_args()


def build_config(args):
    """Convert argparse namespace to config dict."""
    return {
        "model": args.model,
        "main_hidden_dims": [int(x) for x in args.main_hidden_dims.split(",")],
        "main_seq_len": args.main_seq_len,
        "main_seeds": args.main_seeds,
        "sec_seq_lens": [int(x) for x in args.sec_seq_lens.split(",")],
        "sec_hidden_dim": args.sec_hidden_dim,
        "sec_seeds": args.sec_seeds,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "test_len": args.test_len,
        "test_size": args.test_size,
        "large_n_threshold": args.large_n_threshold,
        "large_n_sample": args.large_n_sample,
        "log_interval": args.log_interval,
        "dpi": args.dpi,
        "device": args.device,
        "fig_dir": args.fig_dir,
        "results_file": args.results_file,
    }


if __name__ == "__main__":
    cfg = build_config(parse_args())

    setup_plot_style()
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 60)
    print(f"Grokking in Gated State Space Models — Parity Task ({cfg['model'].upper()})")
    print("=" * 60)
    t_start = time.time()

    # ── Main experiment ──
    print("\n>>> MAIN EXPERIMENT: Varying Hidden Dimension <<<")
    main_results = run_main_experiment(cfg)

    # ── Secondary experiment ──
    print("\n>>> SECONDARY EXPERIMENT: Varying Sequence Length <<<")
    sec_results = run_secondary_experiment(cfg)

    # ── Save results ──
    all_results = {"main": main_results, "secondary": sec_results}
    with open(cfg["results_file"], "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {cfg['results_file']}")

    # ── Generate plots ──
    print("\nGenerating plots...")
    plot1_learning_curves(main_results, cfg)
    plot2_loss_curves(main_results, cfg)
    plot3_transition_epoch(main_results, cfg)
    plot4_gate_selectivity(main_results, cfg)
    plot5_spectral_radius(main_results, cfg)
    plot6_generalization_length(main_results, cfg)
    plot7_transition_vs_seqlen(sec_results, cfg)
    plot8_phase_diagram(main_results, cfg)
    plot9_critical_window(main_results, cfg)
    plot10_C_weights(main_results, cfg)

    # ── Summary ──
    print_summary_table(main_results, cfg)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    print("Done.")
