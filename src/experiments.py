import time

from data import generate_test_set
from training import train_single_run


def _param_count(hidden_dim, model_type):
    """Compute parameter count for a given model type and hidden dimension."""
    if model_type == "mamba":
        return 6 * hidden_dim + 1
    return 2 * (hidden_dim**2 + hidden_dim + hidden_dim) + hidden_dim + 1


def run_main_experiment(cfg):
    """Main experiment: vary hidden_dim with fixed seq_len."""
    results = {}
    test_x, test_y = generate_test_set(cfg["test_len"], cfg["test_size"])

    for hidden_dim in cfg["main_hidden_dims"]:
        results[hidden_dim] = {}
        n_params = _param_count(hidden_dim, cfg["model"])
        print(f"\n=== hidden_dim={hidden_dim} ({n_params} params) ===")

        for seed in range(cfg["main_seeds"]):
            print(f"\n--- seed={seed} ---")
            t0 = time.time()
            history = train_single_run(
                hidden_dim, cfg["main_seq_len"], seed, cfg, test_x, test_y
            )
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s")
            results[hidden_dim][seed] = history

    return results


def run_secondary_experiment(cfg):
    """Secondary experiment: vary seq_len with fixed hidden_dim."""
    results = {}
    test_x, test_y = generate_test_set(cfg["test_len"], cfg["test_size"])

    for seq_len in cfg["sec_seq_lens"]:
        results[seq_len] = {}
        n_seqs = (
            min(2**seq_len, cfg["large_n_sample"])
            if seq_len > cfg["large_n_threshold"]
            else 2**seq_len
        )
        print(f"\n=== seq_len={seq_len} ({n_seqs} sequences) ===")

        for seed in range(cfg["sec_seeds"]):
            print(f"\n--- seed={seed} ---")
            t0 = time.time()
            history = train_single_run(
                cfg["sec_hidden_dim"], seq_len, seed, cfg, test_x, test_y
            )
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s")
            results[seq_len][seed] = history

    return results
