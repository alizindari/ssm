import itertools

import torch


def generate_parity_dataset(seq_len, max_samples=None):
    """Generate binary sequences and their running parity labels.

    If 2^seq_len <= max_samples (or max_samples is None), returns ALL sequences.
    Otherwise, samples max_samples random sequences.
    """
    n_total = 2**seq_len
    if max_samples is None or n_total <= max_samples:
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
