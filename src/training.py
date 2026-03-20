from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from data import generate_parity_dataset
from metrics import compute_metrics
from model import GatedSSM, SelectiveSSM


def train_single_run(hidden_dim, seq_len, seed, cfg, test_x, test_y):
    """Train one model configuration and return full metric history."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    log_interval = cfg["log_interval"]

    # Data
    max_samples = cfg["large_n_sample"] if seq_len > cfg["large_n_threshold"] else None
    train_x, train_y = generate_parity_dataset(seq_len, max_samples)

    # Model + optimizer
    if cfg["model"] == "mamba":
        model = SelectiveSSM(hidden_dim)
    else:
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

        # Log metrics every log_interval epochs
        if epoch % log_interval == 0 or epoch == num_epochs - 1:
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
