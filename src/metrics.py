import torch
import torch.nn.functional as F


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
        mask_1 = train_x.unsqueeze(-1).expand_as(train_gates)
        mask_0 = (1 - train_x).unsqueeze(-1).expand_as(train_gates)

        n_ones = mask_1.sum()
        n_zeros = mask_0.sum()

        gate_mean_1 = (train_gates * mask_1).sum() / n_ones if n_ones > 0 else 0.0
        gate_mean_0 = (train_gates * mask_0).sum() / n_zeros if n_zeros > 0 else 0.0

        gate_mean_1 = gate_mean_1.item() if isinstance(gate_mean_1, torch.Tensor) else gate_mean_1
        gate_mean_0 = gate_mean_0.item() if isinstance(gate_mean_0, torch.Tensor) else gate_mean_0
        gate_selectivity = gate_mean_1 - gate_mean_0

        # Spectral analysis
        if hasattr(model, "W_h"):
            # GRU: eigenvalues of recurrent weight matrix
            eigvals = torch.linalg.eigvals(model.W_h.data)
            spectral_radius = eigvals.abs().max().item()
            eigenvalues = eigvals.tolist()
        else:
            # Mamba: diagonal A, eigenvalues = exp(-exp(A_log))
            A_vals = torch.exp(-torch.exp(model.A_log.data))
            spectral_radius = A_vals.abs().max().item()
            eigenvalues = A_vals.tolist()

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
