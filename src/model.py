import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SelectiveSSM(nn.Module):
    """Minimal Mamba-style selective SSM for scalar binary input.

    Uses diagonal A with input-dependent discretization step delta.
    Selectivity comes from delta being a function of the input:
    large delta lets new input in, small delta preserves state.

    Parameters: 6D + 1 (vs GRU's 2D^2 + 5D + 1).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        D = hidden_dim
        # Diagonal state matrix in log-space: A = -exp(A_log) < 0
        self.A_log = nn.Parameter(torch.zeros(D))
        # Input-dependent discretization: delta = softplus(delta_w * x + delta_b)
        self.delta_w = nn.Parameter(torch.empty(D))
        self.delta_b = nn.Parameter(torch.zeros(D))
        # Input-dependent B: B_t = B_w * x + B_b
        self.B_w = nn.Parameter(torch.empty(D))
        self.B_b = nn.Parameter(torch.zeros(D))
        # Output readout
        self.C = nn.Parameter(torch.empty(D))
        self.d = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.A_log, -2, 0)
        nn.init.normal_(self.delta_w, std=0.5)
        nn.init.normal_(self.B_w, std=0.5)
        nn.init.normal_(self.C, std=0.5)

    def forward(self, x, return_gates=False):
        """
        Args:
            x: (batch, seq_len) float tensor of 0s and 1s
            return_gates: if True, also return delta activations
        Returns:
            outputs: (batch, seq_len) predicted probabilities
            gates (optional): (batch, seq_len, D) delta activations
        """
        batch, T = x.shape
        D = self.A_log.shape[0]
        A = -torch.exp(self.A_log)  # (D,), negative for stability
        h = torch.zeros(batch, D, device=x.device)

        outputs = []
        gates = [] if return_gates else None

        for t in range(T):
            x_t = x[:, t : t + 1]  # (batch, 1)

            # Input-dependent discretization step
            delta = F.softplus(x_t * self.delta_w + self.delta_b)  # (batch, D)

            # Input-dependent B
            B = x_t * self.B_w + self.B_b  # (batch, D)

            # Discretize
            A_bar = torch.exp(delta * A)  # (batch, D)
            B_bar = delta * B  # (batch, D)

            # State update (element-wise, A is diagonal)
            h = A_bar * h + B_bar * x_t  # (batch, D)

            # Output
            y_t = torch.sigmoid(h @ self.C + self.d)  # (batch,)
            outputs.append(y_t)
            if return_gates:
                gates.append(delta.detach())

        outputs = torch.stack(outputs, dim=1)  # (batch, T)
        outputs = outputs.clamp(1e-7, 1 - 1e-7)

        if return_gates:
            gates = torch.stack(gates, dim=1)  # (batch, T, D)
            return outputs, gates
        return outputs
