"""
Mixture of Experts (MoE) layer with a learned gating mechanism.
Each expert is a 2-layer MLP. The gating network computes soft weights
over experts and returns a weighted sum of expert outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single MLP expert."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.

    Args:
        input_dim:   Dimensionality of input features.
        expert_hidden_dim: Hidden size inside each expert MLP.
        output_dim:  Dimensionality of output features.
        num_experts: Number of parallel expert networks.
        top_k:       How many experts to activate per token (soft-weighted sum).
        dropout:     Dropout applied inside each expert.
    """

    def __init__(
        self,
        input_dim: int,
        expert_hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert top_k <= num_experts, "top_k must be <= num_experts"
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_hidden_dim, output_dim, dropout) for _ in range(num_experts)]
        )
        # Gating network: maps input to a distribution over experts
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            out: (batch, seq_len, output_dim)
        """
        B, S, D = x.shape
        x_flat = x.view(B * S, D)  # (B*S, input_dim)

        # Gating
        gate_logits = self.gate(x_flat)                          # (B*S, num_experts)
        gate_scores = F.softmax(gate_logits, dim=-1)             # (B*S, num_experts)

        # Top-k selection
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1)  # (B*S, top_k)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)  # renormalise

        # Compute all expert outputs, then select
        expert_outputs = torch.stack(
            [expert(x_flat) for expert in self.experts], dim=1
        )  # (B*S, num_experts, output_dim)

        # Weighted sum over top-k experts
        top_k_scores_exp = top_k_scores.unsqueeze(-1)            # (B*S, top_k, 1)
        selected = expert_outputs.gather(
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)),
        )                                                         # (B*S, top_k, output_dim)
        out_flat = (top_k_scores_exp * selected).sum(dim=1)      # (B*S, output_dim)

        return out_flat.view(B, S, -1)
