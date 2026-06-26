"""
NER model: Embedding → BiLSTM → MoE layer → Linear classifier
Baseline is the same architecture without the MoE layer.
"""

import torch
import torch.nn as nn
from moe_layer import MoELayer


class BiLSTMNER(nn.Module):
    """Baseline BiLSTM tagger (no MoE)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        embeds = self.dropout(self.embedding(input_ids))          # (B, S, E)
        lstm_out, _ = self.lstm(embeds)                           # (B, S, H)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)                        # (B, S, num_labels)
        return logits


class BiLSTMMoENER(nn.Module):
    """BiLSTM + MoE tagger."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.moe = MoELayer(
            input_dim=hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            output_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        embeds = self.dropout(self.embedding(input_ids))          # (B, S, E)
        lstm_out, _ = self.lstm(embeds)                           # (B, S, H)
        lstm_out = self.dropout(lstm_out)
        moe_out = self.moe(lstm_out)                              # (B, S, H)
        out = self.layer_norm(lstm_out + moe_out)                 # residual
        logits = self.classifier(out)                             # (B, S, num_labels)
        return logits
