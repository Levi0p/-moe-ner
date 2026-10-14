"""
NER tagger: [GloVe word emb + casing emb + char-CNN] -> BiLSTM -> (optional block) -> Linear.

head = "none" : plain BiLSTM tagger (baseline)
head = "ffn"  : dense residual FFN block (capacity-matched control for the MoE)
head = "moe"  : residual MoE block (4 experts, top-2 routing)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from moe_layer import MoELayer, Expert
from dataset import NUM_CASE


class CharCNN(nn.Module):
    def __init__(self, n_chars, char_dim=30, filters=50, kernel=3, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_dim, filters, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, chars):                     # (B, S, W)
        B, S, W = chars.shape
        x = self.drop(self.emb(chars.view(B * S, W))).transpose(1, 2)   # (BS, C, W)
        x = torch.relu(self.conv(x)).max(dim=2).values                  # (BS, F)
        return x.view(B, S, -1)


class NERTagger(nn.Module):
    def __init__(
        self,
        emb_matrix: torch.Tensor,
        n_chars: int,
        num_labels: int,
        head: str = "none",
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int = 256,
    ):
        super().__init__()
        assert head in ("none", "ffn", "moe")
        self.head = head
        self.word_emb = nn.Embedding.from_pretrained(emb_matrix, freeze=False, padding_idx=0)
        self.case_emb = nn.Embedding(NUM_CASE, 8, padding_idx=0)
        self.char_cnn = CharCNN(n_chars)
        in_dim = emb_matrix.size(1) + 8 + 50
        self.lstm = nn.LSTM(in_dim, hidden_dim // 2, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        if head == "moe":
            self.block = MoELayer(hidden_dim, expert_hidden_dim, hidden_dim, num_experts, top_k, dropout)
        elif head == "ffn":
            self.block = Expert(hidden_dim, expert_hidden_dim, hidden_dim, dropout)
        if head != "none":
            self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, words, cases, chars, mask):
        x = torch.cat([self.word_emb(words), self.case_emb(cases), self.char_cnn(chars)], dim=-1)
        x = self.dropout(x)
        lengths = mask.sum(1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=words.size(1))
        out = self.dropout(out)
        if self.head == "moe":
            out = self.layer_norm(out + self.block(out, mask))
        elif self.head == "ffn":
            out = self.layer_norm(out + self.block(out))
        return self.classifier(out)
