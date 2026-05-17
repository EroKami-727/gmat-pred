"""
OrbitGuard Model Architectures — LSTM & Time-Series Transformers
================================================================
Neural network architectures designed to process physics-invariant
trajectory telemetry for simulation pruning.

Contains:
1. **TrajectoryLSTM**: Recurrent model for sequence classification.
2. **TrajectoryTransformer**: Self-attention model with CLS token pooling.
   Preferred over LSTM for long trajectories (global context, no vanishing gradient).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# 1. LSTM Architecture
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryLSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) model for trajectory classification/regression.

    Includes an embedding layer for features, stacked LSTM layers,
    and task-specific output heads.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        task: str = "binary"
    ):
        super().__init__()
        self.task = task

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        lengths: (batch,) actual lengths before padding
        """
        embedded = self.embedding(x)
        packed_x = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed_x)

        if self.lstm.bidirectional:
            last_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            last_hidden = hn[-1, :, :]

        out = self.fc(last_hidden)

        if self.task in ["binary", "regression"]:
            return out.squeeze(1)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. Transformer Architecture
# ═══════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class TrajectoryTransformer(nn.Module):
    """
    Transformer encoder for trajectory pruning.

    Uses a learnable [CLS] token prepended to the sequence. After the encoder
    stack, the CLS output is fed to the classification head — more stable than
    mean pooling for variable-length sequences.

    Defaults are tuned for the OrbitGuard 13-feature, ~576-step sequences:
      d_model=128, nhead=8, num_layers=4, dim_feedforward=512

    Architecture Ablation Flags
    ---------------------------
    use_cls_token    : if False, switches to mean pooling over non-padding positions.
    use_pos_encoding : if False, skips the sinusoidal positional encoding step.
    Both flags are used by arch_ablation.py; the production model keeps both True.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        output_dim: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        task: str = "binary",
        use_cls_token: bool = True,
        use_pos_encoding: bool = True,
    ):
        super().__init__()
        self.task = task
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.use_pos_encoding = use_pos_encoding

        # Project raw features into model dimension
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len + 1)  # +1 for CLS

        # Learnable [CLS] token (only allocated when used)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x    : (batch, seq_len, input_dim)
        mask : (batch, seq_len) bool — True where padding exists (from dataset)
        """
        B = x.size(0)

        # 1. Embed features
        x = self.embedding(x) * math.sqrt(self.d_model)

        # 2. Optionally prepend CLS token, then apply positional encoding
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)
            x = torch.cat([cls, x], dim=1)          # (B, 1+seq_len, d_model)
            if self.use_pos_encoding:
                x = self.pos_encoder(x)
            if mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)  # (B, 1+seq_len)
        else:
            if self.use_pos_encoding:
                x = self.pos_encoder(x)

        # 3. Transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=mask)

        # 4. Pool to a single vector
        if self.use_cls_token:
            pooled = output[:, 0, :]   # CLS token output
        else:
            # Mean pooling over actual (non-padding) positions
            if mask is not None:
                valid = (~mask).unsqueeze(-1).float()
                pooled = (output * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            else:
                pooled = output.mean(dim=1)

        out = self.fc(pooled)

        if self.task in ["binary", "regression"]:
            return out.squeeze(1)
        return out
