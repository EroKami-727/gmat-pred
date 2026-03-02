"""
OrbitGuard Model Architectures — LSTM & Time-Series Transformers
================================================================
Neural network architectures designed to process physics-invariant
trajectory telemetry for simulation pruning.

Contains:
1. **TrajectoryLSTM**: Standard recurrent model for sequence classification.
2. **TrajectoryTransformer**: Self-attention based model for global feature extraction.
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
        output_dim: int = 1,  # 1 for binary classification/regression
        dropout: float = 0.2,
        bidirectional: bool = False,
        task: str = "binary"  # "binary", "multiclass", "regression"
    ):
        super().__init__()
        self.task = task
        
        # Initial projection/embedding of raw physics features
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
        
        # Linear head
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
        # 1. Feature embedding
        embedded = self.embedding(x)
        
        # 2. Pack sequence for efficient computation (ignores padding)
        packed_x = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # 3. Recurrent processing
        _, (hn, _) = self.lstm(packed_x)
        
        # 4. Extract last hidden state
        # If bidirectional, concatenate the last states of both directions
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            last_hidden = hn[-1, :, :]
            
        # 5. Output head
        out = self.fc(last_hidden)
        
        if self.task == "binary":
            # Return raw logits (BCEWithLogitsLoss expects this)
            return out.squeeze(1)
        elif self.task == "regression":
            return out.squeeze(1)
        else:
            # Multiclass
            return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. Transformer Architecture (Advanced Research Option)
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


import math # needs to be at top but adding here for now to avoid break

class TrajectoryTransformer(nn.Module):
    """
    Self-attention based Transformer model for trajectory pruning.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        output_dim: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        task: str = "binary"
    ):
        super().__init__()
        self.task = task
        self.d_model = d_model
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        mask: Optional padding mask
        """
        # 1. Embed and encode position
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # 2. Self-attention processing
        # mask is key_padding_mask
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 3. Global pooling (mean)
        # Instead of just taking the last state, we average across all valid time steps
        if mask is not None:
            # Masked mean
            # (Invert mask because True = skip in PyTorch Transformer)
            valid_mask = (~mask).float().unsqueeze(-1)
            output = (output * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        else:
            output = output.mean(dim=1)
            
        # 4. Output head
        out = self.fc(output)
        
        if self.task in ["binary", "regression"]:
            return out.squeeze(1)
        return out
