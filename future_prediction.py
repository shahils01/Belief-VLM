from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FuturePredictorConfig:
    embed_dim: int
    hidden_dim: int = 1024
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    max_frames: int = 32


class FuturePredictionTransformer(nn.Module):
    def __init__(self, cfg: FuturePredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.embed_dim, cfg.hidden_dim)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_frames, cfg.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=4 * cfg.hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, context_embeddings):
        num_frames = context_embeddings.shape[1]
        if num_frames > self.cfg.max_frames:
            raise RuntimeError(
                f"context length {num_frames} exceeds max_frames={self.cfg.max_frames}. Increase max_frames."
            )
        x = self.input_proj(context_embeddings)
        x = x + self.pos_embed[:, :num_frames]
        x = self.encoder(x)
        x = self.norm(x)
        pred = self.output_proj(x)
        return pred


class FuturePredictionLoss(nn.Module):
    def __init__(self, mse_weight: float = 1.0, cosine_weight: float = 1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        loss = self.mse_weight * mse + self.cosine_weight * cosine
        return {
            "loss": loss,
            "mse": mse,
            "cosine": cosine,
        }
