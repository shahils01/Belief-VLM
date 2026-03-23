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
    max_context_frames: int = 32
    max_future_frames: int = 32


class FuturePredictionTransformer(nn.Module):
    def __init__(self, cfg: FuturePredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.embed_dim, cfg.hidden_dim)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.embed_dim)
        self.context_pos_embed = nn.Parameter(torch.zeros(1, cfg.max_context_frames, cfg.hidden_dim))
        self.future_queries = nn.Parameter(torch.zeros(1, cfg.max_future_frames, cfg.hidden_dim))
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
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=4 * cfg.hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, context_embeddings, future_frames: int, return_layer_outputs: bool = False, num_output_layers: int | None = None):
        num_context_frames = context_embeddings.shape[1]
        if num_context_frames > self.cfg.max_context_frames:
            raise RuntimeError(
                "context length "
                f"{num_context_frames} exceeds max_context_frames={self.cfg.max_context_frames}. "
                "Increase max_context_frames."
            )
        if future_frames > self.cfg.max_future_frames:
            raise RuntimeError(
                f"future length {future_frames} exceeds max_future_frames={self.cfg.max_future_frames}. "
                "Increase max_future_frames."
            )
        memory = self.input_proj(context_embeddings)
        memory = memory + self.context_pos_embed[:, :num_context_frames]
        memory = self.encoder(memory)
        future_queries = self.future_queries[:, :future_frames].expand(memory.size(0), -1, -1)
        decoded = future_queries
        layer_outputs = []
        for layer in self.decoder.layers:
            decoded = layer(decoded, memory)
            layer_outputs.append(self.output_proj(self.norm(decoded)))
        if not layer_outputs:
            raise RuntimeError("Transformer decoder produced no layer outputs.")
        if return_layer_outputs:
            requested_layers = len(layer_outputs) if num_output_layers is None else int(num_output_layers)
            if requested_layers <= 0:
                raise RuntimeError(f"num_output_layers must be positive, got {requested_layers}.")
            if requested_layers > len(layer_outputs):
                raise RuntimeError(
                    f"Requested {requested_layers} predictor layers, but model only has {len(layer_outputs)} layers."
                )
            return torch.stack(layer_outputs[-requested_layers:], dim=2)
        return layer_outputs[-1]


class FuturePredictionLoss(nn.Module):
    def __init__(self, mse_weight: float = 1.0, cosine_weight: float = 1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

    def forward(self, pred, target):
        if target.dim() == pred.dim() and pred.dim() == 4:
            mse = (pred.float() - target.float()).pow(2).mean(dim=(0, 1, 3)).sum()
            cosine = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean(dim=(0, 1)).sum()
        elif target.dim() == pred.dim():
            mse = F.mse_loss(pred, target)
            cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        elif target.dim() == pred.dim() + 1:
            expanded_pred = pred.unsqueeze(2).expand_as(target)
            mse = (expanded_pred.float() - target.float()).pow(2).mean(dim=(0, 1, 3)).sum()
            cosine = (1.0 - F.cosine_similarity(expanded_pred, target, dim=-1)).mean(dim=(0, 1)).sum()
        else:
            raise RuntimeError(
                f"Unsupported prediction/target rank combination: pred={pred.dim()} target={target.dim()}"
            )
        loss = self.mse_weight * mse + self.cosine_weight * cosine
        return {
            "loss": loss,
            "mse": mse,
            "cosine": cosine,
        }
