from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BeliefPredictorConfig:
    embed_dim: int
    hidden_dim: int = 1024
    latent_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    num_belief_tokens: int = 4
    dropout: float = 0.1
    max_context_frames: int = 32
    target_frames: int = 2


class BeliefPredictionTransformer(nn.Module):
    def __init__(self, cfg: BeliefPredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.embed_dim, cfg.hidden_dim)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.embed_dim)
        self.context_pos_embed = nn.Parameter(torch.zeros(1, cfg.max_context_frames, cfg.hidden_dim))
        self.belief_queries = nn.Parameter(torch.zeros(1, cfg.num_belief_tokens, cfg.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=4 * cfg.hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=4 * cfg.hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_layers)
        self.memory_norm = nn.LayerNorm(cfg.hidden_dim)
        self.token_norm = nn.LayerNorm(cfg.hidden_dim)
        self.posterior_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.posterior_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.latent_to_query = nn.Linear(cfg.latent_dim, cfg.num_belief_tokens * cfg.hidden_dim)
        self.future_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def _split_context_and_target(self, frame_embeddings):
        total_frames = int(frame_embeddings.shape[1])
        if total_frames > self.cfg.max_context_frames:
            raise RuntimeError(
                f"context length {total_frames} exceeds max_context_frames={self.cfg.max_context_frames}."
            )
        if total_frames < 2:
            raise RuntimeError("Belief prediction requires at least 2 frames.")
        target_frames = min(int(self.cfg.target_frames), total_frames - 1)
        context_frames = total_frames - target_frames
        return frame_embeddings[:, :context_frames], frame_embeddings[:, context_frames:]

    def forward(self, frame_embeddings):
        context_embeddings, target_embeddings = self._split_context_and_target(frame_embeddings)
        context_len = int(context_embeddings.shape[1])
        memory = self.input_proj(context_embeddings)
        memory = memory + self.context_pos_embed[:, :context_len]
        memory = self.encoder(memory)
        memory = self.memory_norm(memory)

        pooled_memory = memory.mean(dim=1)
        mu = self.posterior_mu(pooled_memory)
        logvar = self.posterior_logvar(pooled_memory).clamp(min=-8.0, max=8.0)
        std = torch.exp(0.5 * logvar)
        latent = mu + torch.randn_like(std) * std

        belief_query_bias = self.latent_to_query(latent).view(
            frame_embeddings.shape[0],
            self.cfg.num_belief_tokens,
            self.cfg.hidden_dim,
        )
        belief_queries = self.belief_queries.expand(frame_embeddings.shape[0], -1, -1) + belief_query_bias
        belief_hidden = self.decoder(self.dropout(belief_queries), memory)
        belief_hidden = self.token_norm(belief_hidden)
        belief_tokens = self.output_proj(belief_hidden)

        belief_summary = belief_tokens.mean(dim=1)
        outputs = {
            "belief_tokens": belief_tokens,
            "future_pred": self.future_head(belief_summary),
            "future_target": target_embeddings.mean(dim=1),
            "reconstruction_pred": self.reconstruction_head(belief_summary),
            "reconstruction_target": context_embeddings.mean(dim=1),
            "mu": mu,
            "logvar": logvar,
        }
        return outputs


class BeliefAuxiliaryLoss(nn.Module):
    def __init__(
        self,
        future_weight: float = 1.0,
        reconstruction_weight: float = 0.5,
        kl_weight: float = 1e-3,
    ):
        super().__init__()
        self.future_weight = future_weight
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight

    @staticmethod
    def _embedding_loss(pred, target):
        mse = F.mse_loss(pred, target)
        cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        return mse + cosine, mse, cosine

    def forward(self, belief_outputs):
        future_loss, future_mse, future_cosine = self._embedding_loss(
            belief_outputs["future_pred"],
            belief_outputs["future_target"],
        )
        reconstruction_loss, reconstruction_mse, reconstruction_cosine = self._embedding_loss(
            belief_outputs["reconstruction_pred"],
            belief_outputs["reconstruction_target"],
        )
        mu = belief_outputs["mu"]
        logvar = belief_outputs["logvar"]
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = (
            self.future_weight * future_loss
            + self.reconstruction_weight * reconstruction_loss
            + self.kl_weight * kl
        )
        return {
            "loss": loss,
            "future": future_loss,
            "future_mse": future_mse,
            "future_cosine": future_cosine,
            "reconstruction": reconstruction_loss,
            "reconstruction_mse": reconstruction_mse,
            "reconstruction_cosine": reconstruction_cosine,
            "kl": kl,
        }
