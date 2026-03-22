import torch
import torch.nn as nn
from typing import Optional, Tuple


class RecurrentBeliefQueryModule(nn.Module):
    """
    Recurrent belief query module.

    Inputs
    ------
    Z_v : torch.Tensor
        Vision tokens of shape [B, N_v, D]
    Z_x : torch.Tensor
        Text tokens of shape [B, N_x, D]
    prev_belief : Optional[torch.Tensor]
        Previous belief tokens of shape [B, M, D], or None

    Outputs
    -------
    belief : torch.Tensor
        Updated belief tokens of shape [B, M, D]
    attn_weights : torch.Tensor
        Cross-attention weights of shape [B, M, N_v + N_x]
    """

    def __init__(
        self,
        d_model: int,
        num_queries: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_text_conditioning: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_queries = num_queries
        self.use_text_conditioning = use_text_conditioning

        # Learned base belief queries: [1, M, D]
        self.q0 = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        # Optional text-conditioning for the belief query
        if use_text_conditioning:
            self.text_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        # Cross-attention: belief query attends to joint multimodal memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Normalization + FFN after cross-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Recurrent update on belief tokens
        # We flatten [B, M, D] -> [B*M, D] for GRUCell
        self.gru = nn.GRUCell(d_model, d_model)

    def _build_query(
        self,
        Z_x: torch.Tensor,
        prev_belief: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build current belief query from:
        - learned base query
        - optional text-conditioned offset
        - optional previous belief
        """
        B = Z_x.size(0)

        # Base query repeated across batch: [B, M, D]
        q = self.q0.expand(B, -1, -1)

        # Add text-conditioned offset
        if self.use_text_conditioning:
            text_summary = Z_x.mean(dim=1)  # [B, D]
            text_offset = self.text_proj(text_summary).unsqueeze(1)  # [B, 1, D]
            q = q + text_offset

        # Add previous belief if available
        if prev_belief is not None:
            q = q + prev_belief

        return q

    def forward(
        self,
        Z_v: torch.Tensor,
        Z_x: torch.Tensor,
        prev_belief: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        Z_v : [B, N_v, D]
        Z_x : [B, N_x, D]
        prev_belief : [B, M, D] or None
        key_padding_mask : [B, N_v + N_x] or None
            True for positions that should be masked.

        Returns
        -------
        belief : [B, M, D]
        attn_weights : [B, M, N_v + N_x]
        """
        assert Z_v.dim() == 3, f"Z_v must be [B, N_v, D], got {Z_v.shape}"
        assert Z_x.dim() == 3, f"Z_x must be [B, N_x, D], got {Z_x.shape}"
        assert Z_v.size(0) == Z_x.size(0), "Batch sizes must match"
        assert Z_v.size(2) == self.d_model and Z_x.size(2) == self.d_model, \
            f"Token dim must equal d_model={self.d_model}"

        B = Z_v.size(0)

        # Joint multimodal memory: [B, N_v + N_x, D]
        Z_m = torch.cat([Z_v, Z_x], dim=1)

        # Build query: [B, M, D]
        q = self._build_query(Z_x=Z_x, prev_belief=prev_belief)

        # Cross-attention: belief query attends to multimodal memory
        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=Z_m,
            value=Z_m,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        # attn_weights: [B, num_heads, M, N_v + N_x]
        attn_weights = attn_weights.mean(dim=1)  # -> [B, M, N_v + N_x]

        # Residual + FFN
        belief_candidate = self.norm1(q + attn_out)
        belief_candidate = self.norm2(belief_candidate + self.ffn(belief_candidate))

        # If no previous belief, initialize with candidate directly
        if prev_belief is None:
            belief = belief_candidate
        else:
            # Recurrent update token-by-token using GRUCell
            belief = self._gru_update(
                prev_belief=prev_belief,
                belief_candidate=belief_candidate,
            )

        return belief, attn_weights

    def _gru_update(
        self,
        prev_belief: torch.Tensor,
        belief_candidate: torch.Tensor,
    ) -> torch.Tensor:
        """
        GRU update over belief tokens.

        prev_belief:     [B, M, D]
        belief_candidate:[B, M, D]
        returns:         [B, M, D]
        """
        B, M, D = prev_belief.shape

        prev_flat = prev_belief.reshape(B * M, D)
        cand_flat = belief_candidate.reshape(B * M, D)

        updated_flat = self.gru(cand_flat, prev_flat)
        updated = updated_flat.reshape(B, M, D)

        return updated


class BeliefProjector(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, belief_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(belief_tokens)