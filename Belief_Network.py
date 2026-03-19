import torch
import torch.nn as nn
import numpy as np
import random
import math


class AttentionConfig:

    def __init__(self,
                 num_attention_heads = 12,
                 num_hidden_layers  =12,
                 state_dim = 12,
                 belief_dim = 3,
                 hidden_size = 768,
                 latent_dim = 32,
                 num_channels = 3,
                 image_size = 224,
                 patch_size = 16,
                 layer_norm_eps = 1e-6,
                 attention_dropout = 0.3,
                 proj_dropout = 0.2 
                 ):
        
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.state_dim = state_dim
        self.belief_dim = belief_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.proj_dropout = proj_dropout





class Embedding(nn.Module):

    def __init__(self,  config: AttentionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size

        # Use conv2d to extract features
        self.patch_embed = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = config.patch_size,
            stride = config.patch_size,
            padding = "valid"
        )

        '''
        Assuming  h = w . Else num_patches = int(H * W/ p^2) where p = patch_size 
        '''
        self.num_patches = (config.image_size//config.patch_size) **2
        self.num_positions = self.num_patches

        # Embed the Propioreceptive state 
        self.state_embed = nn.Linear(config.state_dim, self.embed_dim)

        # Belief Embedding:
        self.belief_embed = nn.Linear(config.belief_dim, self.embed_dim)

        self.flatten = nn.Flatten(2)

        self.positional_embeddings = nn.Parameter(
            torch.randn(1, self.num_positions +2, self.embed_dim),
            requires_grad=True
        )

        # self.register_buffer(
        #     "position_ids",
        #     torch.arange(self.num_positions).expand((-1,1)),
        #     persistent=False
        # )

       
    
    def forward(self, I_t, x_t, b_t):

        _, _, height, width = I_t.shape # [B, C, H, W]

        patch_embeds = self.patch_embed(I_t) # [B, E_dim, N_patch, N_patch]

        patch_embeddings = patch_embeds.flatten(2) 
        patch_embeddings = patch_embeddings.transpose(1, 2) # [B, N, E_dim]

        # State Embedings
        state_embeddings = self.state_embed(x_t) # [B, D]
        state_embeddings = state_embeddings.unsqueeze(1) #[B, 1, D]
        state_tokens = state_embeddings + self.positional_embeddings[:, 1:2]
        # state_embeddings = state_embeddings.repeat(1, self.num_patches, 1) # [B, N, D]

        # Prior Belief Embedding
        belief_embeddings = self.belief_embed(b_t)
        belief_tokens = belief_embeddings.unsqueeze(1) + self.positional_embeddings[:, :1] # [B, 1, D]


        '''
        Z = [b_t, p_i, W*x_{t}, PE(t)]
        Embeddings =  patch features + state features + position embeds
        Each embedding has shape [batch, N, embed_dim]
        
        '''
        patch_tokens = patch_embeddings  + self.positional_embeddings[:, 2:]

        embeddings = torch.cat([belief_tokens, state_tokens,  patch_tokens], dim=1)

        return embeddings, belief_tokens, state_tokens, patch_tokens





class LayerNormalisation(nn.Module):

    def __init__(self, config: AttentionConfig):
        super().__init__()

        self.eps = config.layer_norm_eps
        self.embed_dim = config.hidden_size
        self.alpha = nn.Parameter(torch.ones(self.embed_dim))
        self.bias = nn.Parameter(torch.zeros(self.embed_dim))
    
    def forward(self, x):

        mean = x.mean(dim=-1, keepdim = True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha*(x -mean)/(std + self.eps) + self.bias 


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(p=config.attention_dropout)
        self.proj_dropout = nn.Dropout(p=config.proj_dropout)

        assert self.embed_dim % self.num_heads == 0 

        self.w_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_o = nn.Linear(self.embed_dim, self.embed_dim)


    @staticmethod
    def attention(q, k, v, dropout:nn.Dropout = None):
        d_k = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
        scores = scores.softmax(dim = -1)

        if dropout is not None:
            scores = dropout(scores)
        
        return torch.matmul(scores, v), scores

    def forward(self, q, k ,v):

        batch_size, num_tokens, _ = q.shape

        query = self.w_q(q).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.w_k(k).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.w_v(v).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        x, self.attention_score = MultiHeadSelfAttention.attention(query, key, value, self.attn_dropout)

        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.num_heads * self.head_dim)


        return self.proj_dropout(self.w_o(x))



class AttentionPooling(nn.Module):

    def __init__(self, config: AttentionConfig):
        super().__init__()

        self.embed_dim = config.hidden_size

        # learned query vector dq
        self.query = nn.Parameter(torch.randn(self.embed_dim))


    def forward(self, t_i):
        """
        t_i : [B, N, D] patch tokens
        """

        # attention logits
        scores = torch.matmul(t_i, self.query)     # [B, N]

        alpha = torch.softmax(scores, dim=1)     # [B, N]

        # weighted pooling
        f_t = torch.sum(alpha.unsqueeze(-1) * t_i, dim=1)   # [B, D]

        return f_t, alpha




class TemporalContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.1, device = None):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features):

        # f_t: [B, T, D] t ~[0,1, ...,  T]
        B, T, D = features.shape

        features  = torch.nn.functional.normalize(features, dim=-1)

        loss = 0.0

        for t in range(T-1):

            anchor = features[:, t] # [B, D]
            # positive = features[:, t+1] # [B, D]

            # # Similrity matrix [B, D] @ [B, D, T] = [B, T] (logits)
            logits = torch.einsum("bd, btd -> bt", anchor, features) / self.temperature

            # Positive Index
            labels = torch.full((B,), t+1, dtype=torch.long, device=features.device)

            loss += torch.nn.functional.cross_entropy(logits, labels)

        return loss / (T-1)



'''
'''

class GaussianPriorNet(nn.Module):

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.mlp = nn.Linear(config.hidden_size, 2* config.latent_dim)

    
    def forward(self, b_t):
        params = self.mlp(b_t.squeeze())
        mu, log_sigma = params.chunk(2, dim=-1)
        return mu, log_sigma
    
    # def sample_prior(self):
    #     mu, log_sigma = self.forward()

    #     eps = torch.randn_like(mu)
    #     z_t = mu + torch.exp(log_sigma) * eps

    #     return z_t



class GaussianPosteriorNet(nn.Module):

    def __init__(self,  config:AttentionConfig):
        super().__init__()
        self.mlp = nn.Linear(2* config.hidden_size, 2 *config.latent_dim)
   
    def forward(self, b_t, f_t):

        # evidence: e_t = [b_t, f_t]
        # b_t : [B, 1, D], f_t: [B, D]
        e_t = torch.cat([b_t.squeeze(), f_t], dim=1)
        params = self.mlp(e_t)
        mu, log_sigma = params.chunk(2, dim=-1)
        return mu, log_sigma
    
    # def sample_posterior(self):
    #     mu, log_sigma = self.forward()

    #     eps = torch.randn_like(mu)
    #     z_t = mu + torch.exp(log_sigma) * eps

    #     return z_t #[B, latent_dim]



class GRUNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 2, dropout = 0.2, device = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first= True,
            dropout= self.dropout,
            device = self.device

        )

    def forward(self, b_prev, e_t, z_t, return_hidden=False):

        # evidence: e_t = [b_t, f_t]
        # b_t : [B, 1, D], f_t: [B, D]
        # z_t : [B, latent_dim]
        x = torch.cat([e_t, z_t], dim=-1)
        x = x.unsqueeze(1)
        
        # GRU update
        out, h = self.gru(x, b_prev)
        
        b_t = torch.relu(h[-1])

        if return_hidden:
            return b_t, h
        return b_t



class ELBO_loss:

    def __init__(self, beta=1.0, lambda_multi=0.5):
        super().__init__()
        
        self.beta  = beta
        self.lambda_multi = lambda_multi
        self.mse = nn.MSELoss(reduction='mean')

    def kl_divergence(self, mu_q, log_sigma_q, mu_p, log_sigma_p):


        sigma_q = torch.exp(log_sigma_q)
        sigma_p = torch.exp(log_sigma_p)

        kl = torch.log(sigma_p / sigma_q) + \
             (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5

        return kl.sum(dim=-1).mean()
    
    def forward(
        self,
        x_pred_1,
        x_pred_5,
        x_target_1,
        x_target_5,
        mu_q,
        log_sigma_q,
        mu_p,
        log_sigma_p
    ):

        # one-step prediction loss
        recon_1 = self.mse(x_pred_1, x_target_1)

        # multi-step prediction loss
        recon_5 = self.mse(x_pred_5, x_target_5)

        # KL divergence
        kl = self.kl_divergence(mu_q, log_sigma_q, mu_p, log_sigma_p)

        # ELBO
        loss = recon_1 + self.lambda_multi * recon_5 + self.beta * kl

        return loss, recon_1, recon_5, kl


class RecursiveBeliefNetwork(nn.Module):
    """
    Recursive latent state-space belief model:
    - Prior p(z_t | b_{t-1})
    - Posterior q(z_t | b_{t-1}, f_t)
    - Belief update b_t = GRU(b_{t-1}, [f_t, z_t])
    - Decoder predicts next visual embedding from [b_t, z_t]
    """

    def __init__(
        self,
        config: AttentionConfig,
        visual_dim: int,
        beta: float = 1.0,
        recon_weight: float = 1.0,
        temporal_nce_weight: float = 1.0,
        device=None,
    ):
        super().__init__()
        self.config = config
        self.visual_dim = visual_dim
        self.beta = beta
        self.recon_weight = recon_weight
        self.temporal_nce_weight = temporal_nce_weight
        self.device = device

        self.visual_proj = nn.Linear(visual_dim, config.hidden_size)
        self.state_proj = nn.Linear(config.state_dim, config.hidden_size)
        self.belief_token_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.attention = MultiHeadSelfAttention(config)
        self.pool = AttentionPooling(config)

        self.prior_net = GaussianPriorNet(config)
        self.posterior_net = GaussianPosteriorNet(config)
        self.gru = GRUNet(
            input_size=config.hidden_size + config.latent_dim,
            hidden_size=config.hidden_size,
            num_layers=2,
            dropout=config.proj_dropout,
            device=device,
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size + config.latent_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, visual_dim),
        )
        self.temporal_loss = TemporalContrastiveLoss(temperature=0.1, device=device)
        self.recon = nn.MSELoss(reduction="mean")

    def _reparameterize(self, mu, log_sigma):
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sigma) * eps

    def _kl_divergence(self, mu_q, log_sigma_q, mu_p, log_sigma_p):
        sigma_q = torch.exp(log_sigma_q)
        sigma_p = torch.exp(log_sigma_p)
        kl = torch.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p) ** 2) / (2 * sigma_p**2) - 0.5
        return kl.sum(dim=-1)

    def forward(self, visual_seq, state_seq=None):
        # visual_seq: [B, T, visual_dim]
        batch_size, time_steps, _ = visual_seq.shape
        device = visual_seq.device
        if state_seq is None:
            state_seq = torch.zeros(batch_size, time_steps, self.config.state_dim, device=device, dtype=visual_seq.dtype)

        hidden_state = torch.zeros(
            self.gru.num_layers,
            batch_size,
            self.config.hidden_size,
            device=device,
            dtype=visual_seq.dtype,
        )
        belief_prev = hidden_state[-1]

        pred_next_list = []
        target_next_list = []
        kl_list = []
        pooled_features = []
        belief_traj = []

        # iterate until T-1 since target is x_{t+1}
        for t in range(max(time_steps - 1, 0)):
            v_t = self.visual_proj(visual_seq[:, t])               # [B, D]
            x_t = self.state_proj(state_seq[:, t])                 # [B, D]
            b_tok = self.belief_token_proj(belief_prev)            # [B, D]

            tokens = torch.stack([b_tok, x_t, v_t], dim=1)         # [B, 3, D]
            attn_tokens = self.attention(tokens, tokens, tokens)   # [B, 3, D]
            f_t, _ = self.pool(attn_tokens[:, 2:, :])              # [B, D]

            mu_p, log_sigma_p = self.prior_net(belief_prev)
            mu_q, log_sigma_q = self.posterior_net(belief_prev.unsqueeze(1), f_t)
            z_t = self._reparameterize(mu_q, log_sigma_q)

            belief_t, hidden_state = self.gru(hidden_state, f_t, z_t, return_hidden=True)
            belief_prev = belief_t

            pred_next = self.decoder(torch.cat([belief_t, z_t], dim=-1))
            target_next = visual_seq[:, t + 1]

            pred_next_list.append(pred_next)
            target_next_list.append(target_next)
            kl_list.append(self._kl_divergence(mu_q, log_sigma_q, mu_p, log_sigma_p))
            pooled_features.append(f_t)
            belief_traj.append(belief_t)

        if len(pred_next_list) == 0:
            zero = visual_seq.new_zeros(())
            return {
                "loss": zero,
                "recon_loss": zero,
                "kl_loss": zero,
                "temporal_nce_loss": zero,
                "pred_next": visual_seq.new_zeros(batch_size, 0, self.visual_dim),
                "target_next": visual_seq.new_zeros(batch_size, 0, self.visual_dim),
                "belief_traj": visual_seq.new_zeros(batch_size, 0, self.config.hidden_size),
            }

        pred_next = torch.stack(pred_next_list, dim=1)
        target_next = torch.stack(target_next_list, dim=1)
        kl_loss = torch.stack(kl_list, dim=1).mean()
        recon_loss = self.recon(pred_next, target_next)

        temporal_nce_loss = visual_seq.new_zeros(())
        if self.temporal_nce_weight > 0 and len(pooled_features) > 1:
            temporal_nce_loss = self.temporal_loss(torch.stack(pooled_features, dim=1))

        total = self.recon_weight * recon_loss + self.beta * kl_loss + self.temporal_nce_weight * temporal_nce_loss
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "temporal_nce_loss": temporal_nce_loss,
            "pred_next": pred_next,
            "target_next": target_next,
            "belief_traj": torch.stack(belief_traj, dim=1),
        }
