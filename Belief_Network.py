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
            torch.randn(1, self.num_positions +1, self.embed_dim),
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
        state_embeddings = state_embeddings.repeat(1, self.num_patches, 1) # [B, N, D]

        # Prior Belief Embedding
        belief_embeddings = self.belief_embed(b_t)
        belief_embeddings = belief_embeddings.unsqueeze(1) + self.positional_embeddings[:, :1] # [B, 1, D]


        '''
        Z = [b_t, p_i, W*x_{t}, PE(t)]
        Embeddings =  patch features + state features + position embeds
        Each embedding has shape [batch, N, embed_dim]
        
        '''
        patch_tokens = patch_embeddings + state_embeddings + self.positional_embeddings[:, 1:]

        embeddings = torch.cat([belief_embeddings, patch_tokens], dim=1)

        return embeddings


class LayerNormalisation(nn.Module):

    def __init__(self, config: AttentionConfig):
        super().__init__()

        self.eps = config.layer_norm_eps
        self.emed_dim = config.hidden_size
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
        scores = torch.matmul(q, k.transpose(-2, -1))/torch.sqrt(d_k)
        scores = scores.softmax(dim = -1)

        if dropout is not None:
            scores = dropout(scores)
        
        return torch.matmul(scores, v), scores

    def forward(self, q, k ,v):

        batch_size, num_tokens, _ = q.shape

        query = self.w_q(q).view(batch_size, num_tokens, self.head, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, num_tokens, self.head, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, num_tokens, self.head, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadSelfAttention.attention(query, key, value, self.attn_dropout)

        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.head * self.d_k)


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
            positive = features[:, t+1] # [B, D]

            # Similrity matrix [B, D] @ [B, D, T] = [B, T] (logits)
            logits = torch.matmul(anchor, features.transpose(1,2))/ self.temperature

            # Positive Index
            labels = torch.full((B,), t+1, dtype=torch.long, device=self.device)

            loss += torch.nn.functional.cross_entropy(logits, labels)

        return loss / (T-1)





class GaussianMLP(nn.module):

    def __init__(self, prior):
        
        self.mlp = nn.Linear()

    
    def forward(self, x):
        mu, sigma = self.mlp()
        return mu, sigma