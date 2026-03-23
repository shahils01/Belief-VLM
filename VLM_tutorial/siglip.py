import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size = 768,
        intermediate_size = 3072,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        num_channels = 3,
        image_size = 224,
        patch_size = 16,
        layer_norm_eps = 1e-6,
        attention_dropout = 0.0,
        num_image_tokens: int = None,
        **kwargs):
        
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size 
        self.patch_size = config.patch_size

        self.patch_embedding = nn.conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = "valid" # No padding is added

        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches # Encodings == num_patches [Because we  need positional info of patches]
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((-1, 1)),
            persistent = False,
        ) # Do not want to include in gradient computation



    def forward(self, pixel_values):

        _, _, height, width = pixel_values.shape # [B, C, H, W]

        # Pass it through the convolution Layer to get patch embedding
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten the Patches
        # [B, E_dim, N_patches_H, N_patches_W] -> [B, E_dim, N_patches]
        embeddings = patch_embeds.flatten(2)

        # Transpose
        # [B, E_dim, num_patches]
        embeddings = embeddings.transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SiglipMLP(nn.Module):

    def __init__(self, config:SiglipVisionConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)

        # Apply Non-linear activation Gelu
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")

        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipAttention(nn.Module):

    ''' 
    ViT : contextual embeddings [B, N, D] -> [B, N, D] , but the contextual embeddings capture 
    info about other patches as well. It is different in language model. In language model, each
    token will have information of previous tokens. [t_2] -> [t_2, t_1]; [t_3] -> [t_3, t_2, t_1].
    I -> [I]
    like -> [I Like]
    pizza -> [I like Pizza]
    '''


    def __init__(self, config:SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(p=config.attention_dropout)
        self.proj_dropout = nn.Dropout(p=config.proj_dropout)

        assert self.embed_dim % self.num_heads == 0 

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)


    def forward(self, hidden_states):

        # Shape: [B, N, D]
        batch_size, seq_length, _ = hidden_states.size()

        # Shapes remain unchanges (adding parameters only): [B, N, D] 
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        ''' 
        [B, N, D]  @ [D, D] -> [B, N, D] -> [B, N, N_head,  D_head]        
        '''
        query_states = query_states.view(batch_size,  seq_length, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size,  seq_length, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size,  seq_length, self.num_heads, self.head_dim).transpose(1,2)

        # [B, num_heads, seq_len, seq_len]
        '''
        Attn Weights: Compute the weight between each token within a sequence (key) with every 
        other token (query). so Q@K.T -> attention weights.  
        '''
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        '''
        Attention Masking: We also have  maks for future cells where we 
        give -ve Inf weights to the future cells which vecome zero after applying
        softmax.
        '''
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        # (QK.T) * V
        attn_outputs = torch.matmul(attn_weights, value_states)

        attn_outputs = attn_outputs.transpose(1,2).contiguous()
        attn_outputs = attn_outputs.reshape(batch_size, seq_length, self.embed_dim)

        '''
        W_o (out proj): Projection to learn the mixing of different heads 
        '''
        attn_outputs = self.out_proj(attn_outputs)

        return attn_outputs, attn_weights


class SiglipEncoderLayer(nn.Module):

    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) 
    
    def forward(self, 
                hidden_states):

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)    # Layer Norm does not change dimension

        hidden_states, _ = self.self_attn(hidden_states=hidden_states)

        # Element wise addition of residual from the start + hidden_states from attention
        hidden_states = residual + hidden_states     

        residual = hidden_states

        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states  

        return hidden_states

class SiglipEncoder(nn.Module):

    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, 
                input_embeds):
        
        hidden_states = input_embeds
        for encoder_layer in self.layers:

            hidden_states = encoder_layer(hidden_states)

        return hidden_states
    

class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
    

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds = hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state



class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        return self.vision_model(pixel_values = pixel_values)