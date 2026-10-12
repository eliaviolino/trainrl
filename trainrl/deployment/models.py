import torch
import torch.nn as nn

class DeploymentPolicyMLP(nn.Module):
    """
    Args:
        obs_dim    : total number of floats in the flattened observation
        act_dim    : number of action dimensions (for continuous) or
                     number of discrete actions
        hidden_sizes: list of hidden layer widths, e.g. [256, 256]
        action_space_type: "continuous" or "discrete"
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, action_space_type="continuous", activation="tanh"):
        super().__init__()
        self.action_space_type = action_space_type

        def build_actor_mlp():
            layers = []
            input_size = obs_dim
            act_cls = nn.ReLU if activation == "relu" else nn.Tanh
            for hidden_size in hidden_sizes:
                layers += [nn.Linear(input_size, hidden_size), act_cls()]
                input_size = hidden_size
            return nn.Sequential(*layers)

        self.encoder = nn.ModuleDict({
        "actor_encoder": nn.ModuleDict({"net": nn.ModuleDict({"mlp": build_actor_mlp()})}),
        "critic_encoder": nn.ModuleDict({"net": nn.ModuleDict({"mlp": build_actor_mlp()})}),
        })

        actor_mlp = self.encoder["actor_encoder"]["net"]["mlp"]

        # Policy head: outputs action distribution parameters
        # - continuous Box: act_dim * 2  (mean AND log_std for each dimension)
        # - discrete Discrete: act_dim   (one logit per action)
        pi_out_size = act_dim * 2 if action_space_type == "continuous" else act_dim
        self.pi = nn.ModuleDict({
            "net": nn.ModuleDict({
                "mlp": nn.Sequential(nn.Linear(actor_mlp[-2].out_features, pi_out_size))
            })
        })
        self.pi.log_std_clip_param_const = nn.Parameter(torch.zeros(1))

        self.vf = nn.ModuleDict({
            "net": nn.ModuleDict({
                "mlp": nn.Sequential(nn.Linear(actor_mlp[-2].out_features, 1))
            })
        })
        self.vf.log_std_clip_param_const = nn.Parameter(torch.zeros(1))

    def forward(self, obs_tensor):
        # obs_tensor shape: [1, obs_dim]
        features = self.encoder["actor_encoder"]["net"]["mlp"](obs_tensor)
        policy_output = self.pi["net"]["mlp"](features)
        return policy_output
    
    
class DeploymentPolicyLSTM(nn.Module):
    """
    A combined MLP + LSTM policy architecture.
    The MLP processes the observation at each timestep, and its output is fed into the LSTM.
    Args:
        obs_dim     : number of floats per timestep observation
        act_dim     : action dimensions or number of discrete actions
        mlp_hidden_sizes: list of hidden layer widths for the MLP encoder, e.g. [256, 256]
        lstm_hidden_size : LSTM cell size (e.g. 256)
        lstm_num_layers  : number of stacked LSTM layers (usually 1)
        action_space_type: "continuous" or "discrete"
    """
    def __init__(self, obs_dim, act_dim, mlp_hidden_sizes, lstm_hidden_size, lstm_num_layers=1, action_space_type="continuous", activation="tanh"):
        super().__init__()
        self.action_space_type = action_space_type

        # MLP encoder
        self.action_space_type = action_space_type

        # Build the encoder: a stack of Linear - Tanh layers
        def build_actor_mlp():
                    layers = []
                    input_size = obs_dim
                    act_cls = nn.ReLU if activation == "relu" else nn.Tanh
                    for hidden_size in mlp_hidden_sizes:
                        layers += [nn.Linear(input_size, hidden_size), act_cls()]
                        input_size = hidden_size
                    return nn.Sequential(*layers)
        
        self.encoder = nn.ModuleDict({
            "actor_encoder": nn.ModuleDict({
                "tokenizer": nn.ModuleDict({"net": nn.ModuleDict({"mlp": build_actor_mlp()})}),
                "lstm": nn.LSTM(
                    input_size=mlp_hidden_sizes[-1],
                    hidden_size=lstm_hidden_size,
                    num_layers=lstm_num_layers,
                    batch_first=True
                )
            }),
            "critic_encoder": nn.ModuleDict({
                "tokenizer": nn.ModuleDict({"net": nn.ModuleDict({"mlp": build_actor_mlp()})}),
                "lstm": nn.LSTM(
                    input_size=mlp_hidden_sizes[-1],
                    hidden_size=lstm_hidden_size,
                    num_layers=lstm_num_layers,
                    batch_first=True
                )
            }),
        })

        pi_out_size = act_dim * 2 if action_space_type == "continuous" else act_dim
        self.pi = nn.ModuleDict({
            "net": nn.ModuleDict({
                "mlp": nn.Sequential(nn.Linear(lstm_hidden_size, pi_out_size))
            })
        })
        self.pi.log_std_clip_param_const = nn.Parameter(torch.zeros(1))

        self.vf = nn.ModuleDict({
            "net": nn.ModuleDict({
                "mlp": nn.Sequential(nn.Linear(lstm_hidden_size, 1))
            })
        })
        self.vf.log_std_clip_param_const = nn.Parameter(torch.zeros(1))

    def get_initial_state(self):
        """Call this once before a new episode begins."""
        h0 = torch.zeros(self.encoder["actor_encoder"]["lstm"].num_layers, 1, self.encoder["actor_encoder"]["lstm"].hidden_size)
        c0 = torch.zeros(self.encoder["actor_encoder"]["lstm"].num_layers, 1, self.encoder["actor_encoder"]["lstm"].hidden_size)
        return (h0, c0)
    
    def forward(self, obs_tensor, hidden_state):
        # obs_tensor shape: [1, 1, obs_dim] (B=1, T=1 for inference)
        batch_size, seq_len, _ = obs_tensor.shape
        obs_flat = obs_tensor.view(batch_size * seq_len, -1) 
        features = self.encoder["actor_encoder"]["tokenizer"]["net"]["mlp"](obs_flat) 
        features_seq = features.view(batch_size, seq_len, -1) 
        lstm_out, new_hidden_state = self.encoder["actor_encoder"]["lstm"](features_seq, hidden_state)
        policy_output = self.pi["net"]["mlp"](lstm_out[:, -1, :])
        return policy_output, new_hidden_state
    
    
class DeploymentPolicyCNN(nn.Module):
    """
    Args:
        obs_shape   : (C, H, W) — channels, height, width of one observation frame
        act_dim     : action dimensions or number of discrete actions
        conv_configs: list of (out_channels, kernel_size, stride) tuples
        action_space_type: "continuous" or "discrete"
    """
    def __init__(self, obs_shape, act_dim, conv_configs, action_space_type="continuous", activation="relu"):
        super().__init__()
        self.action_space_type = action_space_type

        activation_cls = nn.ReLU if activation == "relu" else nn.Tanh

        def build_cnn():
            conv_layers = []
            in_channels = obs_shape[0]
            for out_channels, kernel_size, stride in conv_configs:
                padding = (kernel_size - stride) // 2
                conv_layers += [
                    nn.Identity(),
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
                    activation_cls(),
                ]
                in_channels = out_channels
            return nn.Sequential(*conv_layers)

        self.encoder = nn.ModuleDict({
            "actor_encoder": nn.ModuleDict({
                "net": nn.Sequential(
                    nn.ModuleDict({"cnn": build_cnn()})
                )
            }),
            "critic_encoder": nn.ModuleDict({
                "net": nn.Sequential(
                    nn.ModuleDict({"cnn": build_cnn()})
                )
            }),
        })

        actor_cnn = self.encoder["actor_encoder"]["net"][0].cnn

        dummy = torch.zeros(1, *obs_shape)
        flat_size = actor_cnn(dummy).view(1, -1).shape[1]

        pi_out_size = act_dim * 2 if action_space_type == "continuous" else act_dim
        self.pi = nn.ModuleDict({
            "net": nn.ModuleDict({
                "mlp": nn.Sequential(nn.Linear(flat_size, pi_out_size))
            })
        })
        self.pi.log_std_clip_param_const = nn.Parameter(torch.zeros(1))

        self.vf = nn.ModuleDict({
            "net": nn.ModuleDict({
                "mlp": nn.Sequential(nn.Linear(flat_size, 1))
            })
        })
        self.vf.log_std_clip_param_const = nn.Parameter(torch.zeros(1))

    def forward(self, obs_tensor):
        # obs_tensor shape: [1, C, H, W]
        conv_features = self.encoder["actor_encoder"]["net"][0].cnn(obs_tensor)
        flat_features = conv_features.view(conv_features.size(0), -1)
        policy_output = self.pi["net"]["mlp"](flat_features)
        return policy_output
    

class DeploymentPolicyTransformer(nn.Module):
    """
    Args:
        obs_dim         : number of floats in the flat observation (e.g. 4)
        act_dim         : action dimensions (continuous) or number of actions (discrete)
        d_model         : transformer embedding dimension (from model_config)
        nhead           : number of attention heads (from model_config)
        num_layers      : number of transformer encoder layers (from model_config)
        dim_feedforward : FFN hidden size (from model_config)
        history_len     : observation window length (from env_config.history_len)
        dropout         : transformer dropout rate (from model_config)
        action_space_type: "continuous" or "discrete"
    """
    def __init__(self, obs_dim, act_dim, d_model, nhead, num_layers,
                 dim_feedforward, history_len, dropout, action_space_type="continuous"):
        super().__init__()
        self.action_space_type = action_space_type

        self.input_projection = nn.Linear(obs_dim, d_model)

        # Positional encoding is a buffer (not a learned parameter)
        self.pos_encoding = nn.Module()
        pe = torch.zeros(1, history_len, d_model)
        self.pos_encoding.register_buffer("pe", pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        pi_out_size = act_dim * 2 if action_space_type == "continuous" else act_dim
        self.policy_head = nn.Linear(d_model, pi_out_size)
        self.value_head  = nn.Linear(d_model, 1)

    def forward(self, obs_tensor):
        # obs_tensor shape: [B, seq_len, obs_dim], oldest observation first
        x = self.input_projection(obs_tensor)               # [B, seq_len, d_model]
        x = x + self.pos_encoding.pe[:, :x.size(1)]        # add positional encoding

        # Causal mask, as in training: each token attends only to itself and the past
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)           # [B, seq_len, d_model]
        return self.policy_head(x[:, -1])                   # last token → [B, pi_out_size]
    

class _ViTFrameEncoder(nn.Module):
    """ViT over one image given as patches: [B, num_patches, patch_dim] → [B, d_model] ([CLS] output)."""
    def __init__(self, num_patches, patch_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patches):
        x = self.patch_embedding(patches)                             # [B, num_patches, d_model]
        cls = self.cls_token.expand(x.size(0), -1, -1)               # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1) + self.pos_embedding          # [B, num_patches+1, d_model]
        x = self.transformer(x)                                       # [B, num_patches+1, d_model]
        return self.norm(x)[:, 0]                                     # [B, d_model]


class _TemporalEncoder(nn.Module):
    """Causal Transformer over frame embeddings: [B, N, d_model] → [B, d_model] (last frame)."""
    def __init__(self, d_model, history_len, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        # Sinusoidal time encoding is a buffer (not a learned parameter), loaded from the checkpoint
        self.register_buffer("pe", torch.zeros(1, history_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, frame_embeddings):
        N = frame_embeddings.size(1)
        x = frame_embeddings + self.pe[:, :N]                         # add time encoding
        # Causal mask, as in training: each frame attends only to itself and the past
        causal_mask = torch.triu(
            torch.full((N, N), float("-inf"), device=x.device), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)                     # [B, N, d_model]
        return x[:, -1]                                               # last frame → [B, d_model]


class DeploymentPolicyViT(nn.Module):
    """
    Args:
        obs_shape         : (H, W, C) — shape of ONE raw image frame
        act_dim           : number of action dimensions (continuous) or actions (discrete)
        patch_size        : size of each square patch (from model_config)
        d_model           : transformer embedding dimension (e.g. 128)
        nhead             : number of attention heads (e.g. 4)
        num_layers        : number of transformer encoder layers per frame (e.g. 2)
        dim_feedforward   : FFN hidden size inside each encoder layer (e.g. 256)
        dropout           : transformer dropout rate (from model_config)
        action_space_type : "continuous" or "discrete"
        history_len       : frames per window (from env_config.history_len), or None for single images
        temporal_layers   : causal transformer layers over time (from model_config), with history_len only

    Input (from preprocess_obs(..., "vit")):
        single image : [B, num_patches, patch_dim]
        window       : [B, N, num_patches, patch_dim], oldest frame first

    For step-by-step deployment of the temporal ViT, encode_frame() + forward_from_embeddings()
    let you encode only the newest frame at each step and reuse the embeddings of the others.
    """
    def __init__(self, obs_shape, act_dim, patch_size, d_model, nhead, num_layers,
                 dim_feedforward, dropout, action_space_type="discrete",
                 history_len=None, temporal_layers=2):
        super().__init__()
        self.action_space_type = action_space_type
        self.patch_size = patch_size
        self.history_len = history_len

        H, W, C = obs_shape
        num_patches = (H // patch_size) * (W // patch_size)
        patch_dim   = patch_size * patch_size * C

        self.frame_encoder = _ViTFrameEncoder(
            num_patches, patch_dim, d_model, nhead, num_layers, dim_feedforward, dropout,
        )
        if history_len is not None:
            self.temporal_encoder = _TemporalEncoder(
                d_model, history_len, nhead, temporal_layers, dim_feedforward, dropout,
            )

        pi_out_size = act_dim * 2 if action_space_type == "continuous" else act_dim
        self.policy_head = nn.Linear(d_model, pi_out_size)
        self.value_head  = nn.Linear(d_model, 1)

    def encode_frame(self, patches):
        # patches: [B, num_patches, patch_dim] → [B, d_model]
        return self.frame_encoder(patches)

    def forward_from_embeddings(self, frame_embeddings):
        # frame_embeddings: [B, N, d_model], oldest frame first → policy output
        return self.policy_head(self.temporal_encoder(frame_embeddings))

    def forward(self, obs_tensor):
        if self.history_len is None:
            # obs_tensor shape: [B, num_patches, patch_dim]
            return self.policy_head(self.encode_frame(obs_tensor))

        # obs_tensor shape: [B, N, num_patches, patch_dim]
        B, N = obs_tensor.shape[:2]
        embeddings = self.encode_frame(obs_tensor.flatten(0, 1))     # [B*N, d_model]
        return self.forward_from_embeddings(embeddings.view(B, N, -1))
