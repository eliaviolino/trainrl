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
        max_seq_len     : maximum sequence length for positional encoding (from model_config)
        dropout         : transformer dropout rate (from model_config)
        action_space_type: "continuous" or "discrete"
    """
    def __init__(self, obs_dim, act_dim, d_model, nhead, num_layers,
                 dim_feedforward, max_seq_len, dropout, action_space_type="continuous"):
        super().__init__()
        self.action_space_type = action_space_type

        self.input_projection = nn.Linear(obs_dim, d_model)

        # Positional encoding is a buffer (not a learned parameter)
        self.pos_encoding = nn.Module()
        pe = torch.zeros(1, max_seq_len, d_model)
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
        # obs_tensor shape: [B, seq_len, obs_dim]
        x = self.input_projection(obs_tensor)               # [B, seq_len, d_model]
        x = x + self.pos_encoding.pe[:, :x.size(1)]        # add positional encoding
        x = self.transformer(x)                             # [B, seq_len, d_model]
        return self.policy_head(x.mean(dim=1))              # mean pool → [B, d_model]
    

class DeploymentPolicyViT(nn.Module):
    """
    Args:
        obs_shape         : (H, W, C) — raw image observation shape
        act_dim           : number of action dimensions (continuous) or actions (discrete)
        patch_size        : size of each square patch (from model_config)
        d_model           : transformer embedding dimension (e.g. 128)
        nhead             : number of attention heads (e.g. 4)
        num_layers        : number of transformer encoder layers (e.g. 2)
        dim_feedforward   : FFN hidden size inside each encoder layer (e.g. 256)
        dropout           : transformer dropout rate (from model_config)
        action_space_type : "continuous" or "discrete"
    """
    def __init__(self, obs_shape, act_dim, patch_size, d_model, nhead, num_layers,
                 dim_feedforward, dropout, action_space_type="discrete"):
        super().__init__()
        self.action_space_type = action_space_type
        self.patch_size = patch_size

        H, W, C = obs_shape
        num_patches = (H // patch_size) * (W // patch_size)
        patch_dim   = patch_size * patch_size * C

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

        pi_out_size = act_dim * 2 if action_space_type == "continuous" else act_dim
        self.policy_head = nn.Linear(d_model, pi_out_size)
        self.value_head  = nn.Linear(d_model, 1)

    def forward(self, obs_tensor):
        # obs_tensor shape: [B, num_patches, patch_dim]
        x = self.patch_embedding(obs_tensor)                          # [B, num_patches, d_model]
        cls = self.cls_token.expand(x.size(0), -1, -1)               # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1) + self.pos_embedding          # [B, num_patches+1, d_model]
        x = self.transformer(x)                                       # [B, num_patches+1, d_model]
        cls_features = self.norm(x[:, 0])                             # [B, d_model]
        return self.policy_head(cls_features)