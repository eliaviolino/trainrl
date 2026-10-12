import math

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """
    Vision Transformer (ViT) over a single image: patches → [CLS] embedding.

    Args:
        image_size   (int)  : Height and width of the (square) input image
        patch_size   (int)  : Size of each square patch. image_size % patch_size == 0
        in_channels  (int)  : Number of image channels (3 for RGB)
        d_model      (int)  : Transformer embedding dimension
        nhead        (int)  : Number of attention heads
        num_layers   (int)  : Number of TransformerEncoderLayer stacks
        dim_feedforward(int): FFN hidden size inside each encoder layer
        dropout      (float): Dropout probability
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        self.patch_size = patch_size

        # Number of patches the image is split into
        num_patches = (image_size // patch_size) ** 2  # e.g. (84//14)^2 = 36

        # Dimension of each flattened patch
        patch_dim = in_channels * patch_size * patch_size  # e.g. 3*14*14 = 588

        # 1. Patch embedding: flatten + project each patch to d_model
        #    This replaces the CNN entirely — no convolutions.
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),          # normalize raw pixel values per patch
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. [CLS] token — a learnable vector prepended to the patch sequence.
        #    After the Transformer runs, we read this token to get a global
        #    summary of the image.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3. Learned positional embeddings — one per patch + one for [CLS]
        #    These are LEARNED (not sinusoidal) because spatial position in
        #    an image is 2D and learned embeddings handle that better.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        self.dropout = nn.Dropout(dropout)

        # 4. Standard Transformer encoder (no causal mask — all patches are visible)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split image tensor into a sequence of flattened patches.

        Args:
            x: (batch, C, H, W)
        Returns:
            patches: (batch, num_patches, patch_dim), each patch flattened as (C, P, P)
        """
        B, C, H, W = x.shape
        P = self.patch_size

        # Reshape into a grid of patches, then flatten each patch
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)      # (B, H/P, W/P, C, P, P)
        x = x.reshape(B, -1, C * P * P)       # (B, num_patches, patch_dim)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, C, H, W) — pixel values in [0, 1]
        Returns:
            (batch, d_model) — [CLS] embedding of each image
        """
        B = images.shape[0]

        # Step 1: split into patches and embed
        patches = self._to_patches(images)         # (B, num_patches, patch_dim)
        tokens = self.patch_embedding(patches)     # (B, num_patches, d_model)

        # Step 2: prepend [CLS] token (expand to batch size)
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        tokens = torch.cat([cls_tokens, tokens], dim=1) # (B, 1+num_patches, d_model)

        # Step 3: add positional embeddings + dropout
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)

        # Step 4: run Transformer
        out = self.transformer(tokens)             # (B, 1+num_patches, d_model)
        out = self.norm(out)

        # Step 5: read [CLS] token (index 0) — global image representation
        return out[:, 0, :]                        # (B, d_model)


class TemporalEncoder(nn.Module):
    """
    Causal Transformer over a window of frame embeddings: (B, N, d_model) → (B, d_model).

    Same design as the lander2d TransformerPolicyNetwork: sinusoidal time
    encoding, causal mask, and readout from the LAST (most recent) frame.
    """

    def __init__(
        self,
        d_model: int,
        history_len: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        # Sinusoidal time encoding, one row per frame in the window.
        # Register as buffer: it's not a learnable parameter, but it
        # must travel with the model (e.g., during .to(device))
        pe = torch.zeros(history_len, d_model)
        position = torch.arange(0, history_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, history_len, d_model)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_embeddings: (batch, N, d_model), oldest frame first
        Returns:
            (batch, d_model) — output at the most recent frame
        """
        N = frame_embeddings.shape[1]
        x = self.dropout(frame_embeddings + self.pe[:, :N, :])

        # Causal mask: each frame attends only to itself and earlier frames
        causal_mask = torch.triu(
            torch.full((N, N), float("-inf"), device=x.device), diagonal=1
        )
        out = self.transformer(x, mask=causal_mask)    # (B, N, d_model)
        return out[:, -1, :]                           # (B, d_model)


class ViTPolicyNetwork(nn.Module):
    """
    ViT policy for single images or windows of images.

    - history_len=None: obs is one image (B, C, H, W).
      FrameEncoder → [CLS] embedding → heads. This is the plain ViT.
    - history_len=N:    obs is a window (B, N, C, H, W), oldest frame first.
      FrameEncoder on every frame → (B, N, d_model) → TemporalEncoder → heads.
      The FrameEncoder sees space, the TemporalEncoder sees time.

    Args (in addition to the FrameEncoder ones):
        action_dim      (int) : Number of actions (discrete) or action dimensions (continuous)
        is_continuous   (bool): Output [mean | log_std] (2*action_dim) instead of logits
        history_len     (int) : Number of frames in the window, or None for single images
        temporal_layers (int) : Number of causal TransformerEncoderLayers over time
    """

    def __init__(
        self,
        image_size: int = 84,
        patch_size: int = 14,
        in_channels: int = 3,
        action_dim: int = 2,
        is_continuous: bool = False,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        history_len: int = None,
        temporal_layers: int = 2,
    ):
        super().__init__()

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        self.history_len = history_len

        # 1. Spatial: one embedding per image
        self.frame_encoder = FrameEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # 2. Temporal: one embedding per window (only with a history)
        if history_len is not None:
            self.temporal_encoder = TemporalEncoder(
                d_model=d_model,
                history_len=history_len,
                nhead=nhead,
                num_layers=temporal_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

        # 3. Output heads
        # [mean | log_std] concatenated for continuous actions
        policy_output_dim = 2 * action_dim if is_continuous else action_dim
        self.policy_head = nn.Linear(d_model, policy_output_dim)
        self.value_head = nn.Linear(d_model, 1)

        # Weight initialization — important for ViT stability
        self._init_weights()

    def _init_weights(self):
        # CLS token and positional embeddings: small normal init
        nn.init.normal_(self.frame_encoder.cls_token, std=0.02)
        nn.init.normal_(self.frame_encoder.pos_embedding, std=0.02)
        # Linear layers: truncated normal (standard ViT practice)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> dict:
        """
        Args:
            obs: (batch, C, H, W) or, with a history, (batch, N, C, H, W)
                 Pixel values in [0, 1]. RLlib delivers channels-last images:
                 see ViTRLModule for the permutation and scaling.

        Returns:
            dict with keys: 'logits' (action_dist_inputs), 'value'
        """
        if self.history_len is None:
            features = self.frame_encoder(obs)                 # (B, d_model)
        else:
            B, N = obs.shape[:2]
            # Fold time into the batch so every frame is encoded independently
            frames = obs.reshape(B * N, *obs.shape[2:])        # (B*N, C, H, W)
            embeddings = self.frame_encoder(frames)            # (B*N, d_model)
            embeddings = embeddings.reshape(B, N, -1)          # (B, N, d_model)
            features = self.temporal_encoder(embeddings)       # (B, d_model)

        logits = self.policy_head(features)       # (B, action_dim) or (B, 2*action_dim)
        value = self.value_head(features)         # (B, 1)

        return {"logits": logits, "value": value}
