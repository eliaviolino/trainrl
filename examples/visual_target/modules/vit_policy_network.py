import torch
import torch.nn as nn


class ViTPolicyNetwork(nn.Module):
    """
    Vision Transformer (ViT) backbone for RL policies.

    Takes a raw RGB image and outputs action distribution inputs
    and a value estimate — same interface as TransformerPolicyNetwork.

    Args:
        image_size   (int)  : Height and width of the (square) input image
        patch_size   (int)  : Size of each square patch. image_size % patch_size == 0
        in_channels  (int)  : Number of image channels (3 for RGB)
        action_dim   (int)  : Number of continuous action dimensions
        d_model      (int)  : Transformer embedding dimension
        nhead        (int)  : Number of attention heads
        num_layers   (int)  : Number of TransformerEncoderLayer stacks
        dim_feedforward(int): FFN hidden size inside each encoder layer
        dropout      (float): Dropout probability
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
    ):
        super().__init__()

        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        self.patch_size = patch_size
        self.d_model = d_model

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
        #    summary of the image, then feed it to the policy and value heads.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3. Learned positional embeddings — one per patch + one for [CLS]
        #    These are LEARNED (not sinusoidal) because spatial position in
        #    an image is 2D and learned embeddings handle that better.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        self.dropout = nn.Dropout(dropout)

        # 4. Standard Transformer encoder — same as your previous module
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        # 5. Output heads — same interface as TransformerPolicyNetwork
        # [mean | log_std] concatenated for continuous actions
        policy_output_dim = 2 * action_dim  if is_continuous else action_dim
        self.policy_head = nn.Linear(d_model, policy_output_dim)
        self.value_head = nn.Linear(d_model, 1)

        # Weight initialization — important for ViT stability
        self._init_weights()

    def _init_weights(self):
        # CLS token and positional embeddings: small normal init
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        # Linear layers: truncated normal (standard ViT practice)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split image tensor into a sequence of flattened patches.

        Args:
            x: (batch, C, H, W)
        Returns:
            patches: (batch, num_patches, patch_dim)
        """
        B, C, H, W = x.shape
        P = self.patch_size

        # Reshape into a grid of patches, then flatten each patch
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)      # (B, H/P, W/P, C, P, P)
        x = x.reshape(B, -1, C * P * P)       # (B, num_patches, patch_dim)
        return x

    def forward(self, obs: torch.Tensor) -> dict:
        """
        Args:
            obs: (batch, C, H, W)  — normalized pixel values expected [0, 1]
                 Note: RLlib delivers (batch, H, W, C) — see ViTRLModule
                 for the channel permutation

        Returns:
            dict with keys: 'logits' (action_dist_inputs), 'value'
        """
        B = obs.shape[0]

        # Step 1: split into patches and embed
        patches = self._to_patches(obs)            # (B, num_patches, patch_dim)
        tokens = self.patch_embedding(patches)     # (B, num_patches, d_model)

        # Step 2: prepend [CLS] token (expand to batch size)
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        tokens = torch.cat([cls_tokens, tokens], dim=1) # (B, 1+num_patches, d_model)

        # Step 3: add positional embeddings + dropout
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)

        # Step 4: run Transformer (no causal mask — all patches are visible)
        out = self.transformer(tokens)             # (B, 1+num_patches, d_model)
        out = self.norm(out)

        # Step 5: read [CLS] token (index 0) — global image representation
        cls_out = out[:, 0, :]                    # (B, d_model)

        logits = self.policy_head(cls_out)        # (B, 2*action_dim)
        value = self.value_head(cls_out)          # (B, 1)

        return {"logits": logits, "value": value}