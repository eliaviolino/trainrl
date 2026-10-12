import numpy as np
import torch
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override

from modules.vit_policy_network import ViTPolicyNetwork

import gymnasium as gym

class ViTRLModule(TorchRLModule, ValueFunctionAPI):
    """
    Single RLModule that works with both Discrete and Box action spaces.
    No YAML flags needed — action space type is detected automatically.

    Single image or window of images is also detected from the observation space:
        (H, W, C)    → plain ViT
        (N, H, W, C) → temporal ViT over the last N frames (env_config.history_len)
    uint8 observation spaces are scaled to [0, 1]; float ones are used as they are.
    """

    @override(TorchRLModule)
    def setup(self):
        obs_space = self.config.observation_space
        obs_shape = obs_space.shape                      # (H, W, C) or (N, H, W, C)
        history_len = obs_shape[0] if len(obs_shape) == 4 else None
        H, W, C = obs_shape[-3:]
        assert H == W, f"ViTPolicyNetwork expects square images, got {H}x{W}"

        # Read the pixel scale from the space (not from the tensor dtype), so the
        # scaling is the same whether a batch arrives as uint8 or already as float
        self._pixel_scale = 1.0 / 255.0 if obs_space.dtype == np.uint8 else 1.0

        action_space = self.config.action_space
        cfg = self.config.model_config_dict

        # Detect action space type — no manual flag required
        is_continuous = isinstance(action_space, gym.spaces.Box)
        action_dim = action_space.shape[0] if is_continuous else action_space.n

        self.model = ViTPolicyNetwork(
            image_size=H,
            patch_size=cfg.get("patch_size", 14),
            in_channels=C,
            action_dim=action_dim,
            is_continuous=is_continuous,
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
            history_len=history_len,
            temporal_layers=cfg.get("temporal_layers", 2),
        )

    def _prepare_obs(self, batch: dict) -> torch.Tensor:
        obs = batch[Columns.OBS]                  # (B, H, W, C) or (B, N, H, W, C)

        # Convert to float and scale pixels to [0, 1]
        obs = obs.float() * self._pixel_scale

        # Move channels before H, W — PyTorch's channel-first convention
        # (B, H, W, C) → (B, C, H, W)   or   (B, N, H, W, C) → (B, N, C, H, W)
        return obs.movedim(-1, -3)

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        out = self.model(self._prepare_obs(batch))
        return {Columns.ACTION_DIST_INPUTS: out["logits"]}

    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        out = self.model(self._prepare_obs(batch))
        return {Columns.ACTION_DIST_INPUTS: out["logits"]}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        out = self.model(self._prepare_obs(batch))
        return {
            Columns.ACTION_DIST_INPUTS: out["logits"],
            Columns.VF_PREDS: out["value"].squeeze(-1),
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        out = self.model(self._prepare_obs(batch))
        return out["value"].squeeze(-1)