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
    """

    @override(TorchRLModule)
    def setup(self):
        obs_shape = self.config.observation_space.shape  # (H, W, C)
        action_space = self.config.action_space
        cfg = self.config.model_config_dict

        # Detect action space type — no manual flag required
        is_continuous = isinstance(action_space, gym.spaces.Box)
        action_dim = action_space.shape[0] if is_continuous else action_space.n

        self.model = ViTPolicyNetwork(
            image_size=obs_shape[0],               # assumes square image
            patch_size=cfg.get("patch_size", 14),
            in_channels=obs_shape[2],              # C from (H, W, C)
            action_dim=action_dim,
            is_continuous=is_continuous,
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )

    def _prepare_obs(self, batch: dict) -> torch.Tensor:
        obs = batch[Columns.OBS]                  # (B, H, W, C), uint8

        # Normalize pixels to [0, 1] and convert to float
        obs = obs.float() #/ 255.0

        # Permute to (B, C, H, W) — PyTorch's channel-first convention
        obs = obs.permute(0, 3, 1, 2)
        return obs

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