import torch

from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

# Import your standalone Transformer network
from modules.transformer_network import TransformerPolicyNetwork

import gymnasium as gym

class TransformerRLModule(TorchRLModule, ValueFunctionAPI):
    """
    RLlib-compatible wrapper around TransformerPolicyNetwork.

    Config kwargs (passed via RLModuleConfig.model_config_dict):
        d_model         (int)   : Transformer embedding dim. Default: 64
        nhead           (int)   : Number of attention heads. Default: 4
        num_layers      (int)   : Number of TransformerEncoderLayer stacks. Default: 2
        dim_feedforward (int)   : FFN hidden size inside each layer. Default: 128
        max_seq_len     (int)   : Maximum observation history length. Default: 16
        dropout         (float) : Dropout probability. Default: 0.1
    """

    @override(TorchRLModule)
    def setup(self):
        # Pull environment dimensions from the RLModuleConfig
        obs_dim = self.config.observation_space.shape[0]
        action_space = self.config.action_space

        # Detect action space type — no manual flag required
        is_continuous = isinstance(action_space, gym.spaces.Box)
        action_dim = action_space.shape[0] if is_continuous else action_space.n

        # Pull architecture hyperparameters from model_config_dict
        # This is the dict you pass when registering the module — this
        # is what makes your module configurable for your labmates.
        cfg = self.config.model_config_dict

        self.model = TransformerPolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 128),
            max_seq_len=cfg.get("max_seq_len", 16),
            dropout=cfg.get("dropout", 0.1),
        )

    def _prepare_obs(self, batch: dict) -> torch.Tensor:
        """
        RLlib delivers observations as (batch, obs_dim) flat tensors.
        The Transformer expects (batch, seq_len, obs_dim).

        For now we treat each observation as a sequence of length 1.
        This is the simplest correct approach — the Transformer still
        runs, but without history. In a future step you can extend this
        to pass a true observation history via the 'state' mechanism.
        """
        obs = batch[Columns.OBS]          # (batch, obs_dim)
        return obs.unsqueeze(1)           # (batch, 1, obs_dim)

    @override(TorchRLModule)
    def _forward_inference(self, batch: dict, **kwargs) -> dict:
        obs_seq = self._prepare_obs(batch)
        out = self.model(obs_seq)
        return {Columns.ACTION_DIST_INPUTS: out["action_dist_inputs"]}

    @override(TorchRLModule)
    def _forward_exploration(self, batch: dict, **kwargs) -> dict:
        # Same as inference during simple rollouts —
        # stochasticity comes from the action distribution, not the network
        obs_seq = self._prepare_obs(batch)
        out = self.model(obs_seq)
        return {Columns.ACTION_DIST_INPUTS: out["action_dist_inputs"]}

    @override(TorchRLModule)
    def _forward_train(self, batch: dict, **kwargs) -> dict:
        obs_seq = self._prepare_obs(batch)
        out = self.model(obs_seq)
        return {
            Columns.ACTION_DIST_INPUTS: out["action_dist_inputs"],
            Columns.VF_PREDS: out["value"].squeeze(-1),  # (batch,) not (batch, 1)
        }
    
    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict, embeddings=None) -> "torch.Tensor":
        """
        Called by PPO's advantage postprocessor outside of the forward pass.
        Must return a flat (batch,) tensor of value estimates.
        """
        out = self.model(self._prepare_obs(batch))
        return out["value"].squeeze(-1)  # (batch,) — not (batch, 1)