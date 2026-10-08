import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Injected into the token sequence so the Transformer knows the order
    of timesteps. This is important in RL where time order matters.
    """
    def __init__(self, d_model: int, max_seq_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the encoding matrix once
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_seq_len, d_model)
        # Register as buffer: it's not a learnable parameter, but it
        # must travel with the model (e.g., during .to(device))
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerPolicyNetwork(nn.Module):
    """
    Causal Transformer backbone for RL policies.
    
    Inputs:
        obs: (batch, seq_len, obs_dim)  — sequence of observations
    
    Outputs:
        action_dist_inputs:      (batch, action_dim)  — from the LAST timestep token
        value:       (batch, 1)            — state-value estimate
    
    The 'causal' mask ensures the model can only attend to past/current
    tokens, not future ones — this is critical for online RL where the
    agent acts step by step.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        is_continuous: bool = False,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_seq_len: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        # 1. Project raw observations to Transformer embedding space
        self.input_projection = nn.Linear(obs_dim, d_model)

        # 2. Add positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 3. Transformer encoder (we use "encoder" layers with a causal mask,
        #    mimicking a decoder-only architecture like GPT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # <-- IMPORTANT: input is (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Output heads — both read from the LAST token in the sequence
        policy_output_dim = 2*action_dim if is_continuous else action_dim
        self.policy_head = nn.Linear(d_model, policy_output_dim)  # action_dist_inputs
        self.value_head = nn.Linear(d_model, 1)             # V(s)

        self._d_model = d_model
        self._max_seq_len = max_seq_len

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular mask of -inf to block future token attention.
        Shape: (seq_len, seq_len)
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(
        self, obs: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> dict:
        """
        Args:
            obs:          (batch, seq_len, obs_dim)
            padding_mask: (batch, seq_len) — True where tokens are padding
                          (useful during batched training with variable-length episodes)
        Returns:
            dict with keys: 'action_dist_inputs', 'value', 'encoder_out'
        """
        batch, seq_len, _ = obs.shape

        # Project and encode positions
        x = self.input_projection(obs)          # (batch, seq_len, d_model)
        x = self.pos_encoding(x)

        # Build causal mask
        causal_mask = self._causal_mask(seq_len, obs.device)

        # Run Transformer
        encoder_out = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,  # None if not used
        )
        # encoder_out: (batch, seq_len, d_model)

        # Read from the LAST token (most recent observation)
        last_token = encoder_out[:, -1, :]      # (batch, d_model)

        action_dist_inputs = self.policy_head(last_token)   # (batch, action_dim)
        value = self.value_head(last_token)     # (batch, 1)

        return {
            "action_dist_inputs": action_dist_inputs,
            "value": value,
            "encoder_out": encoder_out,         # kept for flexibility in RLModule
        }