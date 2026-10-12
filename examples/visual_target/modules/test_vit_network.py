# Quick shape and gradient check of the ViT network, without RLlib.
# Usage (from the example folder):
#   python -m modules.test_vit_network
from modules.vit_policy_network import ViTPolicyNetwork
import torch

num_actions = 5   # e.g. left, right, up, down, noop

# --- Plain ViT: one image per sample ---
model = ViTPolicyNetwork(
    image_size=84, patch_size=14, in_channels=3, action_dim=num_actions
)

dummy_obs = torch.randn(4, 3, 84, 84)        # (B, C, H, W)
out = model(dummy_obs)

print(out["logits"].shape)   # Expected: (4, 5)  → one logit per action
print(out["value"].shape)    # Expected: (4, 1)

loss = out["logits"].sum() + out["value"].sum()
loss.backward()
print("Gradient check passed ✓")

# --- Temporal ViT: a window of 6 images per sample ---
model = ViTPolicyNetwork(
    image_size=84, patch_size=14, in_channels=3, action_dim=num_actions,
    history_len=6, temporal_layers=2,
)

dummy_obs = torch.randn(4, 6, 3, 84, 84)     # (B, N, C, H, W), oldest frame first
out = model(dummy_obs)

print(out["logits"].shape)   # Expected: (4, 5)  → one logit per action, from the newest frame
print(out["value"].shape)    # Expected: (4, 1)

loss = out["logits"].sum() + out["value"].sum()
loss.backward()
print("Gradient check passed ✓")
