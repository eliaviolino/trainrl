# ViT deployment example using trainrl.deployment — configure these parameters to match your training setup
# Usage (from the example folder):
#   python deployment/deploy_vit.py results/<run_name>/<trial_folder>/best_checkpoint
import sys
import numpy as np
from trainrl.deployment import (
    DeploymentPolicyMLP,
    DeploymentPolicyLSTM,
    DeploymentPolicyCNN,
    DeploymentPolicyViT,
    load_weights,
    preprocess_obs,
    run_inference,
    logits_to_action,
)

ACTION_SPACE_TYPE = "discrete"    # "continuous" or "discrete"
CHECKPOINT_DIR    = sys.argv[1] if len(sys.argv) > 1 else sys.exit(f"Usage: python {sys.argv[0]} <path/to/best_checkpoint>")

# ViT specific
OBS_SHAPE         = (64, 64, 3)   # from your environment
ACT_DIM           = 8
PATCH_SIZE        = 16             # model_config.patch_size
D_MODEL           = 128            # model_config.d_model
NHEAD             = 4              # model_config.nhead
NUM_LAYERS        = 2              # model_config.num_layers
DIM_FEEDFORWARD   = 256            # model_config.dim_feedforward
DROPOUT           = 0.1            # model_config.dropout



# --- Step 1: Build the model ---
model = DeploymentPolicyViT(
    obs_shape=OBS_SHAPE,   # from your environment
    act_dim=ACT_DIM,
    patch_size=PATCH_SIZE,           # model_config.patch_size
    d_model=D_MODEL,             # model_config.d_model
    nhead=NHEAD,                 # model_config.nhead
    num_layers=NUM_LAYERS,            # model_config.num_layers
    dim_feedforward=DIM_FEEDFORWARD,     # model_config.dim_feedforward
    dropout=DROPOUT,
    action_space_type=ACTION_SPACE_TYPE,
)

# --- Step 2: Load weights ---
model = load_weights(model, CHECKPOINT_DIR)

# --- Deployment loop ---
image_size = 64
obs_buf = np.zeros((image_size, image_size, 3), dtype=np.float32)
center = image_size // 2

# Example target and lander offsets (in pixels) relative to center.
# Replace these with your environment's target/lander coordinates.
target_x, target_y = 0, 0
tx = int(np.clip(center + target_x, 0, image_size - 1))
ty = int(np.clip(center - target_y, 0, image_size - 1))
obs_buf[ty, tx] = [0.0, 1.0, 0.0]  # green

lander_x, lander_y = 0, -1
lx = int(np.clip(center + lander_x, 0, image_size - 1))
ly = int(np.clip(center - lander_y, 0, image_size - 1))
obs_buf[ly, lx] = [1.0, 0.0, 0.0]  # red

# Use this image as the raw observation for the ViT branch
raw_obs = obs_buf

# --- Step 3: Preprocess ---
obs_tensor = preprocess_obs(raw_obs, arch_type="vit", obs_shape=PATCH_SIZE)

# --- Step 4: Forward pass ---
logits, _ = run_inference(model, obs_tensor, arch_type="vit")

# --- Step 5: Convert to control ---
action = logits_to_action(logits, action_space_type=ACTION_SPACE_TYPE)

print("Control command:", action) 