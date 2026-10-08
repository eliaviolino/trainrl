# CNN deployment example using trainrl.deployment — configure these parameters to match your training setup
# Usage (from the example folder):
#   python deployment/deploy_cnn.py results/<run_name>/<trial_folder>/best_checkpoint
import sys
import numpy as np
from trainrl.deployment import (
    DeploymentPolicyMLP,
    DeploymentPolicyLSTM,
    DeploymentPolicyCNN,
    load_weights,
    preprocess_obs,
    run_inference,
    logits_to_action,
)

ARCH_TYPE         = "cnn"           # "mlp", "lstm", or "cnn"
ACTION_SPACE_TYPE = "discrete"    # "continuous" or "discrete"
CHECKPOINT_DIR    = sys.argv[1] if len(sys.argv) > 1 else sys.exit(f"Usage: python {sys.argv[0]} <path/to/best_checkpoint>")

# CNN specific
OBS_SHAPE         = (3, 64, 64)     # (C, H, W)
ACT_DIM           = 8
CONV_CONFIGS      = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]  # (out_ch, kernel, stride)
CONV_ACTIVATION   = "relu"

# --- Step 1: Build the model ---
model = DeploymentPolicyCNN(OBS_SHAPE, ACT_DIM, CONV_CONFIGS, ACTION_SPACE_TYPE, CONV_ACTIVATION)

# --- Step 2: Load weights ---
model = load_weights(model, CHECKPOINT_DIR)

# --- Deployment loop ---
hidden_state = model.get_initial_state() if ARCH_TYPE == "lstm" else None

# Build a 64x64 RGB observation image (channels last) with a green target and red lander.
image_size = OBS_SHAPE[1] if OBS_SHAPE is not None else 64
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

# Use this image as the raw observation for the CNN branch
raw_obs = obs_buf

# --- Step 3: Preprocess ---
obs_tensor = preprocess_obs(raw_obs, ARCH_TYPE, obs_shape=OBS_SHAPE)

# --- Step 4: Forward pass ---
logits, hidden_state = run_inference(model, obs_tensor, ARCH_TYPE, hidden_state)

# --- Step 5: Convert to control ---
action = logits_to_action(logits, ACTION_SPACE_TYPE)

print("Control command:", action) 