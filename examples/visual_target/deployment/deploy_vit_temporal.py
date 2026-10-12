# Temporal ViT deployment example using trainrl.deployment — configure these parameters to match your training setup
# Usage (from the example folder):
#   python deployment/deploy_vit_temporal.py results/<run_name>/<trial_folder>/best_checkpoint
import sys
from collections import deque
import numpy as np
import torch
from trainrl.deployment import (
    DeploymentPolicyViT,
    load_weights,
    preprocess_obs,
    run_inference,
    logits_to_action,
)

ACTION_SPACE_TYPE = "discrete"    # "continuous" or "discrete"
CHECKPOINT_DIR    = sys.argv[1] if len(sys.argv) > 1 else sys.exit(f"Usage: python {sys.argv[0]} <path/to/best_checkpoint>")

# Temporal ViT specific
OBS_SHAPE         = (64, 64, 3)   # shape of ONE frame from your environment
ACT_DIM           = 8
PATCH_SIZE        = 16             # model_config.patch_size
D_MODEL           = 128            # model_config.d_model
NHEAD             = 4              # model_config.nhead
NUM_LAYERS        = 2              # model_config.num_layers
DIM_FEEDFORWARD   = 256            # model_config.dim_feedforward
DROPOUT           = 0.1            # model_config.dropout
TEMPORAL_LAYERS   = 2              # model_config.temporal_layers
HISTORY_LEN       = 4              # env_config.history_len


# --- Step 1: Build the model ---
model = DeploymentPolicyViT(
    obs_shape=OBS_SHAPE,
    act_dim=ACT_DIM,
    patch_size=PATCH_SIZE,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    action_space_type=ACTION_SPACE_TYPE,
    history_len=HISTORY_LEN,
    temporal_layers=TEMPORAL_LAYERS,
)

# --- Step 2: Load weights ---
model = load_weights(model, CHECKPOINT_DIR)


# Example frames: green target at the center, red agent moving towards it.
# Replace this with your camera images (uint8 (H, W, 3) or float in [0, 1]).
def make_frame(lander_x, lander_y, image_size=64):
    frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    center = image_size // 2
    frame[center, center] = [0, 255, 0]                    # green target
    frame[center - lander_y, center + lander_x] = [255, 0, 0]  # red agent
    return frame

camera_frames = [make_frame(x, -3) for x in (-6, -4, -2, -1)]   # agent moving right


# --- Option A: full window at every step (simplest) ---
# Keep the last HISTORY_LEN frames, oldest first.
# At episode start, fill the window with the first frame: must match VisualTargetEnv.reset().
window = deque([camera_frames[0]] * HISTORY_LEN, maxlen=HISTORY_LEN)
for frame in camera_frames[1:]:
    window.append(frame)                                   # one new frame per control step

    # --- Step 3: Preprocess ---
    obs_tensor = preprocess_obs(np.stack(window), arch_type="vit", obs_shape=PATCH_SIZE)  # [1, N, num_patches, patch_dim]

    # --- Step 4: Forward pass ---
    logits, _ = run_inference(model, obs_tensor, arch_type="vit")

    # --- Step 5: Convert to control ---
    action = logits_to_action(logits, action_space_type=ACTION_SPACE_TYPE)
print("Control command (full window):  ", action)


# --- Option B: cached frame embeddings (faster on a robot) ---
# Each frame is encoded independently, so the embeddings of past frames never change.
# Encode only the newest frame per step and keep a window of embeddings instead of images.
def embed(frame):
    patches = preprocess_obs(frame, arch_type="vit", obs_shape=PATCH_SIZE)  # [1, num_patches, patch_dim]
    with torch.no_grad():
        return model.encode_frame(patches)                 # [1, d_model]

embeddings = deque([embed(camera_frames[0])] * HISTORY_LEN, maxlen=HISTORY_LEN)
for frame in camera_frames[1:]:
    embeddings.append(embed(frame))                        # ViT runs on ONE frame per step
    with torch.no_grad():
        logits = model.forward_from_embeddings(torch.stack(list(embeddings), dim=1))  # [1, N, d_model]
    action = logits_to_action(logits, action_space_type=ACTION_SPACE_TYPE)
print("Control command (cached frames):", action)
