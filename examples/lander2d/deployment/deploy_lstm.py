# LSTM deployment example using trainrl.deployment — configure these parameters to match your training setup
# Usage (from the example folder):
#   python deployment/deploy_lstm.py results/<run_name>/<trial_folder>/best_checkpoint
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

ARCH_TYPE         = "lstm"           # "mlp", "lstm", or "cnn"
ACTION_SPACE_TYPE = "continuous"    # "continuous" or "discrete"
CHECKPOINT_DIR    = sys.argv[1] if len(sys.argv) > 1 else sys.exit(f"Usage: python {sys.argv[0]} <path/to/best_checkpoint>")

# MLP / LSTM specific
OBS_DIM           = 4
OBS_SHAPE         = None
ACT_DIM           = 2
HIDDEN_SIZES      = [64, 64]      # MLP only
FUN_ACTIVATION    = "tanh"           # MLP only
LSTM_HIDDEN       = 64             # LSTM only
LSTM_LAYERS       = 1               # LSTM only

# Continuous action range (from your gym Box space)
ACTION_LOW        = np.array([-1.0, -1.0])
ACTION_HIGH       = np.array([ 1.0,  1.0])

# --- Step 1: Build the model ---
model = DeploymentPolicyLSTM(OBS_DIM, ACT_DIM, HIDDEN_SIZES, LSTM_HIDDEN, LSTM_LAYERS, ACTION_SPACE_TYPE, FUN_ACTIVATION)

# --- Step 2: Load weights ---
model = load_weights(model, CHECKPOINT_DIR)

# --- Deployment loop ---
hidden_state = model.get_initial_state() if ARCH_TYPE == "lstm" else None

x = 0.7   # example observation values
y = -0.4
vx = 0.0
vy = 0.0
raw_obs = np.array([x, y, vx, vy])  # example raw observation from your environment

# --- Step 3: Preprocess ---
obs_tensor = preprocess_obs(raw_obs, ARCH_TYPE, obs_shape=OBS_SHAPE)

# --- Step 4: Forward pass ---
logits, hidden_state = run_inference(model, obs_tensor, ARCH_TYPE, hidden_state)

# --- Step 5: Convert to control ---
action = logits_to_action(logits, ACTION_SPACE_TYPE, (ACTION_LOW, ACTION_HIGH))

print("Control command:", action)  # e.g. [0.72, -0.14]