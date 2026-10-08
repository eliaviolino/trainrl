# Transformer deployment example using trainrl.deployment — configure these parameters to match your training setup
# Usage (from the example folder):
#   python deployment/deploy_transformer.py results/<run_name>/<trial_folder>/best_checkpoint
import sys
import numpy as np
from trainrl.deployment import (
    DeploymentPolicyMLP,
    DeploymentPolicyLSTM,
    DeploymentPolicyCNN,
    DeploymentPolicyTransformer,
    DeploymentPolicyViT,
    load_weights,
    preprocess_obs,
    run_inference,
    logits_to_action,
)

ACTION_SPACE_TYPE = "continuous"    # "continuous" or "discrete"
CHECKPOINT_DIR    = sys.argv[1] if len(sys.argv) > 1 else sys.exit(f"Usage: python {sys.argv[0]} <path/to/best_checkpoint>")

# TRANSFORMER specific
OBS_DIM           = 4
ACT_DIM           = 2
D_MODEL           = 64
NHEAD             = 4
NUM_LAYERS        = 2
DIM_FEEDFORWARD   = 128
DROPOUT           = 0.0
MAX_SEQ_LEN       = 20

# Continuous action range (from your gym Box space)
ACTION_LOW        = np.array([-1.0, -1.0])
ACTION_HIGH       = np.array([ 1.0,  1.0])

# --- Step 1: Build the model ---
model = DeploymentPolicyTransformer(
    obs_dim=OBS_DIM,            # flat observation size from your environment
    act_dim=ACT_DIM,            # from action space
    d_model=D_MODEL,           # model_config.d_model
    nhead=NHEAD,              # model_config.nhead
    num_layers=NUM_LAYERS,         # model_config.num_layers
    dim_feedforward=DIM_FEEDFORWARD,  # model_config.dim_feedforward
    max_seq_len=MAX_SEQ_LEN,       # model_config.max_seq_len
    dropout=DROPOUT,               # model_config.dropout
    action_space_type="continuous",
)

# --- Step 2: Load weights ---
model = load_weights(model, CHECKPOINT_DIR)

# --- Deployment loop ---
x = 0.7   # example observation values
y = -0.4
vx = 0.0
vy = 0.0
raw_obs = np.array([x, y, vx, vy])  # example raw observation from your environment

# --- Step 3: Preprocess ---
obs_tensor = preprocess_obs(raw_obs, arch_type="transformer")  # raw_obs shape: (seq_len, obs_dim)

# --- Step 4: Forward pass ---
logits, _  = run_inference(model, obs_tensor, arch_type="transformer")

# --- Step 5: Convert to control ---
action      = logits_to_action(logits, action_space_type="continuous", action_space_bounds=(ACTION_LOW, ACTION_HIGH))

print("Control command:", action)