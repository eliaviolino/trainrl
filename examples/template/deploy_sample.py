# Sample deployment script — configure these parameters to match your training setup
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
CHECKPOINT_DIR    = "your_checkpoint_path_here"

if ARCH_TYPE in ["mlp", "lstm"]:
    # MLP / LSTM specific
    OBS_DIM           = 4
    OBS_SHAPE         = None           # only used by CNN
    ACT_DIM           = 2
    HIDDEN_SIZES      = [64, 64]      # MLP only
    FUN_ACTIVATION    = "tanh"           # MLP only
    LSTM_HIDDEN       = 64             # LSTM only
    LSTM_LAYERS       = 1               # LSTM only
elif ARCH_TYPE == "cnn":
    # CNN specific
    OBS_SHAPE         = (3, 64, 64)     # (C, H, W)
    ACT_DIM           = 8
    CONV_CONFIGS      = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]  # (out_ch, kernel, stride)
    CONV_ACTIVATION   = "relu"

# Continuous action range (from your gym Box space)
ACTION_LOW        = np.array([-1.0, -1.0])
ACTION_HIGH       = np.array([ 1.0,  1.0])

# --- Step 1: Build the model ---
if ARCH_TYPE == "mlp":
    model = DeploymentPolicyMLP(OBS_DIM, ACT_DIM, HIDDEN_SIZES, ACTION_SPACE_TYPE, FUN_ACTIVATION)
elif ARCH_TYPE == "lstm":
    model = DeploymentPolicyLSTM(OBS_DIM, ACT_DIM, HIDDEN_SIZES,LSTM_HIDDEN, LSTM_LAYERS, ACTION_SPACE_TYPE, FUN_ACTIVATION)
elif ARCH_TYPE == "cnn":
    model = DeploymentPolicyCNN(OBS_SHAPE, ACT_DIM, CONV_CONFIGS, ACTION_SPACE_TYPE, CONV_ACTIVATION)

# --- Step 2: Load weights ---
model = load_weights(model, CHECKPOINT_DIR)

# --- Deployment loop ---
hidden_state = model.get_initial_state() if ARCH_TYPE == "lstm" else None

raw_obs = ...  # get this from your sensor / simulator

# --- Step 3: Preprocess ---
obs_tensor = preprocess_obs(raw_obs, ARCH_TYPE, obs_shape=OBS_SHAPE)

# --- Step 4: Forward pass ---
logits, hidden_state = run_inference(model, obs_tensor, ARCH_TYPE, hidden_state)

# --- Step 5: Convert to control ---
action = logits_to_action(logits, ACTION_SPACE_TYPE, (ACTION_LOW, ACTION_HIGH))

print("Control command:", action)  # e.g. [0.72, -0.14]