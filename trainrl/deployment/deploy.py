import numpy as np
import pickle
import torch

def load_weights(model, checkpoint_dir):
    """
    checkpoint_dir: path to the 'best_checkpoint' folder.
    Returns: the same model, now with trained weights loaded.
    """
    weights_path = (
        f"{checkpoint_dir}/learner_group/learner/"
        f"rl_module/default_policy/module_state.pkl"
    )

    with open(weights_path, "rb") as f:
        state_dict = pickle.load(f)

    # Convert NumPy arrays in the checkpoint to torch tensors.
    # Copy arrays first to avoid warnings when numpy arrays are read-only.
    state_dict = {
        k: torch.as_tensor(v.copy()) if isinstance(v, np.ndarray) else v
        if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        for k, v in state_dict.items()
    }

    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

    # strict=True (default) will raise an error if ANY key is missing or
    # has a wrong shape — this is your safety net against architecture mismatches
    model.load_state_dict(state_dict, strict=True)

    # eval() switches off training-only behaviors:
    # - Dropout layers will NOT randomly zero out activations
    # - BatchNorm layers will use running statistics, not batch statistics
    model.eval()
    print("Weights loaded successfully.")
    return model


def preprocess_obs(raw_obs, arch_type, obs_shape=None):
    """
    raw_obs   : numpy array from your environment's step() or reset()
    arch_type : "mlp", "lstm", or "cnn"
    obs_shape : required for CNN — tuple (C, H, W) as the network expects it

    Returns a torch.Tensor ready for the network.
    """

    if arch_type == "mlp":
        # raw_obs is already a flat 1D vector, e.g. shape (12,)
        # Result shape: [1, obs_dim]  — the '1' is the batch dimension
        obs_tensor = torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0)
    elif arch_type == "lstm":
        # raw_obs is a flat 1D vector, same as MLP
        # But LSTM needs seq_len dimension too
        # Result shape: [1, 1, obs_dim]  — [batch, seq_len, features]
        obs_tensor = torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    elif arch_type == "cnn":
        # raw_obs is a numpy image, typically shape (H, W, C) from gym (channels last)
        # Step 1: convert to float32 and normalize pixel values from [0, 255] to [0.0, 1.0]
        if raw_obs.max() > 1.0:
            obs = raw_obs.astype(np.float32) / 255.0
        else:
            obs = raw_obs.astype(np.float32)

        # Step 2: transpose from (H, W, C) to (C, H, W) — PyTorch is channels-first
        if obs.ndim == 3 and obs.shape[-1] == obs_shape[0]:  
            obs = np.transpose(obs, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        # else assume obs is already in (C, H, W) format

        # Step 3: add batch dimension
        # result shape: [1, C, H, W]
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    # inside preprocess_obs — add after the "vit" branch:
    elif arch_type == "transformer":
        # raw_obs: (seq_len, obs_dim) — a window of recent flat observations
        # Result shape: [1, seq_len, obs_dim]
        obs_tensor = torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0)
    elif arch_type == "vit":
        # raw_obs: (H, W, C) — obs_shape carries patch_size (single int, from model_config)
        obs = raw_obs.astype(np.float32) / 255.0 if raw_obs.max() > 1.0 else raw_obs.astype(np.float32)
        H, W, C = obs.shape
        p = obs_shape  # patch_size
        patches = obs.reshape(H // p, p, W // p, p, C).transpose(0, 2, 1, 3, 4).reshape(-1, p * p * C)
        obs_tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(0)

    return obs_tensor


def run_inference(model, obs_tensor, arch_type, hidden_state=None):
    """
    model       : your loaded DeploymentPolicy model (in eval mode)
    obs_tensor  : output of preprocess_obs()
    arch_type   : "mlp", "lstm", "cnn", "transformer", or "vit"
    hidden_state: (h, c) tuple — required for LSTM, None otherwise

    Returns:
        output_logits : raw tensor from the policy head
        new_hidden    : updated (h, c) for LSTM, None otherwise
    """
    with torch.no_grad():
        if arch_type in ("mlp", "cnn", "transformer", "vit"):
            output_logits = model(obs_tensor)
            new_hidden = None
        elif arch_type == "lstm":
            if hidden_state is None:
                raise ValueError("LSTM requires a hidden_state. Call model.get_initial_state() first.")
            output_logits, new_hidden = model(obs_tensor, hidden_state)
    return output_logits, new_hidden


def logits_to_action(output_logits, action_space_type, action_space_bounds=None):
    """
    output_logits      : tensor of shape [1, N] from run_inference()
    action_space_type  : "continuous" or "discrete"
    action_space_bounds: for continuous only — tuple (low, high) as numpy arrays
                         matching the gym Box space, e.g. (np.array([-1.0, -1.0]),
                                                            np.array([ 1.0,  1.0]))

    Returns: numpy array of the final action, ready to use
    """

    if action_space_type == "continuous":
        act_dim = output_logits.shape[1] // 2

        # Split the output into mean and log_std parts
        mean    = output_logits[:, :act_dim]    # shape [1, act_dim]
        log_std = output_logits[:, act_dim:]    # shape [1, act_dim]

        # Deterministic deployment: use mean directly and ignore log_std
        action_normalized = torch.tanh(mean)    # shape [1, act_dim], range (-1, 1)

        # Rescale to the original action space bounds if provided
        # formula: action = low + (action_normalized + 1) / 2 * (high - low)
        low  = torch.tensor(action_space_bounds[0], dtype=torch.float32)
        high = torch.tensor(action_space_bounds[1], dtype=torch.float32)
        action = low + (action_normalized + 1.0) / 2.0 * (high - low)

        return action.squeeze(0).numpy()    # shape (act_dim,), e.g. [thrust, angle]
    
    elif action_space_type == "discrete":
        # Deterministic: pick the action with the highest logit
        action_index = torch.argmax(output_logits, dim=-1)
        return int(action_index.item())     # a single integer, e.g. 0, 1, 2, ...