import os
import csv
import yaml
import ray
import torch
import numpy as np
import gymnasium as gym
import tree

from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core.columns import Columns

from trainrl.utils import _load_env_class


def _obs_to_tensor(obs) -> torch.Tensor:
    """Convert a single environment observation to a float32 torch batch."""
    if isinstance(obs, dict):
        obs = obs.get("obs", obs)

    obs_np = np.asarray(obs, dtype=np.float32)
    return torch.from_numpy(obs_np).unsqueeze(0)


def _add_batch_dim_to_state(state):
    """
    RLModule.get_initial_state() returns an unbatched recurrent state.

    RLModule.forward_inference() expects the recurrent state to include
    batch dimension B=1. This function converts [H] to [1, H].
    """
    if not state:
        return state

    return tree.map_structure(
        lambda value: (
            value.unsqueeze(0)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value).unsqueeze(0)
        ),
        state,
    )


def _to_numpy_single_action(action_tensor: torch.Tensor, is_recurrent: bool) -> np.ndarray:
    """
    Convert one RLModule action output to an unbatched NumPy action.

    Expected shapes:
      Feed-forward: [B, action_dim]
      Recurrent:    [B, T, action_dim]

    This evaluator always uses B=1 and, for recurrent modules, T=1.
    """
    action_tensor = action_tensor.detach().cpu()

    if is_recurrent:
        # Expected shape: [1, 1, action_dim].
        action = action_tensor[0, 0]
    else:
        # Expected shape: [1, action_dim].
        action = action_tensor[0]

    return action.numpy()


def _infer_action(
    rl_module,
    obs,
    state,
    is_recurrent: bool,
    explore: bool,
):
    """
    Compute one action from a new-stack RLModule.

    Uses the module's action output if present. Otherwise converts
    ACTION_DIST_INPUTS into the RLModule's inference action distribution.

    Returns:
        action: NumPy environment action, without B/T dimensions.
        next_state: recurrent state to pass at the next environment step.
    """
    obs_tensor = _obs_to_tensor(obs)

    batch = {Columns.OBS: obs_tensor}

    if is_recurrent:
        # Recurrent module contract: [B=1, T=1, obs_dim].
        batch[Columns.OBS] = obs_tensor.unsqueeze(1)
        batch[Columns.STATE_IN] = state
        batch[Columns.SEQ_LENS] = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        fwd_outputs = rl_module.forward_inference(batch)

    next_state = fwd_outputs.get(Columns.STATE_OUT, state)

    # Best case: the RLModule itself has already generated actions.
    if Columns.ACTIONS in fwd_outputs:
        action_tensor = fwd_outputs[Columns.ACTIONS]

    else:
        # Standard PPO path: forward inference returns distribution inputs,
        # which are NOT environment actions. Convert them through the exact
        # inference distribution class declared by this RLModule.
        dist_inputs = fwd_outputs[Columns.ACTION_DIST_INPUTS]

        dist_cls = rl_module.get_inference_action_dist_cls()
        action_dist = dist_cls.from_logits(dist_inputs)

        if explore:
            action_tensor = action_dist.sample()
        else:
            # This matches deterministic evaluation semantics. For a diagonal
            # Gaussian, it chooses the distribution mean.
            action_tensor = action_dist.to_deterministic().sample()

    action = _to_numpy_single_action(action_tensor, is_recurrent)
    return action, next_state


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    num_episodes: int = 10,
    explore: bool = False,
    base_seed: int | None = None,
):
    """
    Evaluate a checkpoint using RLlib's new RLModule API stack.

    Key properties:
    - Restores the full PPO Algorithm checkpoint.
    - Uses RLModule.forward_inference().
    - Converts action-distribution inputs through the module's actual
      inference action distribution.
    - Preserves LSTM state across timesteps.
    - Resets LSTM state at each episode boundary.
    - Uses the same custom environment class and YAML env_config as training.
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    config_path = os.path.abspath(config_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    os.makedirs(output_dir, exist_ok=True)

    step_log_path = os.path.join(output_dir, "episode_step_data.csv")
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    env_path = config_dict.get("env")
    if not env_path:
        raise ValueError("The YAML configuration has no `env` entry.")

    env_config = config_dict.get("env_config", {}) or {}

    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    env_cls = _load_env_class(env_path)
    env_name = env_cls.__name__

    tune.register_env(
        env_name,
        lambda worker_config, cls=env_cls: cls(worker_config),
    )

    algo = None
    env = None

    try:
        print(f"\nRestoring checkpoint:\n  {checkpoint_path}")

        algo = PPO.from_checkpoint(checkpoint_path)
        rl_module = algo.get_module("default_policy")

        env = env_cls(config=env_config)

        if not isinstance(env.action_space, (gym.spaces.Box, gym.spaces.Discrete)):
            raise TypeError(
                "This evaluator currently supports Gymnasium Box and Discrete "
                f"action spaces. Received: {env.action_space!r}"
            )

        initial_state = rl_module.get_initial_state()
        is_recurrent = bool(initial_state)

        print("\nEvaluation setup")
        print(f"  Environment: {env_cls.__module__}.{env_cls.__name__}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Recurrent module: {is_recurrent}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Explore: {explore}")
        print(f"  Base seed: {base_seed}")

        returns = []
        lengths = []
        truncated_count = 0
        terminated_count = 0

        with open(step_log_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = None
            info_keys = None

            for ep in range(num_episodes):
                episode_seed = None if base_seed is None else base_seed + ep

                if episode_seed is None:
                    obs, reset_info = env.reset()
                else:
                    obs, reset_info = env.reset(seed=episode_seed)

                obs = np.asarray(obs, dtype=np.float32)

                if not env.observation_space.contains(obs):
                    raise ValueError(
                        f"Reset observation is outside observation_space.\n"
                        f"Episode: {ep}\n"
                        f"Observation: {obs}"
                    )

                # Reset LSTM state at the beginning of this episode.
                state = _add_batch_dim_to_state(rl_module.get_initial_state())

                terminated = False
                truncated = False
                ep_return = 0.0
                step = 0
                last_info = reset_info if isinstance(reset_info, dict) else {}

                while not (terminated or truncated):
                    action, state = _infer_action(
                        rl_module=rl_module,
                        obs=obs,
                        state=state,
                        is_recurrent=is_recurrent,
                        explore=explore,
                    )

                    action = np.asarray(action, dtype=env.action_space.dtype)

                    if action.shape != env.action_space.shape:
                        raise ValueError(
                            f"Invalid action shape at episode {ep}, step {step}: "
                            f"received {action.shape}; expected "
                            f"{env.action_space.shape}."
                        )

                    if not np.all(np.isfinite(action)):
                        raise FloatingPointError(
                            f"Non-finite action at episode {ep}, step {step}: "
                            f"{action}"
                        )

                    if isinstance(env.action_space, gym.spaces.Box):
                        # Clip only as a numerical safety guard. This should usually
                        # do nothing when action-distribution behavior is correct.
                        action = np.clip(
                            action,
                            env.action_space.low,
                            env.action_space.high,
                        ).astype(env.action_space.dtype, copy=False)
                    else:
                        # Discrete: pass the action index as a plain int.
                        action = int(action)

                    next_obs, reward, terminated, truncated, info = env.step(action)

                    next_obs = np.asarray(next_obs, dtype=np.float32)
                    reward = float(reward)
                    info = info if isinstance(info, dict) else {}
                    last_info = info

                    if not np.all(np.isfinite(next_obs)):
                        raise FloatingPointError(
                            f"Non-finite observation at episode {ep}, step {step}: "
                            f"{next_obs}"
                        )

                    if not np.isfinite(reward):
                        raise FloatingPointError(
                            f"Non-finite reward at episode {ep}, step {step}: "
                            f"{reward}"
                        )

                    if info_keys is None:
                        info_keys = sorted(info.keys())
                        writer = csv.DictWriter(
                            csv_file,
                            fieldnames=[
                                "episode",
                                "seed",
                                "step",
                                "reward",
                                "episode_return_so_far",
                                "terminated",
                                "truncated",
                                "action_mean_abs",
                                "action_max_abs",
                            ] + info_keys,
                        )
                        writer.writeheader()

                    row = {
                        "episode": ep,
                        "seed": episode_seed,
                        "step": step,
                        "reward": reward,
                        "episode_return_so_far": ep_return + reward,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "action_mean_abs": float(np.mean(np.abs(action))),
                        "action_max_abs": float(np.max(np.abs(action))),
                    }

                    for key in info_keys:
                        row[key] = info.get(key, "")

                    writer.writerow(row)

                    obs = next_obs
                    ep_return += reward
                    step += 1

                returns.append(ep_return)
                lengths.append(step)

                terminated_count += int(terminated)
                truncated_count += int(truncated)

                print(
                    f"Episode {ep + 1:>{len(str(num_episodes))}}/{num_episodes} | "
                    f"Return: {ep_return:>10.6f} | "
                    f"Steps: {step:>3} | "
                )

        returns_np = np.asarray(returns, dtype=np.float64)
        lengths_np = np.asarray(lengths, dtype=np.float64)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Episodes: {num_episodes}\n")
            f.write(f"Explore: {explore}\n")
            f.write(f"Base seed: {base_seed}\n")
            f.write(f"Recurrent module: {is_recurrent}\n\n")

            f.write("Episode return statistics\n")
            f.write(f"  mean: {returns_np.mean():.12g}\n")
            f.write(f"  std:  {returns_np.std(ddof=0):.12g}\n")
            f.write(f"  min:  {returns_np.min():.12g}\n")
            f.write(f"  max:  {returns_np.max():.12g}\n\n")

            f.write("Episode-length statistics\n")
            f.write(f"  mean: {lengths_np.mean():.12g}\n")
            f.write(f"  min:  {lengths_np.min():.12g}\n")
            f.write(f"  max:  {lengths_np.max():.12g}\n\n")

            f.write("Terminal statistics\n")
            f.write(f"  terminated: {terminated_count}\n")
            f.write(f"  truncated: {truncated_count}\n")

        print("\nEvaluation complete")
        print(f"  Mean return: {returns_np.mean():.8f}")
        print(f"  Std return:  {returns_np.std(ddof=0):.8f}")
        print(f"  Mean length: {lengths_np.mean():.2f}")
        print(f"  Step log:     {step_log_path}")
        print(f"  Summary:      {summary_path}")

        return {
            "mean_return": float(returns_np.mean()),
            "std_return": float(returns_np.std(ddof=0)),
            "min_return": float(returns_np.min()),
            "max_return": float(returns_np.max()),
            "mean_episode_length": float(lengths_np.mean()),
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "step_log_path": step_log_path,
            "summary_path": summary_path,
        }

    finally:
        if env is not None:
            env.close()

        if algo is not None:
            algo.stop()


if __name__ == "__main__":
    CHECKPOINT_PATH = (
        "YOUR_TRIAL_DIRECTORY/"
        "checkpoint_000000"
    )

    CONFIG_PATH = os.path.join(
        os.path.dirname(CHECKPOINT_PATH),
        "config.yaml",
    )

    OUTPUT_DIR = os.path.join(
        os.path.dirname(CHECKPOINT_PATH),
        "final_evaluation_results",
    )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    final_cfg = cfg.get("final_eval_config", {}) or {}

    evaluate_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
        output_dir=OUTPUT_DIR,
        num_episodes=final_cfg.get("num_episodes", 100),
        explore=final_cfg.get("explore", False),
        # Keep this fixed while validating parity with RLlib evaluation.
        base_seed=10_000,
    )