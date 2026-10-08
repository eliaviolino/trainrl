# TrainRL

A lightweight, pip-installable wrapper around [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) for training, evaluating, deploying, and hardware-tuning PPO policies on custom Gymnasium environments.

`trainrl` handles all RLlib/Tune internals — environment registration, algorithm configuration, checkpointing, logging, final evaluation, and workstation efficiency tuning — so you only need to write your environment and a YAML config.

***

## Installation

```bash
pip install git+https://github.com/eliaviolino/trainrl
```

For a pinned version (recommended for reproducibility):

```bash
pip install git+https://github.com/eliaviolino/trainrl@v0.1.0
```

For local development:

```bash
git clone https://github.com/eliaviolino/trainrl
cd trainrl
pip install -e .
```

> **Note:** For CUDA-enabled `torch`, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the correct wheel before installing `trainrl`.

***

## Quickstart

> **New to `trainrl`?** Start from the [examples](examples/): complete, runnable projects for a [2D lander](examples/lander2d/) (MLP, LSTM, custom Transformer) and a [visual target](examples/visual_target/) task (CNN, custom ViT), plus an empty [template](examples/template/) to fill in with your own environment. Please note that the Transformer and ViT modules are in beta.

### Training

```python
from trainrl import train_ppo

train_ppo("configs/config.yaml")
```

All options:

```python
from trainrl import train_ppo

train_ppo(
    cfg_path="configs/config.yaml",
    restore_dir="results/my_run",   # optional: resume the stopped experiment in this folder
    continue_finished=False,        # with restore_dir: keep training finished trials up to the
                                    # (raised) training_iterations in the YAML
    results_dir="results",          # where runs are written
    debug=False,                    # True: print env-runner logs (e.g. errors inside your env) to the console
)
```

After training completes, `train_ppo` automatically runs a final evaluation and saves per-step logs to `results/<run_name>/final_evaluation/`.

### Evaluation

```python
from trainrl import evaluate_checkpoint

evaluate_checkpoint(
    checkpoint_path="results/my_run/PPO_MyEnv_xxxxx/best_checkpoint",
    config_path="results/my_run/PPO_MyEnv_xxxxx/config.yaml",
    output_dir="results/my_run/PPO_MyEnv_xxxxx/my_eval/",
    num_episodes=100,
)
```

This runs inference natively through the trained RLModule (no Tune/RLlib overhead) and writes two files to `output_dir`:

- `episode_step_data.csv`: one row per step, with reward and every scalar in your env's `info` dict.
- `evaluation_summary.txt`: return and episode-length statistics.

Optional arguments: `explore=True` samples actions instead of taking the deterministic action, and `base_seed=<int>` makes the episodes reproducible.

`evaluate_checkpoint` supports continuous (`Box`) and discrete (`Discrete`) action spaces.

### Workstation Efficiency Tuning

Before launching a long training run, use `tune_workstation_efficiency` to find the fastest resource configuration for your hardware:

```python
from trainrl import tune_workstation_efficiency

tune_workstation_efficiency(
    cfg_path="configs/config.yaml",
    cpus=32,
    gpus=1,
    envs=100,
    min_w=1,
    max_w=25,
    stop_iters=5,
)
```

This systematically tries different combinations of `num_env_runners`, `num_envs_per_env_runner`, and CPU/GPU allocation, then writes a summary file with the best candidate and a ready-to-copy config snippet.

> **Important:** `tune_workstation_efficiency` initializes Ray with explicit CPU/GPU limits and must not be called in the same Python process as `train_ppo` without shutting Ray down in between. Run it as a separate script before training.

### Deployment

Deploy a trained policy without any RLlib/Ray dependency at inference time:

```python
from trainrl.deployment import (
    DeploymentPolicyMLP,
    load_weights,
    preprocess_obs,
    run_inference,
    logits_to_action,
)
import numpy as np

# Step 1: Build the model matching your training architecture
model = DeploymentPolicyMLP(
    obs_dim=4,
    act_dim=2,
    hidden_sizes=[256, 256],
    action_space_type="continuous",
    activation="tanh",
)

# Step 2: Load trained weights from checkpoint
model = load_weights(model, "results/my_run/.../best_checkpoint")

# Step 3: Run inference loop
hidden_state = None  # only used for LSTM
raw_obs = env.reset()[0]

obs_tensor = preprocess_obs(raw_obs, arch_type="mlp")
logits, hidden_state = run_inference(model, obs_tensor, arch_type="mlp", hidden_state=hidden_state)
action = logits_to_action(logits, "continuous", (np.array([-1.0, -1.0]), np.array([1.0, 1.0])))
```

***

## Configuration

All training behavior is controlled by a YAML config file. Create one in your project and pass its path to `train_ppo`.

### Minimal example

```yaml
env: "envs.my_env.MyEnv"        # importable Python path to your Gymnasium env class

run_name: "my_experiment"
training_iterations: 200

env_config:
  max_steps: 500

training_config:
  lr: 3.0e-4
  gamma: 0.99
  train_batch_size_per_learner: 4000

env_runners_config:
  num_env_runners: 4

eval_config:
  evaluation_interval: 10
  evaluation_duration: 5

final_eval_config:
  num_episodes: 100
```

### Full config reference

| Key | Type | Description |
|-----|------|-------------|
| `env` | `str` | Importable path to your env class, e.g. `envs.my_env.MyEnv` |
| `run_name` | `str` | Name for the Tune trial folder under `results/` |
| `training_iterations` | `int` | Number of PPO training iterations |
| `env_config` | `dict` | Passed directly to your env constructor |
| `training_config` | `dict` | PPO hyperparameters (`lr`, `gamma`, `clip_param`, etc.) |
| `env_runners_config` | `dict` | RLlib env runner settings (`num_env_runners`, etc.) |
| `resources_config` | `dict` | Ray resource allocation (`num_gpus`, etc.) |
| `learner_config` | `dict` | RLlib learner settings (`num_learners`, etc.) |
| `eval_config` | `dict` | RLlib evaluation settings (`evaluation_interval`, etc.) |
| `final_eval_config.num_episodes` | `int` | Episodes to run in the post-training evaluation |
| `rl_module_config` | `dict` | Neural network architecture (see below) |
| `metrics_to_log` | `list[str]` | Info-dict keys to log as custom TensorBoard metrics |
| `checkpoint_at_end_of_training` | `bool` | Save a checkpoint at the final iteration (default: `true`) |

***

## Neural Network Configuration

`trainrl` supports three ways to configure the policy network, all via `rl_module_config` in the YAML.

### Default MLP (RLlib built-in)

```yaml
rl_module_config:
  model_config:
    fcnet_hiddens: [256, 256]
    fcnet_activation: relu
```

### LSTM (RLlib built-in)

```yaml
rl_module_config:
  model_config:
    fcnet_hiddens: [256, 256]
    use_lstm: true
    lstm_cell_size: 256
```

### CNN (RLlib built-in, for image observations)

```yaml
rl_module_config:
  model_config:
    conv_filters:
      - [32, [8, 8], 4]
      - [64, [4, 4], 2]
      - [64, [3, 3], 1]
    conv_activation: relu
    fcnet_hiddens: [512]
```

### Custom RLModule

Point to any custom `RLModule` subclass via its importable path:

```yaml
rl_module_config:
  module_class: "modules.my_module.MyRLModule"
  model_config:               # available inside the module as self.config.model_config_dict
    hidden_dim: 128
    num_heads: 4
```

`trainrl` imports the class and passes it to RLlib, so switching networks is a YAML change.

> **Beta:** the Transformer-based modules (Transformer and ViT) and their deployment classes are under development. They train and deploy end to end, but their APIs and behavior may change in future releases.

For complete, working modules, see the [Transformer](examples/lander2d/modules/) (vector observations) and the [Vision Transformer](examples/visual_target/modules/) (images) in the examples. Each one pairs a plain PyTorch network with a small `TorchRLModule` wrapper, and has a matching `trainrl.deployment` class for running it without RLlib.

***

## Implementing Your Environment

Your environment must follow the [Gymnasium API](https://gymnasium.farama.org/). The only requirement from `trainrl`'s side is that its importable path is set in the config under `env`.

```python
# envs/my_env.py
import gymnasium as gym
import numpy as np

class MyEnv(gym.Env):

    def __init__(self, config=None):
        config = config or {}
        self.max_steps = config.get("max_steps", 500)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self.steps += 1
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = self.steps >= self.max_steps
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
```

Set the config to point to it:

```yaml
env: "envs.my_env.MyEnv"
```

`trainrl` will import the class dynamically and register it with Ray Tune automatically.

> **Tip:** Always define your observation and action spaces with `dtype=np.float32` explicitly. Using `float64` will produce Gymnasium warnings and may cause subtle training issues.

***

## Custom Metrics

To log environment metrics to TensorBoard during training, add a `metrics_to_log` key to your config listing the info-dict keys your environment returns:

```yaml
metrics_to_log:
  - "x"
  - "y"
  - "vx"
```

Your environment must return these keys in the `info` dict from `step()`. Values must be scalar (`int`, `float`, or `bool`). Metrics are aggregated with a rolling mean and appear in TensorBoard under **Scalars → metrics_logged/**.

Launch TensorBoard to monitor training in real-time:

```bash
tensorboard --logdir results/
```

***

## Workstation Efficiency Tuning

`tune_workstation_efficiency` searches for the fastest resource configuration for your specific hardware before you commit to a full training run. It tries combinations of env runners, envs per runner, and CPU/GPU allocation, measuring iteration time for each.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfg_path` | `str` | Path to your YAML config (same one used for training) |
| `cpus` | `int` | Total CPU cores available to Ray |
| `gpus` | `int` | Total GPUs available to Ray (`0` for CPU-only) |
| `envs` | `int` | Desired total number of environments across all runners |
| `min_w` | `int` | Minimum number of env runners to try |
| `max_w` | `int` | Maximum number of env runners to try |
| `stop_iters` | `int` | Training iterations per trial (default: `5`) — keep low |
| `results_root` | `str` | Output folder for summary files (default: `results_tuning/workstation_hardware_tuning`) |

### Output

Results are written to `results_tuning/workstation_hardware_tuning/workstation_efficiency_<timestamp>.txt`. The file includes:

- A table of all candidates ranked by average iteration time
- The best candidate highlighted separately
- A ready-to-copy config snippet:

```
Recommended config snippet for best candidate:
env_runners_config:
  batch_mode: complete_episodes
  num_env_runners: 8
  num_envs_per_env_runner: 12
  num_cpus_per_env_runner: 3
  num_gpus_per_env_runner: 0.0
resources_config:
  num_cpus_for_main_process: 4
  num_gpus: 1
eval_config:
  evaluation_parallel_to_training: true
  evaluation_num_env_runners: 8
  evaluation_duration: 8
```

Copy this snippet directly into your `config.yaml` before running `train_ppo`.

***

## Deployment

`trainrl.deployment` provides standalone PyTorch policy classes and helper functions for running trained policies outside of RLlib — on a robot, in a simulator, or in any inference loop. No Ray or RLlib dependency is required at inference time once weights are loaded.

### Supported Architectures

| Class | `arch_type` | Use case |
|-------|-------------|----------|
| `DeploymentPolicyMLP` | `"mlp"` | Flat vector observations |
| `DeploymentPolicyLSTM` | `"lstm"` | Flat vector observations with recurrent memory |
| `DeploymentPolicyCNN` | `"cnn"` | Image observations |
| `DeploymentPolicyTransformer` | `"transformer"` | Sequence of flat observations (beta) |
| `DeploymentPolicyViT` | `"vit"` | Image observations with Vision Transformer (beta) |

### Deployment Steps

**1. Build the model** — parameters must exactly match the `rl_module_config` used during training:

```python
from trainrl.deployment import DeploymentPolicyLSTM

model = DeploymentPolicyLSTM(
    obs_dim=8,
    act_dim=2,
    mlp_hidden_sizes=[256, 256],
    lstm_hidden_size=256,
    lstm_num_layers=1,
    action_space_type="continuous",
    activation="tanh",
)
```

**2. Load weights** from a `best_checkpoint` folder:

```python
from trainrl.deployment import load_weights

model = load_weights(model, "results/my_run/.../best_checkpoint")
```

**3. Preprocess observations** — converts raw numpy arrays to the tensor shape each architecture expects:

```python
from trainrl.deployment import preprocess_obs

obs_tensor = preprocess_obs(raw_obs, arch_type="lstm")
# MLP:         [1, obs_dim]
# LSTM:        [1, 1, obs_dim]
# CNN:         [1, C, H, W]  (normalized to [0, 1])
# Transformer: [1, seq_len, obs_dim]
# ViT:         [1, num_patches, patch_dim]
```

**4. Run inference:**

```python
from trainrl.deployment import run_inference

hidden_state = model.get_initial_state()  # LSTM only; None for all others
logits, hidden_state = run_inference(model, obs_tensor, arch_type="lstm", hidden_state=hidden_state)
```

**5. Convert logits to action:**

```python
from trainrl.deployment import logits_to_action
import numpy as np

action = logits_to_action(
    logits,
    action_space_type="continuous",
    action_space_bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
)
```

For discrete action spaces, `action_space_bounds` is not required — `logits_to_action` returns the index of the highest-scoring action.

***

## Checkpointing

`trainrl` manages checkpoints automatically:

- One checkpoint is saved per iteration; only the most recent is kept on disk.
- The best checkpoint (by evaluation return) is copied to a stable `best_checkpoint/` folder inside the trial directory so it is never overwritten by rotation.
- A `best_checkpoint.txt` file records the checkpoint iteration number for easy reference.
- A copy of the config YAML is saved to the trial folder on the first result, so each run is fully self-contained.

To resume a stopped run, pass the experiment folder (`results/<run_name>`) as `restore_dir`:

```python
train_ppo("configs/config.yaml", restore_dir="results/my_experiment")
```

To keep training a run that already finished, raise `training_iterations` in the YAML and add `continue_finished=True`.

***

## Output Structure

After training, results are written under `results/<run_name>/` (or `<results_dir>/<run_name>/`):

```
results/
└── my_experiment/
    ├── final_evaluation/                  ← post-training evaluation
    │   ├── episode_step_data.csv          ← per-step data
    │   └── evaluation_summary.txt         ← return / length statistics
    └── PPO_MyEnv_xxxxx_00000_0_<timestamp>/
        ├── config.yaml                    ← copy of your config
        ├── best_checkpoint/               ← best checkpoint by eval return
        │   └── best_checkpoint.txt        ← records the checkpoint iteration
        └── checkpoint_000NNN/             ← latest rolling checkpoint
```

If a `run_name` already exists under `results/`, a numeric suffix is appended automatically (`my_experiment_01`, `my_experiment_02`, etc.) to avoid overwriting previous runs.

Workstation tuning results are written separately under:

```
results_tuning/
└── workstation_hardware_tuning/
    ├── eff_01_4r_25e_7.00c/          ← per-trial Ray output
    ├── eff_02_8r_12e_3.00c/
    └── workstation_efficiency_<timestamp>.txt  ← summary + best candidate snippet
```

***

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.10 |
| `ray[rllib]` | ≥ 2.4.0 |
| `torch` | ≥ 1.12.0 |
| `numpy` | ≥ 1.23.0 |
| `gymnasium` | ≥ 0.28.1 |
| `PyYAML` | ≥ 6.0 |
| `tensorboard` | ≥ 2.12.0 |
| `dm-tree` | latest |
| `sympy` | latest |

***

## Examples

The [`examples/`](examples/) folder contains complete projects you can run and copy:

| Example | Observations | Actions | Networks |
|---|---|---|---|
| [lander2d](examples/lander2d/) | vector | continuous | MLP, LSTM, custom Transformer (beta) |
| [visual_target](examples/visual_target/) | 64×64 image | discrete | CNN, custom Vision Transformer (beta) |
| [template](examples/template/) | — | — | empty `SampleEnv` to fill in |

Each one includes the environment, YAML configs, `train.py` / `evaluate.py` / `tune.py`, and deployment scripts that run the trained policy without RLlib.

***

## Citation

If you use `trainrl` in your research, please cite it:

```bibtex
@software{violino_trainrl_2026,
  author  = {Violino, Elia},
  title   = {trainrl: a lightweight RLlib PPO training, evaluation and deployment wrapper},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/eliaviolino/trainrl}
}
```


***

## License

Copyright © 2026 Elia Violino. Released under the MIT License — see [LICENSE](LICENSE) for full terms.