# Example: 2D Lander (MLP, LSTM, Transformer)

A point-mass lander starts at a random position and must fire its 2D thrusters to land on the pad at the origin, with low speed, against gravity. Observations are vectors, actions are continuous.

The example trains the same task with three policy networks:

| Config | Network | Type |
|---|---|---|
| [`configs/config_mlp.yaml`](configs/config_mlp.yaml) | MLP | RLlib built-in |
| [`configs/config_lstm.yaml`](configs/config_lstm.yaml) | MLP + LSTM | RLlib built-in |
| [`configs/config_transformer.yaml`](configs/config_transformer.yaml) | Causal Transformer (beta) | **Custom RLModule** ([`modules/`](modules/)) |

```
lander2d/
├── envs/lander_env.py                ← the Gymnasium environment
├── modules/
│   ├── transformer_network.py        ← plain PyTorch network
│   └── transformer_rl_module.py      ← RLlib wrapper around it
├── configs/                          ← one YAML per experiment
├── train.py  evaluate.py  tune.py    ← trainrl entry points
├── extract_final_results.py          ← load the evaluation CSV per episode
├── plot_lander_results.py            ← plot landing trajectories
└── deployment/
    ├── architecture_check.py         ← list the weights stored in a checkpoint
    └── deploy_mlp.py  deploy_lstm.py  deploy_transformer.py
```

Run every command from inside `examples/lander2d/`.

## 1. The environment

[`envs/lander_env.py`](envs/lander_env.py) defines `Lander2DEnv`:

| | |
|---|---|
| Observation | `[x, y, vx, vy]` |
| Action | `[thrust_x, thrust_y]` in `[-1, 1]` |
| Reward | `-0.01·(|x|+|y|+|vx|+|vy|)` per step, **+100** for landing, **−100** for leaving the area |
| Episode ends | landed (`|x|, |y| < land_pos` and `|vx|, |vy| < land_vel`), out of bounds, or `max_steps` |

All physics constants come from `env_config` in the YAML. The `info` dict returns `x, y, vx, vy`, which `metrics_to_log` sends to TensorBoard and the evaluation writes to its CSV.

Test the env with random actions before training:

```bash
python -m envs.lander_env
```

## 2. Train

```bash
python train.py configs/config_mlp.yaml
python train.py configs/config_lstm.yaml
python train.py configs/config_transformer.yaml
```

> The configs target a **32-core, 1-GPU workstation** (25 env runners, `num_gpus: 1`). On a different machine, run `python tune.py configs/config_mlp.yaml` first. Set your limits (`cpus`, `gpus`, `envs`, ...) at the top of [`tune.py`](tune.py), then copy the suggested snippet from `results_tuning/workstation_hardware_tuning/` into the config.

To resume or extend a run:

```bash
python train.py configs/config_mlp.yaml results/ppo_lander_mlp                # resume a stopped run
python train.py configs/config_mlp.yaml results/ppo_lander_mlp results true   # keep training a finished run
                                                                              # (after raising training_iterations)
```

Follow training live with `tensorboard --logdir results/`. The `x` and `y` metrics appear under `metrics_logged/`.

Training writes:

```
results/ppo_lander_mlp/
├── final_evaluation/                       ← automatic evaluation after training
│   ├── episode_step_data.csv
│   └── evaluation_summary.txt
└── PPO_Lander2DEnv_<id>_<timestamp>/       ← trial folder
    ├── config.yaml                         ← copy of the config used
    ├── best_checkpoint/                    ← best policy by evaluation return
    └── checkpoint_0000NN/                  ← latest checkpoint
```

## 3. Evaluate and plot

```bash
python evaluate.py results/ppo_lander_mlp/<trial_folder>/best_checkpoint
```

This re-runs `final_eval_config.num_episodes` episodes with the policy in `best_checkpoint`. It reads `config.yaml` from the trial folder and writes `episode_step_data.csv` and `evaluation_summary.txt` to `<trial_folder>/final_evaluation_results/`.

```bash
python plot_lander_results.py results/ppo_lander_mlp/final_evaluation/episode_step_data.csv
python extract_final_results.py results/ppo_lander_mlp/final_evaluation/episode_step_data.csv
```

`extract_final_results.load_episode_data()` turns the CSV into one dict of lists per episode (`episodes[0]["x"]`, `episodes[0]["reward"]`, ...), so you can build your own analysis on top of it.

## 4. Custom RLModule: the Transformer (beta)

> **Beta:** the Transformer-based modules (Transformer and ViT) and their deployment classes are under development. They train and deploy end to end, but their APIs and behavior may change in future releases.

To use your own network instead of RLlib's built-ins, the YAML points to an RLModule class:

```yaml
rl_module_config:
  module_class: modules.transformer_rl_module.TransformerRLModule   # importable path
  model_config:            # passed to the module as self.config.model_config_dict
    d_model: 64
    nhead: 4
    num_layers: 2
    dim_feedforward: 128
    max_seq_len: 20
    dropout: 0.0
```

`trainrl` imports the class and hands it to RLlib. No Python changes are needed in `trainrl`. The module has two parts:

- **[`transformer_network.py`](modules/transformer_network.py)**: a plain `nn.Module` (input projection, positional encoding, causal `TransformerEncoder`, policy and value heads). It knows nothing about RLlib, so you can test it on its own.
- **[`transformer_rl_module.py`](modules/transformer_rl_module.py)**: a `TorchRLModule` + `ValueFunctionAPI` wrapper. RLlib requires these pieces:
  - `setup()`: reads the observation and action spaces from the env (continuous or discrete is detected automatically) and the sizes from `model_config`, then builds the network.
  - `_forward_inference` / `_forward_exploration` / `_forward_train`: return `ACTION_DIST_INPUTS` (and `VF_PREDS` during training).
  - `compute_values()`: the value estimates PPO uses for advantages.

Each observation is currently fed as a sequence of length 1 (`_prepare_obs`). The network supports longer histories (`max_seq_len`), and the module is the place to add them.

## 5. Deploy without RLlib

`trainrl.deployment` rebuilds the trained network in plain PyTorch, so you can run the policy on a robot or inside another simulator without Ray or RLlib:

```bash
python deployment/deploy_mlp.py         results/ppo_lander_mlp/<trial_folder>/best_checkpoint
python deployment/deploy_lstm.py        results/ppo_lander_lstm/<trial_folder>/best_checkpoint
python deployment/deploy_transformer.py results/ppo_lander_transformer/<trial_folder>/best_checkpoint   # beta
# Weights loaded successfully.
# Control command: [0.06 0.24]
```

Each script follows the same five steps:

1. **Build the model.** The constants at the top (`HIDDEN_SIZES`, `LSTM_HIDDEN`, `D_MODEL`, ...) must match the training config.
2. **Load the weights** from `best_checkpoint` (`load_weights`). This fails loudly if the architecture doesn't match.
3. **Preprocess** a raw observation (`preprocess_obs`). Replace the example `[x, y, vx, vy]` with your sensor reading.
4. **Run the forward pass** (`run_inference`). For the LSTM, start from `model.get_initial_state()` and pass `hidden_state` back in at every step.
5. **Convert to a control command** (`logits_to_action`), scaled to the action bounds.

If loading fails, run `python deployment/architecture_check.py <best_checkpoint>`. It lists every weight name and shape in the checkpoint, so you can compare them with the deployment model.

> **Security:** checkpoints are loaded with `pickle`. Only load checkpoints you trust.
