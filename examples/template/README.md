# Template: start your own project

A blank `trainrl` project. Copy this folder, fill in the environment, and train.

```
template/
├── envs/sample_env.py   ← fill in the TODOs
├── config.yaml          ← points to SampleEnv; put your env parameters under env_config
├── train.py             ← python train.py config.yaml
├── evaluate.py          ← python evaluate.py results/<run_name>/<trial_folder>/best_checkpoint
├── tune.py              ← python tune.py config.yaml   (set your hardware limits inside)
└── deploy_sample.py     ← run the trained policy without RLlib
```

## Steps

1. **Fill in [`envs/sample_env.py`](envs/sample_env.py)**, following the TODOs:
   - `__init__(self, config)`: read your parameters from `config` (this is `env_config` from the YAML), then define `self.observation_space` and `self.action_space` as `float32` `Box` spaces.
   - `reset()`: initialize the state and return `(obs, info)`.
   - `step(action)`: apply the action, update the state, compute the reward, and return `(obs, reward, terminated, truncated, info)`.
   - `_get_info()`: return a dict of scalars. Every key ends up in the evaluation CSV and can be plotted in TensorBoard through `metrics_to_log`.

   For a filled-in version, see [`../lander2d/envs/lander_env.py`](../lander2d/envs/lander_env.py).

2. **Test the env with random actions** before training. A short loop like the one at the bottom of `lander_env.py` is enough.

3. **Edit [`config.yaml`](config.yaml):** set `run_name`, the `env_config` parameters and `metrics_to_log`. The resource settings (`env_runners_config`, `resources_config`, `learner_config`) target a 32-core, 1-GPU workstation. Run `python tune.py config.yaml` to get values for your machine.

4. **Train, evaluate and deploy:**

   ```bash
   python train.py config.yaml
   python evaluate.py results/my_experiment/<trial_folder>/best_checkpoint
   ```

   In `deploy_sample.py`, set `CHECKPOINT_DIR` and the network sizes to match `rl_module_config`, then replace `raw_obs = ...` with your sensor or simulator reading.

Run every command from inside this folder, so that `envs.sample_env` can be imported.
