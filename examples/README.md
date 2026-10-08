# Examples

Each folder is a complete, self-contained `trainrl` project: an environment, YAML configs, and scripts to train, evaluate, tune and deploy.

| Example | Observations | Actions | Networks | Shows |
|---|---|---|---|---|
| [**lander2d**](lander2d/) | vector `[x, y, vx, vy]` | continuous | MLP, LSTM, **Transformer** (beta) | built-in and custom RLModules, recurrent deployment, trajectory plots |
| [**visual_target**](visual_target/) | 64×64 RGB image | discrete (8) | CNN, **ViT** (beta) | image observations, custom Vision Transformer, image deployment |
| [**template**](template/) | — | — | MLP / LSTM | an empty `SampleEnv` to fill in for your own problem |

All the examples follow the same workflow:

```
envs/my_env.py ──► configs/*.yaml ──► tune.py ──► train.py ──► evaluate.py ──► deployment/deploy_*.py
  your env         one per experiment  (optional)  trainrl     trainrl         plain PyTorch, no RLlib
```

## Running an example

```bash
git clone https://github.com/eliaviolino/trainrl
cd trainrl
pip install -e .
pip install matplotlib        # only for the plotting scripts

cd examples/lander2d          # run every command from inside the example folder
python train.py configs/config_mlp.yaml
```

The configs are set up for a **32-core, 1-GPU workstation**. On another machine, edit the hardware limits in `tune.py`, run `python tune.py <config>`, and paste the suggested `env_runners_config` / `resources_config` snippet into the config before training.

## Starting your own project

Copy [`template/`](template/), fill in `envs/sample_env.py`, and point `env:` in `config.yaml` at your class. To use your own network, copy a `modules/` folder from one of the examples and set `rl_module_config.module_class`.
