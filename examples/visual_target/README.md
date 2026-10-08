# Example: Visual Target (CNN, ViT)

The policy sees **only an image**: a 64×64 RGB frame with a red pixel (the agent) and a green pixel (the target at the center). It must learn to move the agent onto the target with 8 discrete moves. This example shows image observations, discrete actions, and a custom Vision Transformer.

| Config | Network | Type |
|---|---|---|
| [`configs/config_cnn.yaml`](configs/config_cnn.yaml) | CNN | RLlib built-in |
| [`configs/config_vit.yaml`](configs/config_vit.yaml) | Vision Transformer (beta) | **Custom RLModule** ([`modules/`](modules/)) |

```
visual_target/
├── envs/visual_target.py             ← the Gymnasium environment
├── modules/
│   ├── vit_policy_network.py         ← plain PyTorch ViT
│   ├── vit_rl_module.py              ← RLlib wrapper around it
│   └── test_vit_network.py           ← shape/gradient check without RLlib
├── configs/                          ← one YAML per experiment
├── train.py  evaluate.py  tune.py    ← trainrl entry points
└── deployment/
    ├── architecture_check.py         ← list the weights stored in a checkpoint
    └── deploy_cnn.py  deploy_vit.py
```

Run every command from inside `examples/visual_target/`.

## 1. The environment

[`envs/visual_target.py`](envs/visual_target.py) defines `VisualTargetEnv`:

| | |
|---|---|
| Observation | `(64, 64, 3)` float image in `[0, 1]` |
| Action | `Discrete(8)`: left, right, up, down, and the four diagonals |
| Reward | `-0.001` per step, **+100** for reaching the target, **−100** for leaving the image |

Test it with random actions:

```bash
python -m envs.visual_target
```

## 2. Train

```bash
python train.py configs/config_cnn.yaml
python train.py configs/config_vit.yaml
```

> The configs target a **32-core, 1-GPU workstation**. On a different machine, run `python tune.py configs/config_cnn.yaml` first and copy the suggested snippet into the config (see [`tune.py`](tune.py)).

The CNN is configured entirely in YAML. Each `conv_filters` entry is `[num_filters, kernel_size, stride]`, and the comments in the config show how the 64×64 image shrinks layer by layer.

To resume, evaluate, and follow training in TensorBoard, use the same commands as in the [lander example](../lander2d/README.md#2-train).

## 3. Custom RLModule: the Vision Transformer (beta)

> **Beta:** the Transformer-based modules (Transformer and ViT) and their deployment classes are under development. They train and deploy end to end, but their APIs and behavior may change in future releases.

```yaml
rl_module_config:
  module_class: modules.vit_rl_module.ViTRLModule
  model_config:
    patch_size: 16        # 64x64 image → 4x4 = 16 patches
    d_model: 128
    nhead: 4
    num_layers: 2
    dim_feedforward: 256
    dropout: 0.1
```

- **[`vit_policy_network.py`](modules/vit_policy_network.py)**: a plain `nn.Module`. It splits the image into patches, embeds them, prepends a learnable `[CLS]` token, adds learned positional embeddings, runs a `TransformerEncoder`, and reads the policy and value from `[CLS]`. Check it without RLlib:

  ```bash
  python -m modules.test_vit_network
  ```

- **[`vit_rl_module.py`](modules/vit_rl_module.py)**: the RLlib wrapper. It has the same structure as the [lander's Transformer module](../lander2d/README.md#4-custom-rlmodule-the-transformer), plus an image step in `_prepare_obs`: RLlib delivers images as `(B, H, W, C)`, and the network expects PyTorch's `(B, C, H, W)`. The image size and channel count are read from the env's observation space, and discrete or continuous actions are detected automatically.

## 4. Deploy without RLlib

```bash
python deployment/deploy_cnn.py results/visual_target/<trial_folder>/best_checkpoint
python deployment/deploy_vit.py results/ppo_visual_target_vit/<trial_folder>/best_checkpoint   # beta
# Weights loaded successfully.
# Control command: 4
```

Both scripts build an example frame (green target at the center, red agent just below it), then run it through the trained network:

- **CNN:** `preprocess_obs(raw_obs, "cnn", obs_shape=(3, 64, 64))` converts the `(H, W, C)` image to a `[1, C, H, W]` tensor.
- **ViT:** `preprocess_obs(raw_obs, "vit", obs_shape=PATCH_SIZE)` cuts the image into patches exactly as the network does during training.
- For discrete actions, `logits_to_action` returns the index of the best move (0–7).

The constants at the top of each script (`CONV_CONFIGS`, `PATCH_SIZE`, `D_MODEL`, ...) must match the training config. If loading fails, compare them with the output of `python deployment/architecture_check.py <best_checkpoint>`.

> **Security:** checkpoints are loaded with `pickle`. Only load checkpoints you trust.
