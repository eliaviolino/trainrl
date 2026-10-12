# Example: Visual Target (CNN, ViT, temporal ViT)

The policy sees **only an image**: a 64×64 RGB frame with a red pixel (the agent) and a green pixel (the target at the center). It must learn to move the agent onto the target with 8 discrete moves. This example shows image observations, discrete actions, and a custom Vision Transformer, for single frames and for sequences of frames.

| Config | Network | Type |
|---|---|---|
| [`configs/config_cnn.yaml`](configs/config_cnn.yaml) | CNN | RLlib built-in |
| [`configs/config_vit.yaml`](configs/config_vit.yaml) | Vision Transformer (beta) | **Custom RLModule** ([`modules/`](modules/)) |
| [`configs/config_vit_temporal.yaml`](configs/config_vit_temporal.yaml) | Temporal ViT over the last frames (beta), on the harder momentum task | **Custom RLModule** ([`modules/`](modules/)) |

```
visual_target/
├── envs/visual_target.py             ← the Gymnasium environment
├── modules/
│   ├── vit_policy_network.py         ← plain PyTorch ViT (single frame or window of frames)
│   ├── vit_rl_module.py              ← RLlib wrapper around it
│   └── test_vit_network.py           ← shape/gradient check without RLlib
├── configs/                          ← one YAML per experiment
├── train.py  evaluate.py  tune.py    ← trainrl entry points
└── deployment/
    ├── architecture_check.py         ← list the weights stored in a checkpoint
    └── deploy_cnn.py  deploy_vit.py  deploy_vit_temporal.py
```

Run every command from inside `examples/visual_target/`.

## 1. The environment

[`envs/visual_target.py`](envs/visual_target.py) defines `VisualTargetEnv`:

| | |
|---|---|
| Observation | `(64, 64, 3)` float image in `[0, 1]` |
| Action | `Discrete(8)`: left, right, up, down, and the four diagonals |
| Reward | `-0.001` per step, **+100** for reaching the target, **−100** for leaving the image |

Optional `env_config` keys (all off by default, so the CNN and ViT configs are unaffected):

| Key | Effect |
|---|---|
| `momentum: true` | Each action changes the velocity instead of the position (limited to `max_speed` pixels per step, default 2), and the agent must **stop** on the target. The velocity cannot be seen in a single frame, so the policy needs a history of frames. |
| `history_len: N` | The observation becomes the last N frames, shape `(N, 64, 64, 3)`, oldest first. At reset, the window is filled with the first frame. |
| `uint8_obs: true` | Pixels are `uint8` in `[0, 255]` instead of `float32` in `[0, 1]`: 4× less memory per frame. The ViT module scales them back to `[0, 1]`. |

The `info` dict returns `lander_x, lander_y, vel_x, vel_y`.

Test it with random actions:

```bash
python -m envs.visual_target
```

## 2. Train

```bash
python train.py configs/config_cnn.yaml
python train.py configs/config_vit.yaml
python train.py configs/config_vit_temporal.yaml
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

- **[`vit_rl_module.py`](modules/vit_rl_module.py)**: the RLlib wrapper. It has the same structure as the [lander's Transformer module](../lander2d/README.md#4-custom-rlmodule-the-transformer), plus an image step in `_prepare_obs`: RLlib delivers images channels-last, and the network expects PyTorch's channels-first layout. The image size, channel count, window length and pixel format are read from the env's observation space, and discrete or continuous actions are detected automatically.

### Temporal ViT: sequences of frames

A single frame cannot show motion. For tasks where the state includes something you only see over time (velocity, a target that moves, an object that is briefly hidden), set `history_len` in `env_config`. The same module then builds a temporal ViT:

```
window (B, N, H, W, C)
   │  every frame encoded on its own
   ▼
FrameEncoder: the ViT above, one [CLS] embedding per frame → (B, N, d_model)
   │
   ▼
TemporalEncoder: causal Transformer over the N embeddings, read at the newest frame
   │
   ▼
policy and value heads
```

The `FrameEncoder` sees space and the `TemporalEncoder` sees time. The temporal part has the same design as the [lander's Transformer](../lander2d/README.md#4-custom-rlmodule-the-transformer): sinusoidal time encoding, causal mask, and readout from the last token. Its only extra setting is `temporal_layers` in `model_config`. It reuses `nhead`, `dim_feedforward` and `dropout`.

The window length is set only once, in `env_config.history_len`. The module reads it from the observation space. RLlib's `max_seq_len` is not involved: the module is stateless, so RLlib passes the windows through unchanged.

[`config_vit_temporal.yaml`](configs/config_vit_temporal.yaml) trains it on the momentum task with a window of 4 frames. To see why the history matters, train `config_vit.yaml` with `momentum: true` added to its `env_config` and compare the two runs. The momentum task is harder than the default one, so it may need more iterations.

**Memory:** every training sample stores N frames. A 64×64×3 frame is 48 KB as `float32` and 12 KB as `uint8`, so a batch of 10,000 windows of 4 frames takes about 2 GB in float and 0.5 GB in uint8. Keep `uint8_obs: true` and a short window (4–8). Raise them only when the task needs it.

## 4. Deploy without RLlib

```bash
python deployment/deploy_cnn.py results/visual_target/<trial_folder>/best_checkpoint
python deployment/deploy_vit.py results/ppo_visual_target_vit/<trial_folder>/best_checkpoint   # beta
# Weights loaded successfully.
# Control command: 4
python deployment/deploy_vit_temporal.py results/ppo_visual_target_vit_temporal/<trial_folder>/best_checkpoint   # beta
```

The CNN and ViT scripts build an example frame (green target at the center, red agent just below it), then run it through the trained network:

- **CNN:** `preprocess_obs(raw_obs, "cnn", obs_shape=(3, 64, 64))` converts the `(H, W, C)` image to a `[1, C, H, W]` tensor.
- **ViT:** `preprocess_obs(raw_obs, "vit", obs_shape=PATCH_SIZE)` cuts the image into patches exactly as the network does during training.
- **Temporal ViT:** the script keeps a window of the last `HISTORY_LEN` frames and fills it with the first frame at episode start, as the env does. It shows two equivalent loops:
  - **Full window:** pass the `(N, H, W, C)` window to `preprocess_obs(..., "vit")` and run the whole model at every step.
  - **Cached embeddings:** each frame is encoded independently, so embeddings of past frames never change. `model.encode_frame()` encodes only the newest frame, and `model.forward_from_embeddings()` runs the temporal part over the stored embeddings. The ViT then runs on one frame per step instead of N, which is useful on a robot.
- For discrete actions, `logits_to_action` returns the index of the best move (0–7).

The constants at the top of each script (`CONV_CONFIGS`, `PATCH_SIZE`, `D_MODEL`, `HISTORY_LEN`, ...) must match the training config. If loading fails, compare them with the output of `python deployment/architecture_check.py <best_checkpoint>`.

> **Security:** checkpoints are loaded with `pickle`. Only load checkpoints you trust.
