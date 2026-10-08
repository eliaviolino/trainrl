import os
import shutil
import re
import importlib
from ray import tune
from ray.rllib.core.rl_module import RLModuleSpec

# Ensure the best checkpoint is preserved: copy it into a stable folder
def _copy_checkpoint(src_path: str, dst_path: str):
    try:
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            label_dir = dst_path
        else:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            label_dir = os.path.dirname(dst_path)

        match = re.search(r"checkpoint[-_]?(\d+)", src_path)
        if match:
            label = f"cp {int(match.group(1))}"
        else:
            label = f"checkpoint path: {src_path}"

        with open(os.path.join(label_dir, "best_checkpoint.txt"), "w") as f:
            f.write(label + "\n")
    except Exception as e:
        print(f"Warning: failed to copy best checkpoint from {src_path} to {dst_path}: {e}")


def _load_env_class(env_path: str):
    """Load an environment class from a python path like `envs.my_env.MyEnv`."""
    if not isinstance(env_path, str):
        raise TypeError("`env` must be a string import path like 'envs.my_env.MyEnv'.")

    module_name, class_name = env_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        env_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Could not import '{class_name}' from module '{module_name}'.") from exc

    return env_cls


def _register_env_from_config(config_dict: dict):
    env_path = config_dict.get("env")
    if env_path is None:
        raise ValueError("Missing 'env' in config. Set it to the Python path of your env class.")

    env_cls = _load_env_class(env_path)
    env_name = env_cls.__name__
    tune.register_env(env_name, lambda config, cls=env_cls: cls(config))
    return env_name


def build_rl_module_config(rl_module_cfg: dict):
    """
    Reads the rl_module_config block from the YAML and returns
    the correct kwargs to unpack into .rl_module().

    Two cases:
      - 'module_class' present → custom RLModule via RLModuleSpec
      - 'module_class' absent  → default RLlib module, just pass model_config
    """
    module_class_path = rl_module_cfg.get("module_class", None)
    model_config = rl_module_cfg.get("model_config", {})

    if module_class_path is not None:
        # Dynamically import the class from a dot-separated path string.
        # e.g. "modules.transformer_rl_module.TransformerRLModule"
        # This way labmates only edit the YAML — no Python changes needed.
        module_path, class_name = module_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        module_class = getattr(module, class_name)

        return {
            "rl_module_spec": RLModuleSpec(
                module_class=module_class,
                model_config=model_config,
            )
        }
    else:
        # Default RLlib module — just forward model_config as-is
        # (fcnet_hiddens, fcnet_activation, use_lstm, etc. all work here)
        return {
            "model_config": model_config
        }