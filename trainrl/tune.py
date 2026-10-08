import copy
import importlib
import os
import time
from datetime import datetime
from pathlib import Path
from sympy import divisors

import ray
import yaml
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import RunConfig, TuneConfig
from trainrl.utils import _load_env_class, _register_env_from_config


class SimpleProgressCallback(tune.callback.Callback):
    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        self._last_print = 0.0

    def on_trial_result(self, iteration, trials, trial, result, **info):
        now = time.time()
        if now - self._last_print >= self.interval_s:
            elapsed = result.get("time_total_s", 0.0)
            it = result.get("training_iteration", 0)
            speed = result.get("time_this_iter_s", float("nan"))
            reward = float("nan")
            env_runners = result.get("env_runners")
            if isinstance(env_runners, dict):
                reward = env_runners.get("episode_return_mean", reward)
            reward = result.get("env_runners/episode_return_mean", reward)
            reward = result.get("episode_reward_mean", reward)
            reward = result.get("episode_return_mean", reward)
            print(
                f"Trial {trial.trial_id} | iter={it:>2} | time_this_iter_s={speed:>5.2f} | "
                f"time_total_s={elapsed:>5.1f} | reward_mean={reward:>7.2f}"
            )
            self._last_print = now


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _deep_copy_dict(d: dict) -> dict:
    return copy.deepcopy(d)


def _format_candidate(candidate: dict) -> str:
    return (
        f"num_env_runners={candidate['num_env_runners']}, "
        f"num_envs_per_env_runner={candidate['num_envs_per_env_runner']}, "
        f"num_cpus_per_env_runner={candidate['num_cpus_per_env_runner']:.2f}, "
        f"num_gpus_per_env_runner={candidate['num_gpus_per_env_runner']:.3f}, "
        f"num_cpus_for_main_process={candidate['num_cpus_for_main_process']}, "
        # f"train_batch_size={candidate['train_batch_size']}, "
        # f"minibatch_size={candidate['minibatch_size']}"
    )


def _build_candidate_grid(cpus: int, gpus: int, envs: int, min_w: int, max_w: int) -> list[dict]:
    # Build a small set of representative resource configurations.
    # Each candidate is a valid Ray RLlib env_runners/resources setup.
    # We test CPU-only and GPU-driver variants without changing the model.

    if min_w < 1:
        raise ValueError("min_w must be at least 1")
    if max_w < min_w:
        raise ValueError("max_w must be >= min_w")

    candidate_grid: list[dict] = []

    # Check GPU
    if gpus > 0:
        hardware = ["cpu_only", "gpu_driver_cpu_envs"]
    else:
        hardware = ["cpu_only"]

    for hw in hardware:
        for w in divisors(envs):
            if w < min_w or w > max_w:
                continue
            eval_workers = w  # for simplicity, we match eval workers to env runners
            eval_episodes = eval_workers
            total_w = w + eval_workers

            if hw == "cpu_only":
                cpus_per_runner = (cpus - 1.)/ total_w if (cpus - 1.)/total_w < 1 else int((cpus - 1.)/total_w) # leave 1 CPU for Ray internals, divide the rest
                cpus_for_main_process = int(cpus - cpus_per_runner * total_w)
                gpus_per_runner = 0.0
                resources_num_gpus = 0
            elif hw == "gpu_driver_cpu_envs":
                cpus_per_runner = (cpus - 1.)/ total_w if (cpus - 1.)/total_w < 1 else int((cpus - 1.)/total_w) # divide all CPUs among env runners and eval workers
                cpus_for_main_process = int(cpus - cpus_per_runner * total_w)
                gpus_per_runner = 0.0
                resources_num_gpus = gpus  # assign all GPUs to the driver (learners) for simplicity
            else:
                raise ValueError(f"Unknown hardware configuration: {hw}")
            
            total_worker_cpus = w * cpus_per_runner
            total_eval_cpus = w * cpus_per_runner
            total_cpus_requested = total_worker_cpus + total_eval_cpus + cpus_for_main_process

            if total_cpus_requested > cpus:
                print(f"Skipping candidate with {w} env runners due to CPU overallocation: "
                      f"total_cpus_requested={total_cpus_requested} exceeds available cpus={cpus}.")
                continue

            candidate = {
                "hardware_architecture": hw,
                "num_env_runners": w,
                "num_envs_per_env_runner": envs // w,
                "num_cpus_per_env_runner": cpus_per_runner,
                "num_gpus_per_env_runner": gpus_per_runner,
                "num_cpus_for_main_process": cpus_for_main_process,
                "resources_num_gpus": resources_num_gpus,
                "eval_workers": eval_workers,
                "eval_episodes": eval_episodes,
            }
            candidate_grid.append(candidate)

    return candidate_grid


def _build_and_run_trial(base_config: dict, cfg_path: str, candidate: dict, trial_id: int, root_dir: Path, stop_iters: int, env_name: str) -> dict:
    # Copy the base YAML config and override only the tuning-specific fields.
    config_dict = _deep_copy_dict(base_config)

    env_runners = _deep_copy_dict(config_dict.get("env_runners_config", {}))
    env_runners.update(
        {
            "num_env_runners": candidate["num_env_runners"],
            "num_cpus_per_env_runner": candidate["num_cpus_per_env_runner"],
            "num_envs_per_env_runner": candidate["num_envs_per_env_runner"],
            "num_gpus_per_env_runner": candidate["num_gpus_per_env_runner"],
        }
    )
    config_dict["env_runners_config"] = env_runners

    resources = _deep_copy_dict(config_dict.get("resources_config", {}))
    resources["num_cpus_for_main_process"] = candidate["num_cpus_for_main_process"]
    resources["num_gpus"] = candidate.get("resources_num_gpus", resources.get("num_gpus", 0))
    config_dict["resources_config"] = resources

    training = _deep_copy_dict(config_dict.get("training_config", {}))
    # training["train_batch_size"] = (
    #     candidate["train_batch_size"]
    #     if candidate.get("train_batch_size") is not None
    #     else training.get("train_batch_size", 10000)
    # )
    # training["minibatch_size"] = (
    #     candidate["minibatch_size"]
    #     if candidate.get("minibatch_size") is not None
    #     else training.get("minibatch_size", 2000)
    # )
    config_dict["training_config"] = training

    eval_config = _deep_copy_dict(config_dict.get("eval_config", {}))
    eval_config["evaluation_parallel_to_training"] = True
    eval_config["evaluation_num_env_runners"] = candidate["eval_workers"]
    eval_config["evaluation_duration"] = candidate["eval_episodes"]
    eval_config["evaluation_config"] = _deep_copy_dict(eval_config.get("evaluation_config", {}))
    config_dict["eval_config"] = eval_config

    learner_config = _deep_copy_dict(config_dict.get("learner_config", {}))
    learner_config["num_gpus_per_learner"] = resources["num_gpus"]
    config_dict["learner_config"] = learner_config

    config = (
        PPOConfig()
        .environment(env=env_name, env_config=config_dict.get("env_config", {}))
        .framework("torch")
        .env_runners(**config_dict.get("env_runners_config", {}))
        .resources(**config_dict.get("resources_config", {}))
        .training(**config_dict.get("training_config", {}))
        .rl_module(model_config=config_dict.get("rl_module_config", {}).get("model_config", {}))
        .learners(**config_dict.get("learner_config", {}))
    )

    trial_name = f"eff_{trial_id:02d}_{candidate['num_env_runners']}r_{candidate['num_envs_per_env_runner']}e_{candidate['num_cpus_per_env_runner']:.2f}c"
    storage_path = root_dir / trial_name
    storage_path.mkdir(parents=True, exist_ok=True)

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        tune_config=TuneConfig(num_samples=1, metric="time_this_iter_s", mode="min"),
        run_config=RunConfig(
            name=trial_name,
            storage_path=str(root_dir.resolve()),
            stop={"training_iteration": stop_iters},
            verbose=0,
            callbacks=[SimpleProgressCallback(interval_s=3.0)],
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=0,
                checkpoint_at_end=False,
                num_to_keep=None,
            ),
        ),
    )

    print(f"\nStarting trial {trial_id}: {_format_candidate(candidate)}")
    results = tuner.fit()
    df = results.get_dataframe()
    avg_time = float(df["time_this_iter_s"].mean()) if "time_this_iter_s" in df.columns else float("nan")
    best_time = float(df["time_this_iter_s"].min()) if "time_this_iter_s" in df.columns else float("nan")
    last_row = df.iloc[-1] if len(df) > 0 else None

    summary = {
        "trial_id": trial_id,
        **candidate,
        "avg_time_this_iter_s": avg_time,
        "best_time_this_iter_s": best_time,
        "last_training_iteration": int(last_row["training_iteration"]) if last_row is not None else 0,
        "last_time_total_s": float(last_row["time_total_s"]) if last_row is not None else float("nan"),
        "last_episode_return_mean": float(last_row.get("env_runners/episode_return_mean", float("nan"))) if last_row is not None else float("nan"),
        "trial_dir": str(storage_path),
    }
    return summary


def _write_summary_txt(summary: list[dict], output_path: Path, cfg_path: str, cpus: int, gpus: int, envs: int, min_w: int, max_w: int) -> None:
    # Write a summary table using the current Ray RLlib config names.
    # This is intended to be easy to read and map back to config.yaml / PPOConfig.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Workstation efficiency tuning results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Config path: {cfg_path}\n")
        f.write(f"Hardware limits: cpus={cpus}, gpus={gpus}, envs={envs}\n")
        f.write(f"Runner bounds: min_w={min_w}, max_w={max_w}\n")
        f.write("\n")
        f.write("workers=num_env_runners\n")
        f.write("envs_per_w=num_envs_per_env_runner\n")
        f.write("cpus_per_w=num_cpus_per_env_runner\n")
        f.write("gpus_per_w=num_gpus_per_env_runner\n")
        f.write("cpus_per_d=num_cpus_for_main_process\n")
        f.write("\n")
        f.write(
            f"{'hardware':<18}{'workers':>10}{'envs_per_w':>15}{'eval_workers':>15}{'eval_episodes':>15}"
            f"{'cpus_per_w':>15}{'gpus_per_w':>15}{'cpus_per_d':>15}{'gpus_per_d':>15}{'time[s]':>12}\n"
        )
        f.write("-" * 140 + "\n")
        for trial in summary:
            f.write(
                f"{trial.get('hardware_architecture', 'unknown'):<18}"
                f"{trial['num_env_runners']:>10}"
                f"{trial['num_envs_per_env_runner']:>15}"
                f"{trial.get('eval_workers', trial['num_env_runners']):>15}"
                f"{trial.get('eval_episodes', 1):>15}"
                f"{trial['num_cpus_per_env_runner']:>15.5f}"
                f"{trial['num_gpus_per_env_runner']:>15.5f}"
                f"{trial['num_cpus_for_main_process']:>15.5f}"
                f"{trial.get('gpus_per_driver', trial.get('resources_num_gpus', 0)):>15.5f}"
                f"{trial['avg_time_this_iter_s']:>12.5f}"
                f"\n"
            )
        f.write("\n")
        best_trial = min(summary, key=lambda item: item["avg_time_this_iter_s"])
        f.write("Best candidate by average time_this_iter_s:\n")
        f.write(
            f"{best_trial.get('hardware_architecture', 'unknown'):<18}"
            f"{best_trial['num_env_runners']:>10}"
            f"{best_trial['num_envs_per_env_runner']:>15}"
            f"{best_trial.get('eval_workers', best_trial['num_env_runners']):>15}"
            f"{best_trial.get('eval_episodes', 1):>15}"
            f"{best_trial['num_cpus_per_env_runner']:>15.5f}"
            f"{best_trial['num_gpus_per_env_runner']:>15.5f}"
            f"{best_trial['num_cpus_for_main_process']:>15.5f}"
            f"{best_trial.get('gpus_per_driver', best_trial.get('resources_num_gpus', 0)):>15.5f}"
            f"{best_trial['avg_time_this_iter_s']:>12.5f}"
            f"\n"
        )
        f.write("\n")
        f.write("Recommended config snippet for best candidate:\n")
        f.write("env_runners_config:\n")
        f.write(f"  batch_mode: complete_episodes\n")
        f.write(f"  num_env_runners: {best_trial['num_env_runners']}\n")
        f.write(f"  num_envs_per_env_runner: {best_trial['num_envs_per_env_runner']}\n")
        f.write(f"  num_cpus_per_env_runner: {best_trial['num_cpus_per_env_runner']}\n")
        f.write(f"  num_gpus_per_env_runner: {best_trial['num_gpus_per_env_runner']}\n")
        f.write("resources_config:\n")
        f.write(f"  num_cpus_for_main_process: {best_trial['num_cpus_for_main_process']}\n")
        f.write(f"  num_gpus: {int(best_trial.get('resources_num_gpus', best_trial.get('gpus_per_driver', 0)))}\n")
        # f.write("training_config:\n")
        # f.write(f"  train_batch_size: {best_trial['train_batch_size']}\n")
        # f.write(f"  minibatch_size: {best_trial['minibatch_size']}\n")
        f.write("eval_config:\n")
        f.write(f"  evaluation_parallel_to_training: true\n")
        f.write(f"  evaluation_num_env_runners: {best_trial['eval_workers']}\n")
        f.write(f"  evaluation_duration: {best_trial['eval_episodes']}\n")

    print(f"\nSummary written to: {output_path}")


def tune_workstation_efficiency(
    cfg_path: str,
    cpus: int,
    gpus: int,
    envs: int,
    min_w: int,
    max_w: int,
    stop_iters: int = 5,
    results_root: str = "results_tuning/workstation_hardware_tuning",
) -> None:
    # Top-level tuning loop.
    # - cpus: total available CPU cores for Ray.
    # - gpus: total available GPUS for Ray.
    # - envs: desired total number of environments across all env runners.
    # - min_w/max_w: bounds for the number of env runners to try.
    config_path = Path(cfg_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    base_config = load_config(cfg_path)
    env_name = _register_env_from_config(base_config)
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"  

    ray.init(num_cpus=cpus, num_gpus=gpus if gpus > 0 else None, log_to_driver=False, include_dashboard=False)

    try:
        candidates = _build_candidate_grid(cpus=cpus, gpus=gpus, envs=envs, min_w=min_w, max_w=max_w)
        print(f"Found {len(candidates)} candidate configurations for tuning.")

        output_root = Path(results_root).absolute()
        summary = []
        for idx, candidate in enumerate(candidates, start=1):
            run_summary = _build_and_run_trial(
                base_config=base_config,
                cfg_path=cfg_path,
                candidate=candidate,
                trial_id=idx,
                root_dir=output_root,
                stop_iters=stop_iters,
                env_name=env_name,
            )
            summary.append(run_summary)

        output_path = output_root / f"workstation_efficiency_{int(time.time())}.txt"
        _write_summary_txt(summary, output_path, cfg_path, cpus, gpus, envs, min_w, max_w)
    finally:
        ray.shutdown()
