import os
import ray
import yaml
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import RunConfig, TuneConfig, ResumeConfig

from trainrl.utils import _register_env_from_config, _copy_checkpoint, build_rl_module_config
from trainrl.callbacks import ProgressCallback, BestEvalCheckpointCallback, ExtendStopCallback, FixOptimizerAfterCheckpoint, CustomMetricCallbacks
from trainrl.evaluate import evaluate_checkpoint


def train_ppo(cfg_path: str, restore_dir: str = None, results_dir: str = "results", debug: bool = False,
              continue_finished: bool = False):
    """
    continue_finished: with restore_dir, also resume trials that already reached their stop
        iteration, training them up to `training_iterations` from the YAML (the new total).
    """
    # Load config from YAML file
    with open(cfg_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Register the custom environment with Tune from config
    env_name = _register_env_from_config(config_dict)
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"  

    # Initialize Ray (with log_to_driver=False to avoid duplicate logs in this script)
    if not ray.is_initialized():
        ray.init(log_to_driver=debug)

    # Ensure the results directory exists
    results_root = os.path.abspath(results_dir)

    # Create PPO config using the loaded YAML config
    rl_module_cfg = config_dict.get("rl_module_config", {})

    config = (
        PPOConfig()
        .environment(env=env_name, env_config=config_dict.get("env_config", {}))
        .framework("torch")
        .callbacks([FixOptimizerAfterCheckpoint, CustomMetricCallbacks])
        .env_runners(**config_dict.get("env_runners_config", {}))
        .resources(**config_dict.get("resources_config", {}))
        .training(**config_dict.get("training_config", {}))
        .evaluation(**config_dict.get("eval_config", {}))
        .learners(**config_dict.get("learner_config", {}))
        # Load any custom RL module settings from YAML, such as CNN layers or LSTM options.
        .rl_module(
            **build_rl_module_config(rl_module_cfg)
        )
    )

    if restore_dir:
        tuner = tune.Tuner.restore(
            os.path.abspath(restore_dir),
            trainable="PPO",
            resume_unfinished=True,
            resume_errored=True,
            param_space=config.to_dict(),
            _resume_config=ResumeConfig(
                finished=ResumeConfig.ResumeType.RESUME if continue_finished else ResumeConfig.ResumeType.SKIP,
                unfinished=ResumeConfig.ResumeType.RESUME,
                errored=ResumeConfig.ResumeType.RESUME,
            ),
        )
        if continue_finished:
            # The stop criterion is saved inside each trial, so it must be raised after restore.
            tuner._local_tuner.get_run_config().callbacks.append(
                ExtendStopCallback(config_dict.get("training_iterations", 200))
            )
    else:
        # Create Tune Tuner and run training
        train_name = config_dict.get("run_name", "ppo_lander")
        # Ensure a unique run name under the "results" storage dir (visual_target, visual_target_01, etc.).
        # This prevents overwriting earlier training runs when repeating experiments.
        base_name = train_name
        candidate = base_name
        i = 1
        while os.path.exists(os.path.join(results_root, candidate)):
            candidate = f"{base_name}_{i:02d}"
            i += 1
        train_name = candidate
        config_dict["run_name"] = train_name
        tuner = tune.Tuner(
            "PPO",
            param_space=config,
            tune_config=TuneConfig(num_samples=1),
            run_config=RunConfig(
                name=config_dict.get("run_name", "ppo_lander"),
                storage_path=results_root,
                stop={"training_iteration": config_dict.get("training_iterations", 200)},
                verbose=1,   # silence Tune's own table
                callbacks=[
                    ProgressCallback(interval_s=2.0, config_path=cfg_path),
                    BestEvalCheckpointCallback(metric_key="env_runners/episode_return_mean"),
                ],
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=1, 
                    num_to_keep= 1, #config_dict.get("num_checkpoints_to_keep", 2),
                    checkpoint_at_end=config_dict.get("checkpoint_at_end_of_training", True),
                    checkpoint_score_attribute=None #"env_runners/episode_return_mean",
                    # checkpoint_score_order="max",
                ),
                # Keep the best checkpoint in a stable folder for later evaluation.
            ),
        )

    results = tuner.fit()

    # Get the best checkpoint based on the highest mean episode return
    
    best = results.get_best_result(
        metric="evaluation/env_runners/episode_return_mean",
        mode="max",
    )

    try:
        best_ckpt_path = best.path + "/best_checkpoint"
    except Exception:
        best_ckpt_path = None

    if best_ckpt_path:
        preserved_best_dir = os.path.join(best.path, "eval_checkpoint")
        _copy_checkpoint(best_ckpt_path, preserved_best_dir)

    # Define how many episodes to evaluate after training ends
    eval_episodes = config_dict.get('final_eval_config', {}).get('num_episodes', 100)
    
    # Store the final evaluation logs under the trial folder so they stay with the run.
    if restore_dir:
        eval_output_directory = os.path.join(os.path.abspath(restore_dir), "final_evaluation")
    else:
        eval_output_directory = os.path.join(results_root, config_dict.get("run_name", "ppo_lander"), "final_evaluation")
    os.makedirs(eval_output_directory, exist_ok=True)

    # Run the custom evaluation script
    evaluate_checkpoint(
        checkpoint_path=best.checkpoint.path,
        config_path=cfg_path,
        output_dir=eval_output_directory,
        num_episodes=eval_episodes
    )

    ray.shutdown()
