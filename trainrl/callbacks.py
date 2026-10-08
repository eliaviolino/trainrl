import os
import csv
import time
import shutil
import numpy as np
import torch
from ray.tune import Callback
from ray.rllib.callbacks.callbacks import RLlibCallback
from trainrl.utils import _copy_checkpoint

# Custom callback to print training progress and save config to trial dir
class ProgressCallback(Callback):
    def __init__(self, interval_s: float = 2.0, config_path: str = None):
        self.interval_s  = interval_s
        self.config_path = config_path
        self._last_print = 0.0
        self._config_saved = False

    def on_trial_result(self, iteration, trials, trial, result, **info):
        # Save config on first result when we know the trial path
        if not self._config_saved and self.config_path:
            shutil.copy(self.config_path, os.path.join(trial.local_path, "config.yaml"))
            self._config_saved = True

        now = time.time()
        if now - self._last_print >= self.interval_s:
            reward = (
                result.get("env_runners", {}).get("episode_return_mean")
                if isinstance(result.get("env_runners"), dict)
                else None
            )
            if reward is None:
                reward = result.get("env_runners/episode_return_mean")
            if reward is None:
                reward = result.get("episode_reward_mean")
            if reward is None:
                reward = result.get("episode_return_mean", float("nan"))

            it      = result.get("training_iteration", 0)
            steps   = result.get("num_env_steps_sampled_lifetime", 0)
            elapsed = result.get("time_total_s", 0.0)
            print(f"Iter {it:>4} | reward={reward:>10.2f} | steps={steps:>7.0f} | time={elapsed:.1f}s")
            self._last_print = now


class BestEvalCheckpointCallback(Callback):
    def __init__(self, metric_key, dst_dir_name="best_checkpoint"):
        self.metric_key = metric_key
        self.best_score = None
        self.dst_dir_name = dst_dir_name

    def get_state(self):
        # Persisted by Tune so a restored run doesn't overwrite best_checkpoint with a worse one.
        return {"best_score": self.best_score} if self.best_score is not None else None

    def set_state(self, state):
        self.best_score = state.get("best_score")

    def _seed_from_progress(self, trial, current_iter):
        # Runs saved before get_state existed: recover the best eval score from earlier rows of progress.csv.
        path = os.path.join(trial.local_path, "progress.csv")
        if not os.path.exists(path):
            return
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    if int(row["training_iteration"]) >= current_iter:
                        continue
                    score = float(row["evaluation/env_runners/episode_return_mean"])
                except (KeyError, ValueError, TypeError):
                    continue
                if self.best_score is None or score > self.best_score:
                    self.best_score = score

    def on_trial_result(self, iteration, trials, trial, result, **info):
        # 1. Extract eval metric from nested dict
        eval_dict = result.get("evaluation", {})
        eval_env = eval_dict.get("env_runners", {})
        score = eval_env.get("episode_return_mean", None)
        if score is None:
            return

        if self.best_score is None:
            self._seed_from_progress(trial, result.get("training_iteration", 0))

        # 2. If new best, copy the latest checkpoint
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            if trial.checkpoint:
                src = trial.checkpoint.path
                dst = os.path.join(trial.local_path, self.dst_dir_name)
                # use your _copy_checkpoint helper here
                _copy_checkpoint(src, dst)


class ExtendStopCallback(Callback):
    """Overrides the stop iteration stored in a restored trial, so a finished trial can keep training."""
    def __init__(self, training_iterations: int):
        self.training_iterations = training_iterations

    def on_trial_start(self, iteration, trials, trial, **info):
        trial.stopping_criterion["training_iteration"] = self.training_iterations


class FixOptimizerAfterCheckpoint(RLlibCallback):
    """Fixes Ray bug #51560: GPU learner checkpoint restores betas as Tensor instead of float."""
    def on_checkpoint_loaded(self, *, algorithm, **kwargs):
        def fix_betas(learner):
            for optimizer in learner._optimizer_parameters.keys():
                for group in optimizer.param_groups:
                    if isinstance(group["betas"][0], torch.Tensor):
                        group["betas"] = tuple(b.item() for b in group["betas"])
        algorithm.learner_group.foreach_learner(fix_betas)

class CustomMetricCallbacks(RLlibCallback):

    def on_episode_end(self, *, episode, env_runner, metrics_logger,
                       env, env_index, rl_module, **kwargs):

        # get_infos(-1) returns a LIST — take the first (and only) element
        info = episode.get_infos(-1)
        if not info:
            return

        last_info = info[0] if isinstance(info, list) else info

        metrics_to_log = env_runner.config.get("metrics_to_log", None)

        for key, value in last_info.items():
            if not isinstance(value, (int, float, bool, np.floating, np.integer)):
                continue
            if metrics_to_log is not None and key not in metrics_to_log:
                continue
            metrics_logger.log_value(
                ("metrics_logged", key),
                float(value),
                reduce="mean",
                window=100,
            )