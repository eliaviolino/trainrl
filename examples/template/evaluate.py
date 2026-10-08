# Usage:
#   python evaluate.py results/<run_name>/<trial_folder>/best_checkpoint
#
# Reads config.yaml from the trial folder (saved there by training) and writes
# episode_step_data.csv + evaluation_summary.txt to <trial_folder>/final_evaluation_results/.
from trainrl import evaluate_checkpoint
import os
import sys
import yaml

if len(sys.argv) != 2:
    sys.exit("Usage: python evaluate.py results/<run_name>/<trial_folder>/best_checkpoint")

CHECKPOINT_PATH = os.path.abspath(sys.argv[1].rstrip("/"))
CONFIG_PATH = os.path.join(os.path.dirname(CHECKPOINT_PATH), "config.yaml")
OUTPUT_DIR = os.path.join(os.path.dirname(CHECKPOINT_PATH), "final_evaluation_results")

with open(CONFIG_PATH, "r") as f:
    config_dict = yaml.safe_load(f)
print(f"Loaded config from checkpoint directory: {CONFIG_PATH}")

num_episodes = config_dict.get("final_eval_config", {}).get("num_episodes", 5)
print(f"Evaluating for {num_episodes} episodes based on config settings.")

evaluate_checkpoint(
    checkpoint_path=CHECKPOINT_PATH,
    config_path=CONFIG_PATH,
    output_dir=OUTPUT_DIR,
    num_episodes=num_episodes,
)
