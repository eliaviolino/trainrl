# Usage:
#   python train.py [CONFIG] [RESTORE_DIR] [RESULTS_DIR] [CONTINUE_FINISHED]
#
#   python train.py configs/config_mlp.yaml                                  # new training run
#   python train.py configs/config_mlp.yaml results/<run_name>               # resume a stopped run
#   python train.py configs/config_mlp.yaml results/<run_name> results true  # keep training a finished run
#                                                                          # (after raising training_iterations)
import sys
from trainrl import train_ppo

if __name__ == "__main__":
    CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "configs/config_mlp.yaml"
    RESTORE_PATH = sys.argv[2] if len(sys.argv) > 2 else None
    RESULTS_DIR = sys.argv[3] if len(sys.argv) > 3 else "results"
    CONT_TERMINATION = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else False

    train_ppo(CONFIG_PATH, restore_dir=RESTORE_PATH, results_dir=RESULTS_DIR, debug=False,
              continue_finished=CONT_TERMINATION)
