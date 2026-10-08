# Usage:
#   python tune.py [CONFIG]
#
# Searches for the fastest env-runner / CPU / GPU split for your workstation and writes a
# summary with a ready-to-copy config snippet to results_tuning/workstation_hardware_tuning/.
# Run it on its own, not in the same process as train.py.
import sys
from trainrl import tune_workstation_efficiency

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

# Specify your workstation hardware limits right here in the main block
tune_workstation_efficiency(
    cfg_path=CONFIG_PATH,
    cpus=32,
    gpus=1,
    envs=100,
    min_w=1,
    max_w=25,
    stop_iters=1,
)
