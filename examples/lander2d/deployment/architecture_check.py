# Print every weight name and shape stored in a checkpoint.
# Useful to check that a DeploymentPolicy* class matches the trained RLModule.
# Usage (from the example folder):
#   python deployment/architecture_check.py results/<run_name>/<trial_folder>/best_checkpoint
import pickle
import sys

best_cp_path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("Usage: python deployment/architecture_check.py <path/to/best_checkpoint>")

with open(f"{best_cp_path}/learner_group/learner/rl_module/default_policy/module_state.pkl", "rb") as f:
    state_dict = pickle.load(f)

for key, tensor in state_dict.items():
    print(f"{key:60s}  shape={tuple(tensor.shape)}")