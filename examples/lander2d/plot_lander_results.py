# Usage:
#   python plot_lander_results.py results/<run_name>/final_evaluation/episode_step_data.csv
import sys
import matplotlib.pyplot as plt
from extract_final_results import load_episode_data

FILE_PATH = sys.argv[1] if len(sys.argv) > 1 else sys.exit("Usage: python plot_lander_results.py <path/to/episode_step_data.csv>")

episodes = load_episode_data(FILE_PATH)

# Plot all trajectories in one figure
plt.figure(figsize=(8, 6))
for k in range(len(episodes)):
    ep = episodes[k]
    x = ep["x"]
    y = ep["y"]
    plt.plot(x, y, marker='o', markersize=3, label=f"Episode {k}")   
plt.title("Lander Trajectories")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid()
# plt.legend()
plt.show()