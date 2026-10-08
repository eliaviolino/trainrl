import os

def load_episode_data(file_path: str):
    """
    Dynamically reads the episode_step_data.csv file written by evaluate_checkpoint, discovers column names,
    and returns a list of episodes where each episode is a dictionary of lists.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find log file at: {file_path}")

    episodes = []
    
    with open(file_path, "r") as f:
        # 1. Read the first row to get metrics dynamically
        header_line = f.readline().strip()
        if not header_line:
            return episodes
        
        headers = [h.strip() for h in header_line.split(",")]
        
        # Identify where the tracking columns are
        try:
            ep_col_idx = headers.index("episode")
        except ValueError:
            raise ValueError("The log file header must contain an 'episode' column.")

        # Metrics are every header except the 'episode' index tracker itself
        metric_keys = [h for h in headers if h != "episode"]

        current_ep_id = None
        current_ep_dict = None

        # 2. Extract values line by line
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            
            raw_values = line.split(",")
            if len(raw_values) != len(headers):
                continue  # skip malformed lines
            
            # Dynamically convert types: int for step/ep counts, float for decimals
            parsed_values = []
            for val in raw_values:
                val = val.strip()
                try:
                    parsed_values.append(int(val))
                except ValueError:
                    try:
                        parsed_values.append(float(val))  # also handles values like 1e-05
                    except ValueError:
                        parsed_values.append(val)  # keep as string if conversion fails

            # Extract current line's episode ID
            line_ep_id = parsed_values[ep_col_idx]

            # 3. Detect when a new episode begins
            if line_ep_id != current_ep_id:
                current_ep_id = line_ep_id
                
                # Initialize a new dictionary for this episode containing an empty list for every metric
                current_ep_dict = {metric: [] for metric in metric_keys}
                episodes.append(current_ep_dict)

            # 4. Populate the lists dynamically matching headers to values
            for idx, header in enumerate(headers):
                if header != "episode":
                    current_ep_dict[header].append(parsed_values[idx])

    return episodes


# --- EXAMPLE USAGE ---
#   python extract_final_results.py results/<run_name>/final_evaluation/episode_step_data.csv
if __name__ == "__main__":
    import sys
    FILE_PATH = sys.argv[1] if len(sys.argv) > 1 else sys.exit("Usage: python extract_final_results.py <path/to/episode_step_data.csv>")
    
    try:
        episodes = load_episode_data(FILE_PATH)
        print(f"Successfully loaded {len(episodes)} episodes.\n")
        
        # Verify your exact requested syntax:
        first_episode = episodes[0]
        
        print("--- Accessing First Episode Data ---")
        print(f"Total steps in Episode 0: {len(first_episode['step'])}")
        print(f"X positions sequence (first 5 steps): {first_episode['x'][:5]}")
        print(f"Y positions sequence (first 5 steps): {first_episode['y'][:5]}")
        print(f"Total Episode Reward accumulated: {sum(first_episode['reward']):.2f}")
        
        # Show that it works for any arbitrary metric found in the file
        print("\n--- Available metrics inside an episode ---")
        for key in first_episode.keys():
            print(f"- {key} (type: list, length: {len(first_episode[key])})")

    except Exception as e:
        print(f"Error: {e}")