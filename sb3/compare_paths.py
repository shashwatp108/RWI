import json
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Import the A* helper from your analysis script ---
from find_cs import astar_min_steps, WALLS, DOORS, KEYS, GOALS

# --- Configuration ---
CRITICAL_STATES_FILE = "critical_states.json"
EPISODES_FILE = os.path.join("episode_logs", "episodes_full.json")
CRITICAL_STATE_INDEX = 8  # Set the index of the critical state you want to see

# --- Visualization Code ---
OBJECT_TO_INT = {"empty": 0, "Wall": 1, "Door": 2, "Key": 3, "Goal": 4}
cmap = plt.cm.get_cmap('Greys', len(OBJECT_TO_INT))


# --- NEW: A* function that returns the true optimal path coordinates ---
def get_true_optimal_path_with_coords(grid_codes, start_pos):
    """
    Calculates the true optimal path and returns the full coordinate list.
    """
    height, width = len(grid_codes), len(grid_codes[0])
    key_pos = next(((c, r) for r in range(height) for c in range(width) if grid_codes[r][c] in KEYS), None)

    if not key_pos:
        return astar_min_steps(grid_codes, start_pos, has_key_start=False)

    key_pickup_spots = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = key_pos[0] + dx, key_pos[1] + dy
        if 0 <= nx < width and 0 <= ny < height and grid_codes[ny][nx] not in WALLS:
            key_pickup_spots.append((nx, ny))

    if not key_pickup_spots:
        return float("inf"), None

    best_path_to_key_len = float("inf")
    best_pickup_pos = None
    best_path_to_key_coords = None

    for spot in key_pickup_spots:
        dist, path_coords = astar_min_steps(grid_codes, start_pos, has_key_start=False, override_goal=spot)
        if dist < best_path_to_key_len:
            best_path_to_key_len = dist
            best_pickup_pos = spot
            best_path_to_key_coords = path_coords

    if not best_path_to_key_coords:
        return float("inf"), None

    len_from_key, path_from_key_coords = astar_min_steps(grid_codes, best_pickup_pos, has_key_start=True)
    
    if not path_from_key_coords:
        return float("inf"), None

    # Stitch the two path segments together
    full_path = best_path_to_key_coords + path_from_key_coords[1:]
    return len(full_path) - 1, full_path


def plot_critical_comparison(full_episode_data, critical_state_data):
    """
    Generates a side-by-side plot with corrected A* and PPO path visualizations.
    """
    grid_codes = full_episode_data["initial_grid"]
    start_pos = tuple(full_episode_data["initial_agent_pos"])
    
    # --- FIX: Calculate the PPO's successful moves for an honest visual ---
    all_positions = [tuple(step["pos"]) for step in full_episode_data["steps"]]
    ppo_visual_path = [start_pos]
    for pos in all_positions:
        if pos != ppo_visual_path[-1]:
            ppo_visual_path.append(pos)
    # Add the final goal position for a complete path
    if full_episode_data["success"]:
        goal_pos = next(((c, r) for r, row in enumerate(grid_codes) for c, cell in enumerate(row) if cell == "Goal"), None)
        if goal_pos and ppo_visual_path[-1] != goal_pos:
            ppo_visual_path.append(goal_pos)
    
    # The length of the path that is actually drawn
    visual_ppo_len = len(ppo_visual_path) - 1
    # The total moves the agent TRIED to make (from the JSON)
    attempted_ppo_len = critical_state_data["agent_move_count"]

    # --- FIX: Calculate the true A* path for visualization ---
    astar_len, astar_path_coords = get_true_optimal_path_with_coords(grid_codes, start_pos)

    # --- Create Grid and Subplots ---
    height, width = len(grid_codes), len(grid_codes[0])
    grid_ints = np.zeros((height, width))
    for r in range(height):
        for c in range(width):
            grid_ints[r, c] = OBJECT_TO_INT.get(grid_codes[r][c], 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=100)
    fig.suptitle(f"Path Comparison for Critical Episode (ID: {full_episode_data['episode_id']})", fontsize=16)

    # --- Plot PPO Agent's Path (Left) ---
    ax1.imshow(grid_ints, cmap=cmap, origin='lower')
    path_x = [p[0] for p in ppo_visual_path]
    path_y = [p[1] for p in ppo_visual_path]
    ax1.plot(path_x, path_y, marker='o', markersize=4, color='cyan', linewidth=2, label='PPO Path')
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start')
    critical_pos = tuple(critical_state_data['s_star']['pos'])
    ax1.plot(critical_pos[0], critical_pos[1], 'r*', markersize=20, label=f'Critical State (t={critical_state_data["critical_step_t"]})')
    
    # --- FIX: Update title to be consistent and informative ---
    ax1.set_title(f"PPO Agent Path (Successful Moves: {visual_ppo_len})\n(Total Attempted Moves: {attempted_ppo_len})")
    ax1.legend()

    # --- Plot A* Optimal Path (Right) ---
    ax2.imshow(grid_ints, cmap=cmap, origin='lower')
    if astar_path_coords:
        path_x = [p[0] for p in astar_path_coords]
        path_y = [p[1] for p in astar_path_coords]
        ax2.plot(path_x, path_y, marker='o', markersize=4, color='red', linewidth=2, label='A* Path')
    ax2.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start')
    ax2.set_title(f"A* Optimal Path (Moves: {astar_len})")
    ax2.legend()
    
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(CRITICAL_STATES_FILE) or not os.path.exists(EPISODES_FILE):
        print(f"Error: Make sure '{CRITICAL_STATES_FILE}' and '{EPISODES_FILE}' exist.")
    else:
        with open(CRITICAL_STATES_FILE, 'r') as f: critical_states = json.load(f)
        with open(EPISODES_FILE, 'r') as f: all_episodes = json.load(f)
        
        episodes_dict = {ep['episode_id']: ep for ep in all_episodes}

        if len(critical_states) > CRITICAL_STATE_INDEX:
            target_critical_state = critical_states[CRITICAL_STATE_INDEX]
            target_episode_id = target_critical_state['episode_id']
            if target_episode_id in episodes_dict:
                target_full_episode = episodes_dict[target_episode_id]
                plot_critical_comparison(target_full_episode, target_critical_state)
            else:
                print(f"Error: Could not find episode_id {target_episode_id} in '{EPISODES_FILE}'")
        else:
            print(f"Error: Critical state index {CRITICAL_STATE_INDEX} is out of bounds.")