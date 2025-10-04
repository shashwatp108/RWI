import json
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Import the A* function from your other script ---
# Make sure 'find_critical_states.py' is in the same directory
from find_cs import astar_min_steps 

CRITICAL_STATES_FILE = "critical_states.json"

# Maps grid object names to numbers for color plotting
OBJECT_TO_INT = {
    "empty": 0,
    "Wall": 1,
    "Door": 2,
    "Key": 3,
    "Goal": 4,
    "Agent": 5, # We'll add the agent's position for context
}
# A color map for visualization
cmap = plt.cm.get_cmap('viridis', len(OBJECT_TO_INT))

def visualize_astar_path(grid_codes, start_pos, has_key_start, path):
    """
    Renders the grid and overlays the A* path using Matplotlib.
    """
    if not path:
        print("No path to visualize.")
        return

    # Convert grid of strings to a grid of numbers
    height = len(grid_codes)
    width = len(grid_codes[0])
    grid_ints = np.zeros((height, width))
    for r in range(height):
        for c in range(width):
            grid_ints[r, c] = OBJECT_TO_INT.get(grid_codes[r][c], 0)
    
    # Mark the agent's starting position
    grid_ints[start_pos[1], start_pos[0]] = OBJECT_TO_INT["Agent"]

    # Plot the grid
    fig, ax = plt.subplots()
    ax.imshow(grid_ints, cmap=cmap)

    # Overlay the A* path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, marker='o', color='red', linewidth=2, label='A* Path')

    # Formatting the plot
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, height, 1))
    
    plt.title(f"A* Optimal Path (Length: {len(path)-1})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(CRITICAL_STATES_FILE):
        print(f"Error: Critical states file not found at '{CRITICAL_STATES_FILE}'")
        print("Please run the corrected 'find_critical_states.py' script first.")
    else:
        with open(CRITICAL_STATES_FILE, 'r') as f:
            critical_states = json.load(f)

        if not critical_states:
            print("No critical states found to visualize.")
        else:
            # Visualize the first critical state found
            state_to_viz = critical_states[0]
            print(f"Visualizing A* path for episode_id: {state_to_viz['episode_id']}")
            
            s_star = state_to_viz['s_star']
            grid = s_star['grid_snapshot']
            
            # Note: We visualize the path from the START of the episode
            # To see the optimal path from the critical state, change start_pos
            initial_pos = tuple(critical_states[0]['s_star']['pos'])
            
            # For DoorKey, the agent never starts with a key
            has_key = False 
            
            # Recalculate the A* path to get the list of coordinates
            length, path_coords = astar_min_steps(grid, initial_pos, has_key_start=has_key)

            if path_coords:
                visualize_astar_path(grid, initial_pos, has_key, path_coords)
            else:
                print(f"Could not find an A* path for episode {state_to_viz['episode_id']}")