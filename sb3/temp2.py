# File: visualize_population_batch.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# --- 1. Configuration ---
# Set this to the population file you generated
POPULATION_FILE = "populations/population_crit0_pop120.json"
# Directory where the output images will be saved
OUTPUT_DIR = "visualizations"
# The size of the highlight box to draw around the agent's critical position
# Set to 0 to disable the highlight box. A value like 7 is good for highlighting.
HIGHLIGHT_BOX_SIZE = 0

# --- 2. Style Configuration (No changes needed) ---
OBJECT_TO_INT = {"empty": 0, "Wall": 1, "Door": 2, "Key": 3, "Goal": 4}

# Define specific colors for each object type
custom_colors = [
    'white',       # 0: empty
    'darkgrey',    # 1: Wall
    'saddlebrown', # 2: Door
    'gold',        # 3: Key
    'limegreen'    # 4: Goal
]
cmap = mcolors.ListedColormap(custom_colors)
bounds = [i - 0.5 for i in range(len(OBJECT_TO_INT) + 1)]
norm = mcolors.BoundaryNorm(bounds, cmap.N)


# --- 3. Main Visualization Function (Modified to save files) ---
def plot_and_save_comparison(population_data, individual_index, output_path):
    """
    Generates and saves a side-by-side plot of the original grid vs. a modified one.
    """
    # --- Extract Data ---
    original_individual = population_data[0]
    modified_individual = population_data[individual_index]

    original_grid_str = original_individual["grid"]
    modified_grid_str = modified_individual["grid"]

    agent_pos = tuple(original_individual["agent_pos"])
    episode_id = original_individual["meta"]["source_episode"]
    
    # --- Convert string grids to integer arrays for plotting ---
    def grid_to_int_array(grid_str):
        int_array = []
        for row in grid_str:
            int_row = [OBJECT_TO_INT.get(tile, 0) for tile in row]
            int_array.append(int_row)
        return np.array(int_array)

    original_grid_int = grid_to_int_array(original_grid_str)
    modified_grid_int = grid_to_int_array(modified_grid_str)

    # --- Create Subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    fig.suptitle(f"Counterfactual Environment Comparison (Source Episode: {episode_id})", fontsize=18, weight='bold')

    # --- Plot Original Grid (Left) ---
    ax1.imshow(original_grid_int, cmap=cmap, norm=norm, origin='lower')
    ax1.set_title("Original Environment (Individual 0)", fontsize=14)
    
    # --- Plot Modified Grid (Right) ---
    ax2.imshow(modified_grid_int, cmap=cmap, norm=norm, origin='lower')
    ax2.set_title(f"Modified Environment (Individual {individual_index})", fontsize=14)

    # --- Add Highlight Rectangle and Agent Marker to BOTH plots ---
    for ax in [ax1, ax2]:
        # Only draw the highlight box if its size is greater than 0
        if HIGHLIGHT_BOX_SIZE > 0:
            radius = HIGHLIGHT_BOX_SIZE // 2
            rect_origin = (agent_pos[0] - radius - 0.5, agent_pos[1] - radius - 0.5)
            rect = patches.Rectangle(
                rect_origin, HIGHLIGHT_BOX_SIZE, HIGHLIGHT_BOX_SIZE,
                linewidth=2.5, edgecolor='cyan', facecolor='none', label='Area of Interest'
            )
            ax.add_patch(rect)
        
        # Mark the critical agent position
        ax.plot(agent_pos[0], agent_pos[1], 'r*', markersize=18, markeredgecolor='black', label="Agent's Position")
        ax.legend(loc='upper right')
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Add a shared, descriptive colorbar ---
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(mappable, cax=cbar_ax, ticks=np.arange(len(OBJECT_TO_INT)))
    cbar.ax.set_yticklabels(list(OBJECT_TO_INT.keys()), fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # --- Save the figure instead of showing it ---
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory


# --- 4. Main Execution Block (Modified for batch processing) ---
if __name__ == "__main__":
    if not os.path.exists(POPULATION_FILE):
        print(f"Error: Population file not found at '{POPULATION_FILE}'")
    else:
        with open(POPULATION_FILE, 'r') as f:
            population_data = json.load(f)
        
        if len(population_data) > 1:
            # Create the output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"Visualizations will be saved to the '{OUTPUT_DIR}' directory.")

            # Loop through all modified individuals (skipping the original at index 0)
            num_individuals = len(population_data)
            for i in range(1, num_individuals):
                # Construct a unique filename for each plot
                filename = f"comparison_{i:03d}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                
                print(f"[{i}/{num_individuals - 1}] Generating visualization for individual {i} -> {output_path}")
                
                # Call the function to generate and save the plot
                plot_and_save_comparison(population_data, i, output_path)
            
            print("\n✅ Batch visualization complete!")
        else:
            print("Population file contains only the original individual. No comparisons to generate.")