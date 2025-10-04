import json
import os
import heapq
import math

# --- 1. Define Constants & Configuration ---
INPUT_LOG_FILE = os.path.join("episode_logs", "episodes_full.json")
OUTPUT_CRITICAL_STATES_FILE = "critical_states.json"
ALPHA = 1.25  # Sub-optimality threshold (25% longer than optimal)

# (0:right, 1:down, 2:left, 3:up) are directions, not actions
# The actions are enum members: 0:left, 1:right, 2:forward, 3:pickup, 4:drop, 5:toggle, 6:done
ACTION_FORWARD = 2 

# --- 2. A* Implementation (from your plan) ---

# Helper sets for identifying grid object types
# Note: Double-check these names match what's in your episodes_full.json
WALLS = {"Wall"}
DOORS = {"Door"}
KEYS = {"Key"}
GOALS = {"Goal"}

def astar_min_steps(grid_codes, start_pos, has_key_start=False, override_goal=None):
    """
    Calculates the shortest path in a grid using the A* algorithm.
    Accepts an optional 'override_goal' to find paths to specific points.
    """
    height = len(grid_codes)
    width = len(grid_codes[0])
    
    # --- CHANGE: Allows specifying a temporary goal (like a key pickup spot) ---
    if override_goal:
        goal_positions = [override_goal]
    else:
        goal_positions = [(c, r) for r in range(height) for c in range(width) if grid_codes[r][c] in GOALS]
    
    if not goal_positions:
        return float("inf"), None

    def heuristic(pos):
        return min(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in goal_positions)

    start_node = (start_pos[0], start_pos[1], bool(has_key_start))
    g_score = {start_node: 0}
    f_score_initial = heuristic(start_pos)
    pq = [(f_score_initial, 0, start_node)]
    parent_map = {}

    while pq:
        _, current_g, current_node = heapq.heappop(pq)
        
        if current_g > g_score.get(current_node, float("inf")):
            continue

        x, y, has_key = current_node
        
        if (x, y) in goal_positions:
            path = []
            while current_node in parent_map:
                path.append((current_node[0], current_node[1]))
                current_node = parent_map[current_node]
            path.append(start_pos)
            path.reverse()
            return current_g, path

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if not (0 <= nx < width and 0 <= ny < height):
                continue

            tile = grid_codes[ny][nx]

            if tile in WALLS:
                continue
            
            if tile in DOORS and not has_key:
                continue

            # This logic remains from the old version for pathing from key to goal
            new_has_key = has_key or (tile in KEYS)
            
            neighbor_node = (nx, ny, new_has_key)
            tentative_g_score = current_g + 1

            if tentative_g_score < g_score.get(neighbor_node, float("inf")):
                parent_map[neighbor_node] = current_node
                g_score[neighbor_node] = tentative_g_score
                h_score = heuristic((nx, ny))
                f_score = tentative_g_score + h_score
                heapq.heappush(pq, (f_score, tentative_g_score, neighbor_node))

    return float("inf"), None

# --- 3. Critical State Identification Logic ---

def compute_has_key_sequence(episode):
    """Determines at which step the agent acquires a key."""
    has_key_seq = []
    has_key = False
    for step in episode["steps"]:
        grid = step["grid_snapshot"]
        x, y = step["pos"]
        if grid[y][x] in KEYS:
            has_key = True
        has_key_seq.append(has_key)
    return has_key_seq

def get_true_optimal_path(grid_codes, start_pos):
    """
    Calculates the true optimal path by modeling the 'pickup' action correctly.
    It finds the path to a tile *next* to the key, not *on* it.
    """
    height = len(grid_codes)
    width = len(grid_codes[0])

    # Find the key's position
    key_pos = None
    for r in range(height):
        for c in range(width):
            if grid_codes[r][c] in KEYS:
                key_pos = (c, r)
                break
    
    if not key_pos: # No key in the grid
        return astar_min_steps(grid_codes, start_pos, has_key_start=False)

    # Find all valid, non-wall tiles adjacent to the key
    key_pickup_spots = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = key_pos[0] + dx, key_pos[1] + dy
        if 0 <= nx < width and 0 <= ny < height and grid_codes[ny][nx] not in WALLS:
            key_pickup_spots.append((nx, ny))

    if not key_pickup_spots: # Key is boxed in
        return float("inf"), None

    # Stage 1: Find the shortest path from start to any key pickup spot
    best_path_to_key_len = float("inf")
    best_pickup_pos = None
    
    for spot in key_pickup_spots:
        # We use the original A* as a helper to find distance to the spot
        dist, _ = astar_min_steps(grid_codes, start_pos, has_key_start=False, override_goal=spot)
        if dist < best_path_to_key_len:
            best_path_to_key_len = dist
            best_pickup_pos = spot
    
    if math.isinf(best_path_to_key_len):
        return float("inf"), None

    # Stage 2: Find the shortest path from the best pickup spot to the goal (with the key)
    path_from_key_len, _ = astar_min_steps(grid_codes, best_pickup_pos, has_key_start=True)

    # The total optimal length is the sum of the two stages
    return best_path_to_key_len + path_from_key_len, None # We only need the length for now


def find_critical_state(episode):
    initial_grid = episode["initial_grid"]
    initial_pos = tuple(episode["initial_agent_pos"])
    
    # <-- CHANGE: Use the new, more accurate A* path calculation
    astar_total_len, _ = get_true_optimal_path(initial_grid, initial_pos)
    
    if math.isinf(astar_total_len):
        return None

    steps = episode["steps"]
    agent_move_count = sum(1 for step in steps if step["action"] == ACTION_FORWARD)

    # <-- FIX: Add 1 to account for the final, unlogged goal-reaching move.
    if episode["success"]:
        agent_move_count += 1

    # Now, compare the CORRECTED move count to the A* length.
    if agent_move_count > ALPHA * astar_total_len:
        T = len(steps)
        has_key_seq = compute_has_key_sequence(episode)
        
        for t in range(T):
            step_data = steps[t]
            current_pos = tuple(step_data["pos"])
            has_key_at_t = has_key_seq[t]

            astar_remaining_len, _ = astar_min_steps(initial_grid, current_pos, has_key_start=has_key_at_t)

            agent_remaining_moves = sum(1 for step in steps[t:] if step["action"] == ACTION_FORWARD)
            # <-- FIX: Also correct the remaining moves calculation.
            if episode["success"]:
                agent_remaining_moves += 1
            
            if math.isinf(astar_remaining_len):
                reason = "agent_entered_state_with_no_optimal_path"
            elif agent_remaining_moves > ALPHA * astar_remaining_len:
                reason = "agent_path_exceeded_alpha_threshold"
            else:
                continue

            # Found the first critical state
            return {
                "episode_id": episode["episode_id"],
                "critical_step_t": t,
                "reason": reason,
                "alpha": ALPHA,
                "agent_total_actions": episode["final_step_count"],
                "agent_move_count": agent_move_count, # The corrected metric
                "astar_total_len": round(astar_total_len, 2),
                "agent_remaining_moves_at_t": agent_remaining_moves, # Corrected
                "astar_remaining_len_at_t": round(astar_remaining_len, 2),
                "s_star": step_data
            }
            
    return None

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    print(f"--- Starting Critical State Analysis ---")
    
    if not os.path.exists(INPUT_LOG_FILE):
        print(f"Error: Log file not found at '{INPUT_LOG_FILE}'")
        print("Please run the 'generate_logs.py' script first.")
    else:
        print(f"Loading episodes from {INPUT_LOG_FILE}...")
        with open(INPUT_LOG_FILE, 'r') as f:
            all_episodes = json.load(f)

        critical_states = []
        total_episodes = len(all_episodes)
        print(f"Processing {total_episodes} episodes...")

        for i, episode in enumerate(all_episodes):
            cs = find_critical_state(episode)
            if cs:
                critical_states.append(cs)
            
            if (i + 1) % 100 == 0:
                print(f"  ...processed {i + 1}/{total_episodes} episodes. Found {len(critical_states)} critical states so far.")

        print("\n--- Analysis Complete ---")
        print(f"Found {len(critical_states)} critical states out of {total_episodes} episodes.")

        print(f"Saving results to {OUTPUT_CRITICAL_STATES_FILE}...")
        with open(OUTPUT_CRITICAL_STATES_FILE, 'w') as f:
            json.dump(critical_states, f, indent=2)

        print("Successfully saved critical states. You are ready for Phase 2!")