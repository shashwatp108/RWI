import gymnasium as gym
import json
import os
from datetime import datetime
import numpy as np
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO

# --- 1. Define Constants ---
MODEL_FILENAME = "ppo_doorkey_6x6_mini_2.zip"
ENV_NAME = "MiniGrid-DoorKey-6x6-v0"
LOG_DIR = "episode_logs" # <-- CHANGE: New folder for clarity
N_SUCCESSFUL_EPISODES = 1000  # <-- CHANGE: Now specifies the target number of SUCCESSFUL episodes
MAX_STEPS_PER_EPISODE = 200

# --- 2. Helper Functions (No changes here) ---

def serialize_grid(env):
    """
    Converts the MiniGrid environment's grid object into a serializable
    2D list of object class names (e.g., 'Wall', 'Key', 'Goal').
    """
    grid = env.unwrapped.grid
    width, height = grid.width, grid.height
    code_grid = [["empty" for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            obj = grid.get(x, y)
            if obj is not None:
                code_grid[y][x] = obj.__class__.__name__
    return code_grid

# --- 3. Main Logging Function (Modified) ---

def log_successful_episodes(n_episodes=N_SUCCESSFUL_EPISODES):
    """
    Loads a trained model and runs episodes UNTIL the desired number of
    SUCCESSFUL episodes have been collected and logged.
    """
    print(f"--- Starting Episode Logging (Target: {n_episodes} Successful Episodes) ---")
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Loading trained model from: {MODEL_FILENAME}")
    model = PPO.load(MODEL_FILENAME)

    env = gym.make(ENV_NAME)
    env = FlatObsWrapper(env)

    successful_logs = []
    total_episodes_run = 0

    # <-- CHANGE: Loop until we have enough successful logs, not for a fixed number of runs.
    while len(successful_logs) < n_episodes:
        total_episodes_run += 1
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_steps = []

        initial_grid = serialize_grid(env)
        # Convert numpy types to standard python types for JSON serialization
        initial_agent_pos = tuple(map(int, env.unwrapped.agent_pos))
        initial_agent_dir = int(env.unwrapped.agent_dir)

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            action, _ = model.predict(obs, deterministic=False)
            
            current_pos = tuple(map(int, env.unwrapped.agent_pos))
            current_dir = int(env.unwrapped.agent_dir)
            
            episode_steps.append({
                "step": step_count,
                "pos": current_pos,
                "dir": current_dir,
                "action": int(action),
                "grid_snapshot": serialize_grid(env)
            })

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            step_count += 1
        
        # Determine if the agent reached the goal
        final_cell = env.unwrapped.grid.get(*env.unwrapped.agent_pos)
        success = final_cell is not None and final_cell.type == 'goal'

        # <-- CHANGE: Only append the log if the episode was a success.
        if success:
            successful_logs.append({
                "episode_id": total_episodes_run, # Use total run count for a unique ID
                "initial_grid": initial_grid,
                "initial_agent_pos": initial_agent_pos,
                "initial_agent_dir": initial_agent_dir,
                "steps": episode_steps,
                "final_step_count": step_count,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })
            
            # <-- CHANGE: More informative progress report
            print(f"Collected {len(successful_logs)}/{n_episodes} successful episodes. (Total episodes run: {total_episodes_run})")


    # Final save of all collected logs
    log_file_path = os.path.join(LOG_DIR, "episodes_full.json")
    print(f"\nTarget reached. Saving {len(successful_logs)} successful episodes to {log_file_path}")
    with open(log_file_path, "w") as f:
        json.dump(successful_logs, f, indent=2)

    env.close()
    print("--- Episode Logging Finished Successfully! ---")
    return successful_logs

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: Model file '{MODEL_FILENAME}' not found!")
    else:
        log_successful_episodes(n_episodes=N_SUCCESSFUL_EPISODES)