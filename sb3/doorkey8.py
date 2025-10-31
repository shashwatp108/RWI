import gymnasium as gym
import numpy as np
from minigrid.core.world_object import Door, Key
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# --- 1. Define the Custom Wrapper (Now with Refined Reward Shaping) ---

class EngineeredObsWrapper(gym.Wrapper):
    """
    A wrapper to engineer observations and shape rewards for the MiniGrid environment.

    This wrapper transforms the raw grid observation into a compact feature vector AND
    provides intermediate, one-time rewards to guide the agent without being exploitable.
    """
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        # --- State flags for one-time rewards ---
        self.picked_up_key = False
        self.was_next_to_door = False
        self.opened_door = False
        # --- Add a tracker for the episode's cumulative reward ---
        self.episode_reward = 0.0

    def reset(self, **kwargs):
        # Reset the environment and all internal state flags
        obs, info = self.env.reset(**kwargs)
        self.picked_up_key = False
        self.was_next_to_door = False
        self.opened_door = False
        # Reset the episode reward tracker
        self.episode_reward = 0.0
        return self.observation(obs), info

    def step(self, action):
        # Take a step and apply the refined reward shaping
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- Refined Reward Shaping Logic ---
        shaped_reward = reward
        
        # 1. Small penalty for each step to encourage efficiency
        shaped_reward -= 0.01

        # 2. ONE-TIME bonus for picking up the key
        if self.unwrapped.carrying and not self.picked_up_key:
            shaped_reward += 0.1
            self.picked_up_key = True

        # 3. ONE-TIME bonus for being next to the door with the key
        agent_pos = np.array(self.unwrapped.agent_pos)
        door, door_pos = self._get_door()
        is_next_to_door = door_pos is not None and np.linalg.norm(agent_pos - door_pos) == 1

        if is_next_to_door and self.picked_up_key and not self.was_next_to_door:
             shaped_reward += 0.2
             self.was_next_to_door = True # This flag is NOT reset, preventing the exploit

        # 4. ONE-TIME bonus for opening the door
        if door and door.is_open and not self.opened_door:
            shaped_reward += 0.5
            self.opened_door = True

        # Update the cumulative episode reward
        self.episode_reward += shaped_reward

        # Process the observation
        engineered_obs = self.observation(obs)
        
        return engineered_obs, shaped_reward, terminated, truncated, info

    def _get_door(self):
        """Helper to find the door object and its position."""
        for obj in self.unwrapped.grid.grid:
            if isinstance(obj, Door):
                return obj, np.array(obj.cur_pos)
        return None, None

    def observation(self, obs):
        """
        Processes the original observation and returns the engineered vector.
        """
        agent_pos = np.array(self.unwrapped.agent_pos)
        agent_dir = self.unwrapped.agent_dir

        key_pos_tuple, door_pos_tuple, door = None, None, None
        for obj in self.unwrapped.grid.grid:
            if isinstance(obj, Key):
                key_pos_tuple = obj.cur_pos
            elif isinstance(obj, Door):
                door_pos_tuple = obj.cur_pos
                door = obj

        key_pos = np.array(key_pos_tuple) if key_pos_tuple is not None else np.array([-1, -1])
        door_pos = np.array(door_pos_tuple) if door_pos_tuple is not None else np.array([-1, -1])

        norm_agent_pos = agent_pos / self.unwrapped.width
        norm_key_pos = key_pos / self.unwrapped.width
        norm_door_pos = door_pos / self.unwrapped.width
        
        rel_key_pos = norm_key_pos - norm_agent_pos
        rel_door_pos = norm_door_pos - norm_agent_pos

        agent_dir_norm = agent_dir / 4.0
        carrying_key = 1.0 if self.unwrapped.carrying else 0.0
        door_is_open = 1.0 if door and door.is_open else 0.0

        engineered_obs = np.array(
            [
                rel_key_pos[0], rel_key_pos[1],
                rel_door_pos[0], rel_door_pos[1],
                agent_dir_norm, carrying_key, door_is_open,
            ],
            dtype=np.float32,
        )
        return engineered_obs

# --- 2. Define Constants ---
ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
MODEL_FILENAME = "ppo_doorkey_8x8.zip"
LOGS_DIR = "logs/"
TOTAL_TIMESTEPS = 100_000

# --- 3. Training the Agent ---
def train_agent():
    """
    Train the PPO agent using the custom EngineeredObsWrapper.
    """
    print("--- Starting Training (MLP Policy with Refined Rewards) ---")

    vec_env = make_vec_env(
        ENV_NAME,
        n_envs=4,
        wrapper_class=EngineeredObsWrapper
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=0.0003,
        ent_coef=0.01 # Encourage exploration
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_FILENAME)
    print(f"--- Training Complete. Model saved to {MODEL_FILENAME} ---")
    vec_env.close()

# --- 4. Visualizing the Agent ---
def visualize_agent():
    """
    Load the trained agent and watch it play.
    """
    print("\n--- Starting Visualization ---")
    
    env = gym.make(ENV_NAME, render_mode="human")
    env = EngineeredObsWrapper(env)

    model = PPO.load(MODEL_FILENAME)

    obs, _ = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # FIX: Access the new 'episode_reward' attribute from the wrapper
            print(f"Episode finished. Total shaped reward for episode: {env.episode_reward:.2f}")
            obs, _ = env.reset()
    
    print("--- Visualization Complete ---")
    env.close()

# --- 5. Evaluating the Agent ---
def evaluate_agent():
    """
    Load the trained agent and evaluate its performance.
    """
    print("\n--- Starting Evaluation ---")
    
    # We evaluate on the wrapped env to see the shaped reward performance
    eval_env = make_vec_env(ENV_NAME, n_envs=1, wrapper_class=EngineeredObsWrapper)

    model = PPO.load(MODEL_FILENAME)
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=100
    )
    
    print(f"--- Evaluation Complete ---")
    print(f"Mean shaped reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    eval_env.close()

# --- Main execution block ---
if __name__ == "__main__":
    # train_agent()
    evaluate_agent()
    visualize_agent()

