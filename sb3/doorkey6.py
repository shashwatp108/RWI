import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# --- 1. Define Constants ---
ENV_NAME = "MiniGrid-DoorKey-6x6-v0"  # <-- Changed
MODEL_FILENAME = "ppo_doorkey_6x6_mini.zip" # <-- Changed
LOGS_DIR = "logs/"
TOTAL_TIMESTEPS = 200_000 # Increased slightly for a slightly larger env

# behaved suboptimally at 85000 total timestamp

# --- 2. Training the Agent ---
def train_agent():
    """
    Train the PPO agent using the correct FlatObsWrapper and MlpPolicy.
    """
    print("--- Starting Training (MLP Policy) ---")

    vec_env = make_vec_env(
        ENV_NAME,
        n_envs=4,
        wrapper_class=FlatObsWrapper
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
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_FILENAME)
    print(f"--- Training Complete. Model saved to {MODEL_FILENAME} ---")
    vec_env.close()

# --- 3. Visualizing the Agent ---
def visualize_agent():
    """
    Load the trained agent and watch it play.
    """
    print("\n--- Starting Visualization ---")
    
    env = gym.make(ENV_NAME, render_mode="human")
    env = FlatObsWrapper(env)

    model = PPO.load(MODEL_FILENAME)

    obs, _ = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, _ = env.reset()
    
    print("--- Visualization Complete ---")
    env.close()

# --- 4. Evaluating the Agent ---
def evaluate_agent():
    """
    Load the trained agent and evaluate its performance.
    """
    print("\n--- Starting Evaluation ---")
    
    eval_env = gym.make(ENV_NAME)
    eval_env = FlatObsWrapper(eval_env)

    model = PPO.load(MODEL_FILENAME)
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=100
    )
    
    print(f"--- Evaluation Complete ---")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    eval_env.close()

# --- Main execution block ---
if __name__ == "__main__":
    train_agent()
    evaluate_agent()
    visualize_agent()