# phase2_at_helpers.py
"""
Helpers for:
 - Action-change scoring (A)
 - Trajectory-improvement scoring (T) using original environment path as baseline (NO A*)

Key functions:
 - build_env_from_grid_codes(grid_codes, agent_pos, agent_dir, env_name, has_key, door_is_locked)
 - sample_policy_action_probs(env, model, n_samples)
 - action_change_score(candidate, s_star, model, n_samples, threshold_binary)
 - rollout_policy_and_get_path_length(env, model, max_steps, deterministic)
 - compute_path_len_from_grid(grid_codes, s_star, model, env_name, n_rollouts, max_steps, deterministic, has_key, door_is_locked)
 - rollout_trajectory_improvement(candidate, s_star, model, path_orig_median, env_name, n_rollouts, max_steps, deterministic, has_key, door_is_locked)

"""
import gymnasium as gym
import copy
import time
from typing import Dict, Tuple, List
from minigrid.wrappers import FlatObsWrapper

# Attempt to import Grid & world object classes; may need to adjust for your minigrid version.
try:
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import Goal, Key, Door, Wall
except Exception as e:
    Grid = None
    Goal = Key = Door = Wall = None
    # We'll raise helpful error later if Grid is missing when needed.

# Mapping from tile name used in your JSON -> constructor function / handler
_TILE_TO_CLASS = {
    "Wall": Wall,
    "Key": Key,
    "Door": Door,
    "Goal": Goal,
    # "empty" -> None
}

def build_env_from_grid_codes(grid_codes: List[List[str]], agent_pos: Tuple[int,int],
                              agent_dir: int, env_name: str = "MiniGrid-DoorKey-6x6-v0",
                              has_key: bool = False, door_is_locked: bool = True):
    """
    Construct a gym-minigrid env.
    
    ✅ NEW: Now correctly sets agent's inventory (has_key) and the
    door's locked state (door_is_locked) based on the s_star state.
    """
    if Grid is None:
        raise ImportError("minigrid Grid/world objects not importable. Ensure 'minigrid' is installed and importable.")

    H = len(grid_codes); W = len(grid_codes[0])
    # create environment and disable randomization by not using env.reset to generate objects
    env = gym.make(env_name)
    env = FlatObsWrapper(env)

    # create empty Grid of correct size
    grid = Grid(W, H)

    # Fill grid according to grid_codes
    for y in range(H):
        for x in range(W):
            tile = grid_codes[y][x]
            if tile == "empty":
                continue
            cls = _TILE_TO_CLASS.get(tile)
            if cls is None:
                continue
            try:
                if tile == "Door":
                    # --- ✅ FIX ---
                    # Use the door_is_locked state passed from the evaluation script
                    obj = Door('yellow', is_locked=door_is_locked)
                elif tile == "Key":
                    obj = Key('yellow')
                elif tile == "Goal":
                    obj = Goal()
                else:
                    obj = cls()
            except Exception:
                try:
                    obj = cls()
                except Exception as e:
                    raise RuntimeError(f"Cannot construct object for tile {tile}: {e}")
            grid.set(x, y, obj)

    # attach constructed grid and set agent loc/dir
    env.unwrapped.grid = grid
    env.unwrapped.agent_pos = tuple(agent_pos)
    env.unwrapped.agent_dir = int(agent_dir)

    # --- ✅ FIX ---
    # Correctly set the agent's inventory based on s_star
    if has_key:
        # We must create a Key object for the agent to be "carrying"
        # This assumes the key is 'yellow'.
        try:
            env.unwrapped.carrying = Key('yellow')
        except Exception:
            # Fallback if Key constructor fails (should not happen if imports work)
             env.unwrapped.carrying = "KEY_OBJECT_SENTINEL"
    else:
        env.unwrapped.carrying = None


    # Ensure the env builds its internal observations correctly
    try:
        # This reset *does* generate a new grid, but we fix it immediately after
        obs, info = env.reset()
    except Exception:
        _ = env.reset()

    # re-attach grid, agent pos, and inventory in case reset replaced them
    env.unwrapped.grid = grid
    env.unwrapped.agent_pos = tuple(agent_pos)
    env.unwrapped.agent_dir = int(agent_dir)
    if has_key:
        try:
            env.unwrapped.carrying = Key('yellow')
        except Exception:
             env.unwrapped.carrying = "KEY_OBJECT_SENTINEL"
    else:
        env.unwrapped.carrying = None

    return env

def sample_policy_action_probs(env, model, n_samples=20, deterministic=False):
    """
    Sample the (stochastic) policy n_samples times to estimate action probabilities.
    Returns dict {action: prob}.
    env: FlatObsWrapper env (position/dir already set).
    """
    # Do NOT call env.reset(). It regenerates the grid.
    # Instead, manually generate the observation for the *current* state.
    obs = env.observation(env.unwrapped.gen_obs())
    
    counts = {}
    for _ in range(n_samples):
        a, _ = model.predict(obs, deterministic=deterministic)
        a = int(a)
        counts[a] = counts.get(a, 0) + 1
    probs = {a: counts[a] / n_samples for a in counts}
    return probs

def action_change_score(candidate: Dict, s_star: Dict, model, env_name="MiniGrid-DoorKey-6x6-v0",
                        n_samples=20, deterministic=False, threshold_binary=0.5,
                        has_key_at_s: bool = False, door_is_locked_at_s: bool = True) -> Dict: # ✅ NEW PARAMS
    """
    Estimate change in action distribution at the s* position when the environment is modified.
    Returns diagnostics dict.
    """
    out = {"ok": False}
    start = time.time()

    grid_codes = candidate["grid"]
    agent_pos = tuple(s_star["pos"])
    agent_dir = int(s_star.get("dir", 0))
    orig_action = int(s_star.get("action"))

    try:
        # --- ✅ FIX: Pass the correct states to the builder ---
        env = build_env_from_grid_codes(grid_codes, agent_pos, agent_dir, env_name=env_name,
                                        has_key=has_key_at_s, door_is_locked=door_is_locked_at_s)
    except Exception as e:
        out["error"] = f"build_env_failed: {e}"
        return out

    try:
        probs = sample_policy_action_probs(env, model, n_samples=n_samples, deterministic=deterministic)
    except Exception as e:
        out["error"] = f"policy_sampling_failed: {e}"
        env.close()
        return out

    p_orig = probs.get(orig_action, 0.0)
    
    # find best alternative action (handle empty probs dict if sampling fails)
    if not probs:
        best_action, best_prob = orig_action, 0.0
    else:
        best_action, best_prob = max(probs.items(), key=lambda x: x[1])

    a_score = max(0.0, float(best_prob - p_orig))
    a_binary = (best_action != orig_action) or (p_orig < threshold_binary)

    out.update({
        "ok": True,
        "p_orig_action": float(p_orig),
        "best_alt_action": int(best_action),
        "best_alt_prob": float(best_prob),
        "a_score": float(a_score),
        "a_binary": bool(a_binary),
        "probs": probs,
        "elapsed": time.time() - start
    })
    env.close()
    return out

def rollout_policy_and_get_path_length(env, model, max_steps=200, deterministic=False) -> Tuple[int,bool,List[Tuple[int,int]]]:
    """
    Run the environment until termination or max_steps.
    Returns:
      - path_len: number of tile transitions (i.e., count of steps where position changed)
      - success: bool (whether goal reached)
      - positions: list of (x,y) positions visited
    deterministic: whether to use deterministic model.predict
    """
    # Do NOT call env.reset(). It regenerates the grid.
    # Instead, manually generate the observation for the *current* state.
    obs = env.observation(env.unwrapped.gen_obs())
    pos0 = tuple(env.unwrapped.agent_pos)
    
    positions = [ (int(pos0[0]), int(pos0[1])) ]
    success = False

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        pos_cur = tuple(env.unwrapped.agent_pos)
        positions.append( (int(pos_cur[0]), int(pos_cur[1])) )
        if terminated or truncated:
            final_obj = env.unwrapped.grid.get(*env.unwrapped.agent_pos)
            success = final_obj is not None and getattr(final_obj, "type", "").lower() == "goal"
            break

    # count transitions where position actually changed
    transitions = 0
    for i in range(1, len(positions)):
        if positions[i] != positions[i-1]:
            transitions += 1

    return transitions, success, positions

def compute_path_len_from_grid(grid_codes, s_star, model, env_name="MiniGrid-DoorKey-6x6-v0",
                               n_rollouts=3, max_steps=200, deterministic=False,
                               has_key_at_s: bool = False, door_is_locked_at_s: bool = True): # ✅ NEW PARAMS
    """
    Compute median path length (tile transitions) from s* for a given grid by performing n_rollouts
    and returning the median transitions value and diagnostics.
    """
    agent_pos = tuple(s_star["pos"])
    agent_dir = int(s_star.get("dir", 0))
    lengths = []
    successes = []
    positions_list = []

    for _ in range(n_rollouts):
        # --- ✅ FIX: Pass the correct states to the builder ---
        env = build_env_from_grid_codes(grid_codes, agent_pos, agent_dir, env_name=env_name,
                                        has_key=has_key_at_s, door_is_locked=door_is_locked_at_s)
        
        l, succ, pos = rollout_policy_and_get_path_length(env, model, max_steps=max_steps, deterministic=deterministic)
        env.close()
        lengths.append(l)
        successes.append(succ)
        positions_list.append(pos)

    # Only consider path lengths from *successful* rollouts
    successful_lengths = [lengths[i] for i in range(len(lengths)) if successes[i]]

    if not successful_lengths:
        # If no rollouts reached the goal, the path length is effectively "infinite"
        median_len = float('inf')
    else:
        # Otherwise, take the median of only the successful runs
        lengths_sorted = sorted(successful_lengths)
        median_len = lengths_sorted[len(lengths_sorted)//2]
    
    diagnostics = {"lengths": lengths, "successes": successes, "successful_lengths_considered": successful_lengths, "positions": positions_list}
    
    return int(median_len) if median_len != float('inf') else float('inf'), diagnostics


def rollout_trajectory_improvement(candidate: Dict, s_star: Dict, model,
                                   path_orig_median: int,
                                   env_name="MiniGrid-DoorKey-6x6-v0",
                                   n_rollouts=3, max_steps=200, deterministic=False,
                                   has_key_at_s: bool = False, door_is_locked_at_s: bool = True): # ✅ NEW PARAMS
    """
    Compute T-score for candidate by rolling out the policy on candidate grid (n_rollouts -> median cf_len)
    Compares to provided path_orig_median (computed once from original grid).
    """
    out = {"ok": False}
    start = time.time()

    grid_codes = candidate["grid"]

    # --- ✅ FIX: Pass the correct states to the compute function ---
    cf_len_median, diag = compute_path_len_from_grid(grid_codes, s_star, model,
                                                     env_name=env_name, n_rollouts=n_rollouts,
                                                     max_steps=max_steps, deterministic=deterministic,
                                                     has_key_at_s=has_key_at_s, door_is_locked_at_s=door_is_locked_at_s)

    T = 0.0
    if path_orig_median > 0 and cf_len_median < float('inf'):
        # Only calculate T if the original path wasn't 0 AND the candidate succeeded
        raw = (path_orig_median - cf_len_median) / float(path_orig_median)
        if raw > 0:
            T = float(max(0.0, min(1.0, raw)))
        # If raw is <= 0 (no improvement), T remains 0.0
    # If candidate failed (cf_len_median is inf), T remains 0.0
        
    out.update({
        "ok": True,
        "path_orig": int(path_orig_median),
        "cf_path_len_median": int(cf_len_median) if cf_len_median != float('inf') else -1, # Use -1 for JSON compatibility
        "T": float(T),
        "t_diag": diag,
        "elapsed": time.time() - start
    })
    return out