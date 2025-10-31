# phase2_at_helpers.py
"""
Helpers for Action-change (A) and Trajectory-improvement (T).

Functions:
 - build_env_from_grid_codes(grid_codes, agent_pos, agent_dir, env_name) -> env (FlatObsWrapper)
 - action_change_score(candidate, s_star_meta, model, env_name, n_samples=20, deterministic=False, threshold=0.5)
 - rollout_trajectory_improvement(candidate, s_star_meta, model, astar_fn, env_name, max_steps=200)

Notes:
 - grid_codes: 2D list of strings: "Wall","empty","Key","Door","Goal" (same as in your logs).
 - s_star_meta: the 's_star' object from critical_states (should include step, pos, dir, action, grid_snapshot)
 - astar_fn: import astar_min_steps from your find_cs.py and pass it in.
"""

import gymnasium as gym
import copy
import numpy as np
from minigrid.wrappers import FlatObsWrapper
import time
from typing import Tuple, Dict

# Try common imports for object classes (may need adjustment depending on installed gym_minigrid)
try:
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import Goal, Key, Door, Wall, Lava
except ImportError:
    print("Error: Failed to import Grid and World Objects from minigrid.core.")
    print("Please ensure the minigrid package is installed correctly.")
    Grid = None
    Goal = None
    Key = None
    Door = None
    Wall = None
    Lava = None

# mapping from string to class for object creation — you may need to adjust constructors
_TILE_TO_CLASS = {
    "Wall": Wall,
    "Key": Key,
    "Door": Door,
    "Goal": Goal,
    # "empty" -> None
}

def build_env_from_grid_codes(grid_codes, agent_pos, agent_dir, env_name="MiniGrid-DoorKey-6x6-v0"):
    """
    Construct a gym-minigrid env whose internal grid matches grid_codes and agent position/dir.
    Returns a FlatObsWrapper-wrapped env ready for model.predict(obs).
    NOTE: This uses Grid.set() and object constructors — if imports fail, adapt to your version.
    """
    if Grid is None:
        raise ImportError(
            "Couldn't import Grid / object classes. Please ensure gym_minigrid or minigrid is installed, "
            "and update import mapping in phase2_at_helpers.py."
        )

    H = len(grid_codes); W = len(grid_codes[0])
    env = gym.make(env_name)
    env = FlatObsWrapper(env)

    # create empty Grid
    grid = Grid(W, H)
    # fill with empty objects first (Grid uses None by default)
    # place objects according to grid_codes
    for y in range(H):
        for x in range(W):
            tile = grid_codes[y][x]
            if tile == "empty":
                continue
            cls = _TILE_TO_CLASS.get(tile)
            if cls is None:
                continue
            # Door / Key / Goal constructors often take different args.
            # We attempt the simplest constructors and catch any exceptions.
            try:
                if tile == "Door":
                    # Door may require color parameter; fallback to default
                    obj = Door(color='yellow', is_locked=False)
                elif tile == "Key":
                    obj = Key('yellow')
                elif tile == "Goal":
                    obj = Goal()
                else:
                    obj = cls()
            except Exception:
                # Try default no-arg
                try:
                    obj = cls()
                except Exception as e:
                    raise RuntimeError(f"Cannot construct object for tile {tile}: {e}")

            grid.set(x, y, obj)

    # attach the constructed grid
    env.unwrapped.grid = grid

    # set agent position and dir
    env.unwrapped.agent_pos = tuple(agent_pos)
    env.unwrapped.agent_dir = int(agent_dir)

    # build observation by resetting with the same configuration (no randomization)
    # Some minigrid versions require env.reset() to build internal state; we'll reset and then override grid again
    try:
        obs, info = env.reset()
    except Exception:
        # older gym style
        _ = env.reset()

    # ensure grid & agent pos/dir persist
    env.unwrapped.grid = grid
    env.unwrapped.agent_pos = tuple(agent_pos)
    env.unwrapped.agent_dir = int(agent_dir)

    return env

def sample_policy_action_probs(env, model, n_samples=20):
    """
    Estimate action frequency distribution by sampling the (stochastic) policy n_samples times.
    Returns dict {action: prob}.
    env should be ready so calling model.predict(obs) works.
    """
    obs, _ = env.reset() if True else env.reset()  # reset but preserve grid and agent_pos
    # We must ensure obs corresponds to current state (some envs require env.reset to compute obs)
    # To be safe, call model.predict on the SAME obs repeatedly.
    counts = {}
    for _ in range(n_samples):
        action, _ = model.predict(obs, deterministic=False)
        a = int(action)
        counts[a] = counts.get(a, 0) + 1
    probs = {a: counts[a] / n_samples for a in counts}
    return probs

def action_change_score(candidate: Dict, s_star: Dict, model, env_name="MiniGrid-DoorKey-6x6-v0",
                        n_samples=20, orig_action=None, threshold_binary=0.5) -> Dict:
    """
    Returns a dictionary with:
    - p_orig_action (float)
    - best_alt_action (int)
    - best_alt_prob (float)
    - a_score (float in [0,1]) = max(0, best_alt_prob - p_orig_action)
    - a_binary (bool): whether best_alt_action != orig_action OR p_orig_action < threshold_binary
    Also returns diagnostics and timing.
    """
    out = {"ok": False}
    start = time.time()
    grid_codes = candidate["grid"]
    # s_star contains pos, dir and original action. Use orig_action param to override
    agent_pos = s_star["pos"]
    agent_dir = s_star.get("dir", 0)
    if orig_action is None:
        orig_action = s_star.get("action")
    # Ensure it's a standard python int
    orig_action = int(orig_action)

    try:
        env = build_env_from_grid_codes(grid_codes, agent_pos, agent_dir, env_name=env_name)
    except Exception as e:
        out["error"] = f"build_env_failed: {e}"
        return out

    probs = sample_policy_action_probs(env, model, n_samples=n_samples)
    p_orig = probs.get(int(orig_action), 0.0)
    # find best alternative action
    best_action = max(probs.items(), key=lambda x: x[1])[0]
    best_prob = probs[best_action]

    a_score = max(0.0, float(best_prob - p_orig))
    a_binary = (best_action != int(orig_action)) or (p_orig < threshold_binary)

    out.update({
        "ok": True,
        "p_orig_action": float(p_orig),
        "best_alt_action": int(best_action),
        "best_alt_prob": float(best_prob),
        "a_score": a_score,
        "a_binary": bool(a_binary),
        "probs": probs,
        "elapsed": time.time() - start
    })
    env.close()
    return out

# Replace existing rollout_policy_and_get_path_length and rollout_trajectory_improvement
def rollout_policy_and_get_path_length(env, model, max_steps=200):
    """
    Run the environment until termination or max_steps.
    Returns:
      - path_len (number of position transitions: len(positions)-1)
      - success (bool)
      - positions (list of (x,y) lists)
    This counts tile moves (not forward-action counts) so it matches A*/path lengths.
    """
    # Reset just to ensure observation matches the env's current grid & agent_pos
    obs, _ = env.reset()
    # capture initial position as standard tuple/list
    pos0 = tuple(env.unwrapped.agent_pos)
    positions = [ (int(pos0[0]), int(pos0[1])) ]
    success = False

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=False)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)

        pos_cur = tuple(env.unwrapped.agent_pos)
        # append the current position after the action step (tile position)
        positions.append((int(pos_cur[0]), int(pos_cur[1])))

        if terminated or truncated:
            # check goal
            final_obj = env.unwrapped.grid.get(*env.unwrapped.agent_pos)
            success = final_obj is not None and getattr(final_obj, "type", "").lower() == "goal"
            break

    # path length measured as number of tile transitions (unique successive moves)
    # count transitions where position actually changed
    transitions = 0
    for i in range(1, len(positions)):
        if positions[i] != positions[i-1]:
            transitions += 1

    return transitions, success, positions


def rollout_trajectory_improvement(candidate: Dict, s_star: Dict, model, astar_min_steps_fn,
                                   env_name="MiniGrid-DoorKey-6x6-v0", max_steps=200):
    """
    For candidate starting from s*, compute two-stage orig_len using get_true_optimal_path (if available),
    then roll out the policy and compute T = (orig_len - cf_len) / orig_len where
    cf_len is the number of tile transitions in the rollout.
    """
    out = {"ok": False}
    start = time.time()

    grid_codes = candidate["grid"]
    agent_pos = tuple(s_star["pos"])
    agent_dir = int(s_star.get("dir", 0))

    # Prefer get_true_optimal_path if it's provided in the astar module,
    # otherwise fall back to astar_min_steps (less accurate for DoorKey).
    orig_len = None
    try:
        # try to use two-stage helper (get_true_optimal_path)
        from find_cs import get_true_optimal_path
        orig_len, _ = get_true_optimal_path(grid_codes, agent_pos)
    except Exception:
        try:
            orig_len, _ = astar_min_steps_fn(grid_codes, agent_pos, has_key_start=False)
        except Exception as e:
            out["error"] = f"astar_failed: {e}"
            return out

    if orig_len is None or orig_len == float("inf"):
        out["error"] = "orig_infinite"
        return out

    # Build env and run rollout
    try:
        env = build_env_from_grid_codes(grid_codes, agent_pos, agent_dir, env_name=env_name)
    except Exception as e:
        out["error"] = f"build_env_failed: {e}"
        return out

    cf_len, success, positions = rollout_policy_and_get_path_length(env, model, max_steps=max_steps)

    # Compute improvement (if rollout reached shorter path)
    T = 0.0
    try:
        if cf_len < orig_len:
            T = float((orig_len - cf_len) / float(orig_len))
            T = max(0.0, min(1.0, T))
        elif cf_len == orig_len:
            # reaching the same optimal path is considered full improvement
            T = 1.0
    except Exception:
        T = 0.0

    out.update({
        "ok": True,
        "orig_remaining_len": float(orig_len),
        "cf_path_len": float(cf_len),
        "success": bool(success),
        "T": float(T),
        "positions": positions,
        "elapsed": time.time() - start
    })
    env.close()
    return out