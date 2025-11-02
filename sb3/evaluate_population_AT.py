# evaluate_population_AT.py
"""
Driver to compute A (action-change sampling) and T (trajectory improvement)
where T compares candidate rollouts against the *ground-truth* original path
read from episode logs.

✅ NEW: This script now correctly determines the agent's inventory (has_key)
and the door's state (is_locked) at the critical state s_star and passes
this information to the simulation functions.

Usage example:
python evaluate_population_AT.py \
  --population populations/evaluated_sparsity/plausible_population_with_sparsity.json \
  --episodes-file episode_logs/episodes_full.json \
  --critical-states critical_states.json \
  --critical-index 9 \
  --model-file ppo_doorkey_6x6_mini_2.zip \
  --out-dir populations/evaluated_AT \
  --n_samples 20 \
  --top_k 30 \
  --n_rollouts 1 \
  --max_steps 200
"""
import os
import json
import argparse
import math
from phase2_at_helpers import (action_change_score,
                               rollout_trajectory_improvement,
                               build_env_from_grid_codes)
# We must import from find_cs.py to check key/door status
from find_cs import compute_has_key_sequence
from plausibility import find_tile_positions
from stable_baselines3 import PPO

# --- Constants for action enums (must match your environment) ---
ACTION_PICKUP = 3
ACTION_TOGGLE = 5

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def get_ground_truth_path_from_log(episode_record, critical_step_t):
    """
    Extracts the ground-truth path and its length (tile transitions)
    from the episode log, starting from the critical step t.
    """
    # The critical state s* is the *start* of our remaining path.
    start_pos = tuple(episode_record["steps"][critical_step_t]["pos"])
    
    # The rest of the path is all positions *after* t
    path_coords = [start_pos]
    for step in episode_record["steps"][critical_step_t + 1:]:
        path_coords.append(tuple(step["pos"]))

    # Add goal if successful
    if episode_record.get("success", False):
        try:
            grid = episode_record["initial_grid"]
            goal_pos = None
            for r, row in enumerate(grid):
                for c, cell in enumerate(row):
                    if cell == "Goal":
                        goal_pos = (c, r)
                        break
            if goal_pos and path_coords[-1] != goal_pos:
                path_coords.append(goal_pos)
        except Exception:
            pass # Goal finding failed, just use last recorded step
    
    # Calculate path length based on tile transitions
    transitions = 0
    for i in range(1, len(path_coords)):
        if path_coords[i] != path_coords[i-1]:
            transitions += 1
            
    return transitions, path_coords

def get_s_star_state(episode_record, critical_step_t):
    """
    ✅ NEW FUNCTION
    Determines the agent's inventory and door's state at the critical step t.
    Returns: (has_key_at_s, door_is_locked_at_s)
    """
    # 1. Check if agent has key
    has_key_seq = compute_has_key_sequence(episode_record)
    has_key_at_s = has_key_seq[critical_step_t] if critical_step_t < len(has_key_seq) else False

    # 2. Check if door was unlocked
    # By default, the door starts locked.
    door_is_locked_at_s = True 
    
    # Find the door's position from the initial grid
    door_pos_list = find_tile_positions(episode_record["initial_grid"], {"Door"})
    if not door_pos_list:
        return has_key_at_s, True # No door, so "locked" is fine
    
    door_pos = tuple(door_pos_list[0])
    
    # Check all steps *before* the critical state
    for i in range(critical_step_t):
        step = episode_record["steps"][i]
        agent_pos = tuple(step["pos"])
        agent_dir = step["dir"]
        action = step["action"]
        
        # Check if agent is facing the door
        dx, dy = 0, 0
        if agent_dir == 0: dx = 1  # Right
        elif agent_dir == 1: dy = 1 # Down
        elif agent_dir == 2: dx = -1 # Left
        elif agent_dir == 3: dy = -1 # Up
        
        front_pos = (agent_pos[0] + dx, agent_pos[1] + dy)

        # If agent toggled *while* facing the door *and* had the key, it's now unlocked
        if action == ACTION_TOGGLE and front_pos == door_pos and has_key_seq[i]:
            door_is_locked_at_s = False
            # Once unlocked, it stays unlocked
            break 
            
    return has_key_at_s, door_is_locked_at_s


def load_population_auto(population_arg_path):
    # (Same as your previous version)
    pop_dir = os.path.dirname(population_arg_path) or "."
    candidates = []
    possible_paths = [
        os.path.join(pop_dir, "plausible_population.json"),
        os.path.join(pop_dir, "evaluated", "plausible_population.json"),
        os.path.join(pop_dir, "evaluated_plausibility", "plausible_population.json"),
        population_arg_path
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                c = load_json(p)
                if isinstance(c, list) and len(c) > 0:
                    print(f"Using population file: {p} (n={len(c)})")
                    return c, p
            except Exception:
                continue
    # fallback
    pop = load_json(population_arg_path)
    return pop, population_arg_path

def main(args):
    pop, used_path = load_population_auto(args.population)
    episodes = load_json(args.episodes_file)
    crits = load_json(args.critical_states)
    if args.critical_index >= len(crits): raise IndexError("critical_index out of range")
    crit = crits[args.critical_index]

    ep_map = {ep['episode_id']: ep for ep in episodes}
    ep = ep_map[crit["episode_id"]]
    orig_grid = ep["initial_grid"] 
    s_star = crit["s_star"]
    
    # --- ✅ FIX: Compute ground-truth states at s_star ---
    critical_step_t = crit['critical_step_t']
    has_key_at_s, door_is_locked_at_s = get_s_star_state(ep, critical_step_t)
    print(f"--- Analyzing Critical State (Episode {crit['episode_id']}, t={critical_step_t}) ---")
    print(f"  State at s*: Agent Has Key = {has_key_at_s}, Door Is Locked = {door_is_locked_at_s}")
    # --- End Fix ---

    # --- FIX: Compute path_orig from the JSON log ---
    print(f"Reading ground-truth path from log...")
    path_orig_len, orig_path_coords = get_ground_truth_path_from_log(ep, critical_step_t)
    print(f"  Ground-truth path_orig = {path_orig_len} transitions.")
    # --- End Fix ---

    # load model (still needed for A and T rollouts on *candidates*)
    model = PPO.load(args.model_file)

    # 1) compute A for all candidates (cheap sampling)
    annotated = []
    print(f"Computing A (action-change sampling) on population (n={len(pop)}) ...")
    for i, cand in enumerate(pop):
        # --- ✅ FIX: Pass the correct states to the helper ---
        a_res = action_change_score(cand, s_star, model,
                                    env_name=args.env_name,
                                    n_samples=args.n_samples,
                                    deterministic=args.deterministic,
                                    threshold_binary=args.a_threshold,
                                    has_key_at_s=has_key_at_s,
                                    door_is_locked_at_s=door_is_locked_at_s)
        cand_copy = dict(cand)
        cand_copy['a_result'] = a_res
        annotated.append(cand_copy)
        if (i+1) % 50 == 0:
            print(f"  processed {i+1}/{len(pop)}")

    # Save intermediate A annotated population
    os.makedirs(args.out_dir, exist_ok=True)
    annotated_path = os.path.join(args.out_dir, "population_with_A.json")
    save_json(annotated, annotated_path)
    print(f"Saved A-annotated population to {annotated_path}")

    # 2) select top-K by a simple pre-score (plausibility, sparsity, a_score)
    def pre_score(entry):
        p = entry.get('plausibility_score') or 0.0
        s = entry.get('sparsity_score') or 0.0
        a = entry['a_result'].get('a_score') if entry['a_result'].get('ok') else 0.0
        return args.wP * p + args.wS * s + args.wA * a

    annotated_sorted = sorted(annotated, key=lambda e: pre_score(e), reverse=True)
    topk = annotated_sorted[:args.top_k]
    print(f"Selected top {len(topk)} candidates for rollouts (T).")

    # 3) compute T for top-K (compare to path_orig_len)
    final_list = []
    for i, cand in enumerate(topk):
        # --- ✅ FIX: Pass the correct states to the helper ---
        t_res = rollout_trajectory_improvement(cand, s_star, model, path_orig_len,
                                               env_name=args.env_name,
                                               n_rollouts=args.n_rollouts,
                                               max_steps=args.max_steps,
                                               deterministic=args.deterministic,
                                               has_key_at_s=has_key_at_s,
                                               door_is_locked_at_s=door_is_locked_at_s)
        cand2 = dict(cand)
        cand2['t_result'] = t_res
        final_list.append(cand2)
        print(f"  Rollout {i+1}/{len(topk)}: ok={t_res.get('ok')} T={t_res.get('T')} cf_len={t_res.get('cf_path_len_median')} (orig={path_orig_len}) elapsed={t_res.get('elapsed'):.3f}")

    final_path = os.path.join(args.out_dir, "topk_with_T.json")
    save_json(final_list, final_path)
    print(f"Saved top-K with T to {final_path}")

    # 4) Compute final fitness for top-K
    wA, wT, wS, wP = args.wA, args.wT, args.wS, args.wP
    scored = []
    for cand in final_list:
        a_score = cand['a_result'].get('a_score') if cand['a_result'].get('ok') else 0.0
        t_score = cand['t_result'].get('T') if cand['t_result'].get('ok') else 0.0
        s_score = cand.get('sparsity_score') or 0.0
        p_score = cand.get('plausibility_score') or 0.0
        fitness = wA*a_score + wT*t_score + wS*s_score + wP*p_score
        cand['final_fitness'] = float(fitness)
        scored.append((fitness, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    ranked_path = os.path.join(args.out_dir, "topk_ranked.json")
    save_json([c for _,c in scored], ranked_path)
    print(f"Saved ranked candidates to {ranked_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True)
    parser.add_argument("--episodes-file", default=os.path.join("episode_logs","episodes_full.json"))
    parser.add_argument("--critical-states", default="critical_states.json")
    parser.add_argument("--critical-index", type=int, default=0)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--out-dir", default="populations/evaluated_AT")
    parser.add_argument("--n_samples", type=int, default=20, help="policy samples for A")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--env-name", type=str, default="MiniGrid-DoorKey-6x6-v0")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--a_threshold", type=float, default=0.5)
    parser.add_argument("--n_rollouts", type=int, default=3, help="rollouts median for T")
    parser.add_argument("--deterministic", action='store_true', help="if set, use deterministic policy.predict")
    parser.add_argument("--wA", type=float, default=1.0)
    parser.add_argument("--wT", type=float, default=2.0)
    parser.add_argument("--wS", type=float, default=1.0)
    parser.add_argument("--wP", type=float, default=1.0)
    args = parser.parse_args()
    main(args)