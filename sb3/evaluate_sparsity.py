# evaluate_population_sparsity.py
"""
Driver for computing sparsity S-term for a population JSON.
Automatically uses plausible_population.json if available.

python evaluate_sparsity.py \
  --population populations/population_crit0_pop100000.json \
  --episodes-file episode_logs/episodes_full.json \
  --critical-states critical_states.json \
  --critical-index 0 \
  --out-dir populations/evaluated_sparsity \
  --top-k 100
"""
import argparse
import os
import json
from sparsity import sparsity_score, find_tile_positions
from typing import Dict, Tuple

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def fetch_original_info(episode_record: Dict) -> Tuple[list, Tuple[int,int], Dict[str,Tuple[int,int]]]:
    orig_grid = episode_record["initial_grid"]
    orig_agent_pos = tuple(episode_record["initial_agent_pos"])
    kpos = find_tile_positions(orig_grid, {"Key"})
    dpos = find_tile_positions(orig_grid, {"Door"})
    gpos = find_tile_positions(orig_grid, {"Goal"})
    special = {
        "Key": tuple(kpos[0]) if kpos else None,
        "Door": tuple(dpos[0]) if dpos else None,
        "Goal": tuple(gpos[0]) if gpos else None
    }
    return orig_grid, orig_agent_pos, special

# --- MODIFIED --- New function to auto-detect plausible population
def load_population_auto(population_arg_path, episodes_path, criticals_path, critical_index):
    """
    Load population: prefer plausible_population.json (from previous plausibility stage).
    Fallback to provided population_arg_path if plausible not available or empty.
    """
    # Note: This assumes plausible_population.json is in a subdir named after the out-dir of the plausibility script.
    # A more robust implementation might take the plausibility output dir as an argument.
    # Per instructions, we look in the same directory as the input population file's parent structure.
    # e.g., if input is `populations/population_crit0_pop120.json`, we check `populations/evaluated_plausibility/plausible_population.json`
    # The user's instructions imply a simpler structure, we will follow that for now.
    pop_dir = os.path.dirname(population_arg_path) or "."
    
    # A common output path from the plausibility script would be in a subdirectory
    plausible_path = os.path.join(pop_dir, "evaluated_plausibility", "plausible_population.json") # A likely path
    if not os.path.exists(plausible_path):
        plausible_path = os.path.join(pop_dir, "evaluated", "plausible_population.json") # Another likely path
    if not os.path.exists(plausible_path):
        plausible_path = os.path.join(pop_dir, "plausible_population.json") # Path from user instructions

    if os.path.exists(plausible_path):
        try:
            pop_plaus = load_json(plausible_path)
            if isinstance(pop_plaus, list) and len(pop_plaus) > 0:
                print(f"✅ Using plausible population: {plausible_path} (size={len(pop_plaus)})")
                return pop_plaus, plausible_path
            else:
                print(f"⚠️ Found plausible_population.json but it's empty. Falling back to full population...")
        except Exception as e:
            print(f"⚠️ Error loading plausible_population.json: {e}. Falling back to full population.")
    
    print(f"✅ Using full population file: {population_arg_path}")
    pop = load_json(population_arg_path)
    return pop, population_arg_path


def main(args):
    # --- MODIFIED --- Replaced direct loading with the auto-detection function
    pop, used_path = load_population_auto(args.population, args.episodes_file, args.critical_states, args.critical_index)
    
    episodes = load_json(args.episodes_file)
    ep_map = {ep['episode_id']: ep for ep in episodes}
    criticals = load_json(args.critical_states)
    
    if args.critical_index >= len(criticals):
        raise IndexError("critical_index out of range")
    crit = criticals[args.critical_index]
    ep_id = crit["episode_id"]
    if ep_id not in ep_map:
        raise KeyError(f"episode_id {ep_id} not found in episodes file")
    
    episode_record = ep_map[ep_id]
    orig_grid, _, _ = fetch_original_info(episode_record)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- MODIFIED --- Dynamic output filenames based on the input used
    base_name = os.path.splitext(os.path.basename(used_path))[0]
    annotated_path = os.path.join(out_dir, f"{base_name}_with_sparsity.json")
    topk_path = os.path.join(out_dir, f"{base_name}_topk_by_sparsity.json")

    annotated = []
    scored = []
    cfg = {
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "w_spec": args.w_spec
    }

    for idx, cand in enumerate(pop):
        score, diag = sparsity_score(cand, orig_grid, config=cfg)
        ccopy = dict(cand)
        ccopy['sparsity_score'] = score
        ccopy['sparsity_diag'] = diag
        annotated.append(ccopy)
        scored.append((score, idx, ccopy))

    save_json(annotated, annotated_path)
    scored.sort(key=lambda x: x[0], reverse=True)
    topk = [item[2] for item in scored[:args.top_k]]
    save_json(topk, topk_path)

    print(f"\nAnnotated population saved: {annotated_path}")
    print(f"Top {args.top_k} by sparsity saved: {topk_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True, help="Path to the full population json (used as fallback)")
    parser.add_argument("--episodes-file", default=os.path.join("episode_logs","episodes_full.json"), help="Full episodes file")
    parser.add_argument("--critical-states", default="critical_states.json")
    parser.add_argument("--critical-index", type=int, default=0)
    parser.add_argument("--out-dir", default="populations/evaluated_sparsity")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--w-spec", type=float, default=1.5)
    args = parser.parse_args()
    main(args)