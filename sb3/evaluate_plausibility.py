# evaluate_plausibility.py
"""
Driver script:
- Loads population JSON and episodes
- Filters out duplicate chromosomes to ensure uniqueness.
- Computes plausibility score for each unique candidate.
- Writes annotated population and plausible population files.

Usage:
python evaluate_plausibility.py --population populations/population_crit9_pop100000.json --out-dir populations/evaluated --threshold 0.75

"""
import argparse
import os
import json
from plausibility import plausibility_score, find_tile_positions
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

def main(args):
    pop = load_json(args.population)
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
    orig_grid, orig_agent_pos, orig_special = fetch_original_info(episode_record)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    annotated_path = os.path.join(out_dir, "population_with_plausibility.json")
    plausible_path = os.path.join(out_dir, "plausible_population.json")

    annotated = []
    plausible = []
    threshold = args.threshold

    # --- NEW --- Set to store hashes of grids we have already processed
    seen_grids = set()
    duplicates_found = 0

    for idx, cand in enumerate(pop):
        # --- NEW --- Convert grid to a hashable type (tuple of tuples)
        grid_tuple = tuple(map(tuple, cand["grid"]))

        # --- NEW --- If we've seen this grid before, skip it
        if grid_tuple in seen_grids:
            duplicates_found += 1
            continue

        # --- NEW --- If it's a new grid, add it to the set for future checks
        seen_grids.add(grid_tuple)

        # Proceed with plausibility scoring for this unique candidate
        score, diag = plausibility_score(cand, orig_grid, orig_agent_pos, orig_special)
        
        ccopy = dict(cand)
        ccopy['plausibility_score'] = score
        ccopy['plausibility_diag'] = diag
        annotated.append(ccopy)
        if score >= threshold:
            plausible.append(ccopy)

    save_json(annotated, annotated_path)
    save_json(plausible, plausible_path)

    print(f"\nProcessed {len(pop)} initial chromosomes.")
    print(f"Skipped {duplicates_found} duplicate chromosomes.")
    print(f"Saved {len(annotated)} unique annotated chromosomes to: {annotated_path}")
    print(f"Saved {len(plausible)} unique and plausible chromosomes to: {plausible_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True, help="Path to population json")
    parser.add_argument("--episodes-file", default=os.path.join("episode_logs","episodes_full.json"), help="Full episodes log")
    parser.add_argument("--critical-states", default="critical_states.json", help="critical_states.json")
    parser.add_argument("--critical-index", type=int, default=0, help="index in critical_states.json")
    parser.add_argument("--out-dir", default="populations/evaluated", help="output dir")
    parser.add_argument("--threshold", type=float, default=0.9, help="plausibility threshold to save plausible population")
    args = parser.parse_args()
    main(args)