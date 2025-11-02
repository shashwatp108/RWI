"""
init_population.py

Phase 2 Part 1:
- Load a critical state from critical_states.json
- Create an initial population of full-environment chromosomes (2D grid)
- Each chromosome is a dict with 'grid' (2D list of tile names) and metadata
- Apply random edits to the original grid.
- NOTE: Plausibility checks have been REMOVED from this version.

Usage:
    python init_population.py --critical-index 9 --pop-size 100000 --mode exploratory
"""

import json
import random
import copy
import argparse
import os
from typing import List, Tuple, Dict

# Try to import tile sets from your find_cs module
try:
    from find_cs import WALLS, DOORS, KEYS, GOALS
except Exception:
    # Fallback definitions if import fails
    print("Warning: could not import from find_cs; using basic tile definitions.")
    WALLS = {"Wall"}
    DOORS = {"Door"}
    KEYS = {"Key"}
    GOALS = {"Goal"}

# ---- Configuration / hyperparameters ----
DEFAULT_POP_SIZE = 120
DEFAULT_MAX_EDITS = 3
RANDOM_SEED = 42
DEFAULT_MODE = "conservative"

# ---- Helpers ----

def load_json(fname: str):
    with open(fname, "r") as f:
        return json.load(f)

def save_json(obj, fname: str):
    with open(fname, "w") as f:
        json.dump(obj, f, indent=2)

def find_tile_positions(grid: List[List[str]], tile_set: set) -> List[Tuple[int,int]]:
    """Return list of (x,y) positions where grid[y][x] is in tile_set."""
    pos = []
    H = len(grid); W = len(grid[0])
    for y in range(H):
        for x in range(W):
            if grid[y][x] in tile_set:
                pos.append((x,y))
    return pos

# ---- Candidate generation utils ----

def copy_grid(grid):
    return [row[:] for row in grid]

def random_interior_positions(grid):
    H = len(grid); W = len(grid[0])
    positions = []
    for y in range(1, H-1):
        for x in range(1, W-1):
            positions.append((x,y))
    random.shuffle(positions)
    return positions

def flip_tile(grid: List[List[str]], pos: Tuple[int,int]):
    """Flip between 'empty' and 'Wall' at pos. (Conservative edit)"""
    x,y = pos
    cur = grid[y][x]
    if cur == "empty":
        grid[y][x] = "Wall"
    elif cur == "Wall":
        grid[y][x] = "empty"
    else:
        # if current is special (Key/Door/Goal), leave unchanged
        pass

# def random_edit_conservative(grid: List[List[str]], max_edits=3, forbidden_positions=set()):
#     """
#     Conservative edits: flip empty<->Wall at random interior positions excluding forbidden_positions.
#     Returns new grid (copy).
#     """
#     new_grid = copy_grid(grid)
#     positions = random_interior_positions(new_grid)
#     edits = random.randint(1, max(1, max_edits))
#     edits_done = 0
#     for pos in positions:
#         if edits_done >= edits:
#             break
#         if pos in forbidden_positions:
#             continue
        
#         if new_grid[pos[1]][pos[0]] in ("empty", "Wall"):
#             flip_tile(new_grid, pos)
#             edits_done += 1
#     return new_grid

def random_edit_conservative(grid: List[List[str]], max_edits=3, forbidden_positions=set()):
    """
    Conservative edits: either flips an empty<->wall tile OR moves the door.
    """
    new_grid = copy_grid(grid)
    
    # Get all possible locations for edits (interior, not forbidden)
    H, W = len(grid), len(grid[0])
    all_editable_positions = [
        (x, y) for y in range(1, H - 1) for x in range(1, W - 1)
        if (x, y) not in forbidden_positions
    ]
    
    if not all_editable_positions:
        return new_grid # Cannot perform any edits

    num_edits = random.randint(1, max(1, max_edits))

    for _ in range(num_edits):
        # Randomly choose to either flip a tile or move the door
        edit_type = random.choice(['flip', 'move_door'])

        if edit_type == 'flip':
            pos_to_flip = random.choice(all_editable_positions)
            # Flip only if it's a standard empty/wall tile
            if new_grid[pos_to_flip[1]][pos_to_flip[0]] in ("empty", "Wall"):
                flip_tile(new_grid, pos_to_flip)
        
        elif edit_type == 'move_door':
            # 1. Find the current door position in the potentially modified grid
            door_pos_list = find_tile_positions(new_grid, DOORS)
            if not door_pos_list: continue # Safeguard if door is missing
            current_door_pos = door_pos_list[0]

            # 2. Find a new valid destination that isn't the current door position
            possible_destinations = [p for p in all_editable_positions if p != current_door_pos]
            if not possible_destinations: continue # No valid place to move the door
            
            new_door_pos = random.choice(possible_destinations)

            # 3. Swap the door with the tile at the new destination
            (x1, y1), (x2, y2) = current_door_pos, new_door_pos
            new_grid[y1][x1], new_grid[y2][x2] = new_grid[y2][x2], new_grid[y1][x1]

    return new_grid

def random_edit_exploratory(grid: List[List[str]], max_edits=3, forbidden_positions=set()):
    """
    Exploratory edits: allow swapping special tiles (Key/Door/Goal) with empty cells,
    or flipping empty<->Wall.
    """
    new_grid = copy_grid(grid)
    H = len(grid); W = len(grid[0])
    all_interior = [(x,y) for y in range(1,H-1) for x in range(1,W-1) if (x,y) not in forbidden_positions]
    if not all_interior: return new_grid # Cannot edit if no valid positions

    edits = random.randint(1, max(1, max_edits))
    for _ in range(edits):
        op = random.choice(["flip", "swap_special"])
        if op == "flip":
            pos = random.choice(all_interior)
            if new_grid[pos[1]][pos[0]] in ("empty","Wall"):
                flip_tile(new_grid, pos)
        else:
            # pick a special tile to move (Key/Door/Goal)
            special_positions = [(x,y) for y in range(1,H-1) for x in range(1,W-1) if new_grid[y][x] in ("Key","Door")]
            if not special_positions:
                continue # No special tiles to swap
            
            sp = random.choice(special_positions)
            dest = random.choice(all_interior)
            # swap
            new_grid[sp[1]][sp[0]], new_grid[dest[1]][dest[0]] = new_grid[dest[1]][dest[0]], new_grid[sp[1]][sp[0]]
    return new_grid

# ---- Plausibility check functions have been REMOVED ----

# ---- Population initialization ----

def build_population_from_critical(critical_state_obj: dict, episode_record: dict,
                                   pop_size=DEFAULT_POP_SIZE, max_edits=DEFAULT_MAX_EDITS,
                                   mode=DEFAULT_MODE, seed=RANDOM_SEED):
    """
    Returns a list of candidate chromosomes. Plausibility is NOT checked.
    """
    random.seed(seed)
    population = []

    initial_grid = episode_record["initial_grid"]
    agent_pos = tuple(episode_record["initial_agent_pos"])
    
    # Original chromosome (unmodified) MUST be included
    orig_chrom = {
        "grid": copy_grid(initial_grid),
        "agent_pos": agent_pos,
        "meta": {
            "source_episode": episode_record["episode_id"],
            "type": "original"
        }
    }
    population.append(orig_chrom)

    # Determine positions that should not be edited
    forbidden = set()
    forbidden.add(agent_pos)
    
    goal_pos = find_tile_positions(initial_grid, GOALS)[0]
    forbidden.add(goal_pos)
    # In conservative mode, also protect key, door, goal, and outer walls
    if mode == 'conservative':
        key_pos = find_tile_positions(initial_grid, KEYS)[0]
        door_pos = find_tile_positions(initial_grid, DOORS)[0]
        forbidden.add(key_pos)
        # forbidden.add(door_pos)
        H = len(initial_grid); W = len(initial_grid[0])
        for y in range(H):
            forbidden.add((0,y)); forbidden.add((W-1,y))
        for x in range(W):
            forbidden.add((x,0)); forbidden.add((x,H-1))

    # Generate remaining individuals without checking for plausibility
    for i in range(pop_size - 1):
        if mode == "conservative":
            cand_grid = random_edit_conservative(initial_grid, max_edits=max_edits, forbidden_positions=forbidden)
        else: # exploratory
            cand_grid = random_edit_exploratory(initial_grid, max_edits=max_edits, forbidden_positions=forbidden)

        # Add the candidate directly to the population
        population.append({
            "grid": cand_grid,
            "agent_pos": agent_pos,
            "meta": {
                "source_episode": episode_record["episode_id"],
                "type": "random_seeded",
                "seed_id": i,
                "edits": mode
            }
        })

    return population

# ---- CLI / Runner ----

def main(args):
    # load files
    crit_file = args.critical_states if args.critical_states else "critical_states.json"
    episodes_file = args.episodes_file if args.episodes_file else os.path.join("episode_logs","episodes_full.json")
    out_dir = "populations"
    os.makedirs(out_dir, exist_ok=True)
    out_file = args.output if args.output else os.path.join(out_dir, f"population_crit{args.critical_index}_pop{args.pop_size}.json")

    if not os.path.exists(crit_file) or not os.path.exists(episodes_file):
        raise FileNotFoundError(f"Ensure '{crit_file}' and '{episodes_file}' exist.")

    criticals = load_json(crit_file)
    episodes = load_json(episodes_file)
    ep_map = {ep['episode_id']: ep for ep in episodes}
    if args.critical_index >= len(criticals):
        raise IndexError("critical_index out of range")

    target_crit = criticals[args.critical_index]
    ep_id = target_crit["episode_id"]
    if ep_id not in ep_map:
        raise KeyError(f"Episode id {ep_id} from critical state not found in episodes file.")

    episode_record = ep_map[ep_id]
    pop = build_population_from_critical(target_crit, episode_record,
                                        pop_size=args.pop_size, max_edits=args.max_edits,
                                        mode=args.mode, seed=args.seed)

    save_json(pop, out_file)
    print(f"✅ Saved population (size={len(pop)}) to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an initial population of environments based on a critical state.")
    parser.add_argument("--critical-index", type=int, default=0, help="Index of the critical state in critical_states.json")
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE, help="Total size of the population to generate.")
    parser.add_argument("--max-edits", type=int, default=DEFAULT_MAX_EDITS, help="Maximum number of random edits per individual.")
    parser.add_argument("--mode", choices=["conservative","exploratory"], default=DEFAULT_MODE, help="Editing mode.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--critical-states", type=str, default="critical_states.json", help="Path to the critical states file.")
    parser.add_argument("--episodes-file", type=str, default=os.path.join("episode_logs","episodes_full.json"), help="Path to the full episodes log file.")
    parser.add_argument("--output", type=str, default=None, help="Custom output file path.")
    args = parser.parse_args()
    main(args)