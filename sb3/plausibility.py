# plausibility.py
"""
Plausibility scoring utilities for Phase-2 fitness function (P term).

Functions:
- plausibility_score(candidate, original_grid, original_agent_pos, original_special_positions)
Returns: (score_float_0_1, diagnostics_dict)

Relies on find_cs.py for astar_min_steps and (optionally) get_true_optimal_path.
"""
from typing import List, Tuple, Dict
import math

# Try to import helper A* from your find_cs module
try:
    from find_cs import astar_min_steps, get_true_optimal_path, WALLS, DOORS, KEYS, GOALS
except Exception:
    # Fallback simple definitions if not present. You should normally have find_cs.py available.
    print("Warning: find_cs import failed in plausibility.py. Make sure find_cs.py is present.")
    def astar_min_steps(grid, start_pos, has_key_start=False, override_goal=None):
        return float("inf"), None
    def get_true_optimal_path(grid, start_pos):
        return float("inf"), None
    WALLS = {"Wall"}
    DOORS = {"Door"}
    KEYS = {"Key"}
    GOALS = {"Goal"}

def find_tile_positions(grid: List[List[str]], tile_set: set) -> List[Tuple[int,int]]:
    pos = []
    H, W = len(grid), len(grid[0])
    for y in range(H):
        for x in range(W):
            if grid[y][x] in tile_set:
                pos.append((x,y))
    return pos

def is_outer_wall_intact(grid: List[List[str]]) -> bool:
    H, W = len(grid), len(grid[0])
    for x in range(W):
        if grid[0][x] != "Wall" or grid[H-1][x] != "Wall": return False
    for y in range(H):
        if grid[y][0] != "Wall" or grid[y][W-1] != "Wall": return False
    return True

def single_inner_wall_column_with_door(grid: List[List[str]]) -> Tuple[bool,int]:
    H, W = len(grid), len(grid[0])
    inner_candidates = []
    for col in range(1, W-1):
        wall_count = sum(1 for row in range(H) if grid[row][col] == "Wall")
        door_count = sum(1 for row in range(H) if grid[row][col] in DOORS)
        if wall_count >= (H-1) and door_count >= 1:
            inner_candidates.append(col)
    if len(inner_candidates) == 1:
        return True, inner_candidates[0]
    return False, -1

# --- MODIFIED HELPER FUNCTION ---
def check_for_floating_walls(grid: List[List[str]]) -> bool:
    """
    Checks the entire grid for any 'Wall' tile with an 'empty' tile directly above or below it.
    This is a hard constraint.
    Returns True if the grid is valid (no floating walls), False otherwise.
    """
    H, W = len(grid), len(grid[0])
    # Iterate through all internal tiles
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if grid[y][x] == "Wall":
                # Check for empty space directly above or below
                if grid[y-1][x] == "empty" or grid[y+1][x] == "empty":
                    return False # Invalid grid
    return True # Grid is valid

def check_agent_to_key_reachable(grid: List[List[str]], agent_pos: Tuple[int,int]) -> Tuple[bool, str]:
    # ... (function remains the same)
    key_pos_list = find_tile_positions(grid, KEYS)
    if not key_pos_list: return False, "no_key"
    key_pos = key_pos_list[0]
    H, W = len(grid), len(grid[0])
    pickup_spots = []
    for dx, dy in [(1,0),(-1,0),(0,1),(-1,0)]:
        nx, ny = key_pos[0] + dx, key_pos[1] + dy
        if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] != "Wall":
            pickup_spots.append((nx, ny))
    if not pickup_spots: return False, "key_boxed"
    for spot in pickup_spots:
        dist, _ = astar_min_steps(grid, agent_pos, has_key_start=False, override_goal=spot)
        if not math.isinf(dist): return True, "ok"
    return False, "no_path_to_key_without_door"

def check_full_solvable(grid: List[List[str]], agent_pos: Tuple[int,int]) -> Tuple[bool, str]:
    # ... (function remains the same)
    try:
        total_len, _ = get_true_optimal_path(grid, agent_pos)
        if not math.isinf(total_len): return True, "ok"
        else: return False, "no_full_path"
    except Exception:
        key_pos_list = find_tile_positions(grid, KEYS)
        goal_pos_list = find_tile_positions(grid, GOALS)
        if not key_pos_list or not goal_pos_list: return False, "no_key_or_goal"
        key_pos = key_pos_list[0]; goal_pos = goal_pos_list[0]
        H, W = len(grid), len(grid[0])
        pickup_spots = []
        for dx, dy in [(1,0),(-1,0),(0,1),(-1,0)]:
            nx, ny = key_pos[0] + dx, key_pos[1] + dy
            if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] != "Wall": pickup_spots.append((nx, ny))
        if not pickup_spots: return False, "key_boxed"
        for spot in pickup_spots:
            d1, _ = astar_min_steps(grid, agent_pos, has_key_start=False, override_goal=spot)
            if math.isinf(d1): continue
            d2, _ = astar_min_steps(grid, spot, has_key_start=True)
            if not math.isinf(d2): return True, "ok"
        return False, "no_full_path_composed"

def plausibility_score(candidate: Dict, original_grid: List[List[str]],
                       original_agent_pos: Tuple[int,int], original_special_positions: Dict[str,Tuple[int,int]]) -> Tuple[float, Dict]:
    diag = {}
    grid = candidate["grid"]
    agent_pos = tuple(candidate.get("agent_pos", original_agent_pos))

    # --- NEW LOGIC: HARD CONSTRAINT CHECK ---
    # Check for floating walls first. If found, fail immediately.
    diag['floating_walls_ok'] = check_for_floating_walls(grid)
    if not diag['floating_walls_ok']:
        diag['final_score'] = 0.0
        diag['reason'] = "Failed hard constraint: floating wall detected."
        return 0.0, diag

    # --- MODIFIED --- Weight for floating walls is removed as it's now a hard constraint
    weights = {
        "agent_unchanged": 1.5,
        "goal_unchanged": 1.0,
        "counts": 1.0,
        "outer_walls": 1.5,
        "not_in_wall": 1.0,
        "distinct": 0.5,
        "agent_to_key": 2.0,
        "full_solve": 2.5,
        "single_inner_wall": 1.0,
        "door_same": 1.0,
    }
    total_weight = sum(weights.values())

    pass_agent_unchanged = (agent_pos == tuple(original_agent_pos))
    diag['agent_unchanged'] = pass_agent_unchanged

    # ... (rest of the checks remain the same)
    cand_goals = find_tile_positions(grid, GOALS)
    orig_goal = original_special_positions.get("Goal")
    pass_goal_unchanged = (len(cand_goals) == 1 and tuple(cand_goals[0]) == tuple(orig_goal))
    diag['goal_unchanged'] = pass_goal_unchanged

    cand_keys = find_tile_positions(grid, KEYS)
    cand_doors = find_tile_positions(grid, DOORS)
    counts_ok = (len(cand_keys) == 1 and len(cand_doors) == 1 and len(cand_goals) == 1)
    diag['counts_ok'] = counts_ok

    pass_outer_walls = is_outer_wall_intact(grid)
    diag['outer_walls'] = pass_outer_walls

    H, W = len(grid), len(grid[0])
    def tile_at(pos): return grid[pos[1]][pos[0]]
    pass_not_in_wall = True
    if not (0 <= agent_pos[0] < W and 0 <= agent_pos[1] < H) or tile_at(agent_pos) == "Wall":
        pass_not_in_wall = False
    for pos in [p[0] if p else None for p in [cand_keys, cand_doors, cand_goals]]:
        if pos is None or tile_at(pos) == "Wall":
            pass_not_in_wall = False; break
    diag['not_in_wall'] = pass_not_in_wall

    try: distinct_ok = (len({tuple(agent_pos), tuple(cand_keys[0]), tuple(cand_goals[0])}) == 3)
    except Exception: distinct_ok = False
    diag['distinct'] = distinct_ok

    agent_to_key_ok, reason7 = check_agent_to_key_reachable(grid, agent_pos)
    diag['agent_to_key'] = reason7

    full_ok, reason8 = check_full_solvable(grid, agent_pos)
    diag['full_solve'] = reason8

    single_wall_col_ok, _ = single_inner_wall_column_with_door(grid)
    diag['single_inner_wall'] = single_wall_col_ok

    orig_door_pos = original_special_positions.get("Door")
    cand_door_pos = tuple(cand_doors[0]) if cand_doors else None
    door_same = (cand_door_pos == tuple(orig_door_pos))
    diag['door_same'] = door_same

    weighted_sum = (
        weights['agent_unchanged'] * pass_agent_unchanged +
        weights['goal_unchanged'] * pass_goal_unchanged +
        weights['counts'] * counts_ok +
        weights['outer_walls'] * pass_outer_walls +
        weights['not_in_wall'] * pass_not_in_wall +
        weights['distinct'] * distinct_ok +
        weights['agent_to_key'] * agent_to_key_ok +
        weights['full_solve'] * full_ok +
        weights['single_inner_wall'] * single_wall_col_ok +
        weights['door_same'] * door_same
    )

    final_score = weighted_sum / total_weight if total_weight > 0 else 0
    final_score = max(0.0, min(1.0, final_score))

    diag['weighted_sum'] = weighted_sum
    diag['final_score'] = final_score

    return final_score, diag