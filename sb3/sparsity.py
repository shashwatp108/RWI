# sparsity.py
"""
Sparsity scoring for Phase-2 GA fitness (S-term).
Provides:
 - sparsity_score(candidate, original_grid, config) -> (score, diag)

Candidate format: { "grid": [[...]], "agent_pos": (x,y), "meta": {...} }
Grid cells are strings like "Wall","empty","Key","Door","Goal".

Config (optional) keys:
 - alpha, beta, gamma: weights for NH, SEP, EC (default 0.6,0.3,0.1)
 - w_spec: multiplier for special edits penalty (default 1.5)
"""

from typing import List, Tuple, Dict
from collections import deque
import math

# Default tile sets (override if needed)
WALLS = {"Wall"}
DOORS = {"Door"}
KEYS = {"Key"}
GOALS = {"Goal"}
SPECIAL_TILES = KEYS | DOORS | GOALS  # special tiles considered for SEP

def hamming_distance(grid1: List[List[str]], grid2: List[List[str]]) -> int:
    H = len(grid1); W = len(grid1[0])
    cnt = 0
    for y in range(H):
        for x in range(W):
            if grid1[y][x] != grid2[y][x]:
                cnt += 1
    return cnt

def interior_cells_count(grid: List[List[str]]) -> int:
    H = len(grid); W = len(grid[0])
    return max(1, (W-2) * (H-2))

def find_tile_positions(grid: List[List[str]], tile_set: set) -> List[Tuple[int,int]]:
    pos = []
    H = len(grid); W = len(grid[0])
    for y in range(H):
        for x in range(W):
            if grid[y][x] in tile_set:
                pos.append((x,y))
    return pos

def changed_cells_mask(grid1: List[List[str]], grid2: List[List[str]]) -> List[List[int]]:
    H = len(grid1); W = len(grid1[0])
    mask = [[0]*W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            if grid1[y][x] != grid2[y][x]:
                mask[y][x] = 1
    return mask

def count_connected_components(mask: List[List[int]]) -> int:
    H = len(mask); W = len(mask[0])
    visited = [[False]*W for _ in range(H)]
    components = 0
    for y in range(H):
        for x in range(W):
            if mask[y][x] == 1 and not visited[y][x]:
                components += 1
                # BFS
                q = deque([(x,y)])
                visited[y][x] = True
                while q:
                    cx, cy = q.popleft()
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < W and 0 <= ny < H and not visited[ny][nx] and mask[ny][nx] == 1:
                            visited[ny][nx] = True
                            q.append((nx, ny))
    return components

def count_special_edits(original_grid: List[List[str]], candidate_grid: List[List[str]]) -> Tuple[int, dict]:
    """
    Count how many special tiles (Key/Door/Goal) have been moved or changed.
    Return (num_special_edits, details)
    """
    orig_positions = {}
    cand_positions = {}
    for tset, name in [(KEYS, "Key"), (DOORS, "Door"), (GOALS, "Goal")]:
        orig_list = find_tile_positions(original_grid, tset)
        cand_list = find_tile_positions(candidate_grid, tset)
        orig_positions[name] = orig_list
        cand_positions[name] = cand_list

    edits = 0
    details = {}
    for name in ("Key","Door","Goal"):
        olist = orig_positions.get(name, [])
        clist = cand_positions.get(name, [])
        o = tuple(olist[0]) if olist else None
        c = tuple(clist[0]) if clist else None
        details[name] = {"orig": o, "cand": c}
        if o != c:
            edits += 1
    return edits, details

def sparsity_score(candidate: Dict, original_grid: List[List[str]], config: Dict=None) -> Tuple[float, Dict]:
    """
    Compute composite sparsity score S in [0,1].
    Returns (score, diag)
    """
    cfg = {"alpha":0.6, "beta":0.3, "gamma":0.1, "w_spec":1.5}
    if config:
        cfg.update(config)
    alpha = cfg["alpha"]; beta = cfg["beta"]; gamma = cfg["gamma"]; w_spec = cfg["w_spec"]

    orig = original_grid
    cand = candidate["grid"]

    H = len(orig); W = len(orig[0])
    interior = interior_cells_count(orig)

    # 1) Normalized Hamming (NH)
    hdist = hamming_distance(orig, cand)
    # Usually outer walls are untouched: ignore border cells in hamming normalization
    # But Hamming counted all cells; better to count only interior differences:
    interior_hdist = 0
    for y in range(1,H-1):
        for x in range(1,W-1):
            if orig[y][x] != cand[y][x]:
                interior_hdist += 1
    NH = 1.0 - (interior_hdist / interior)
    NH = max(0.0, min(1.0, NH))

    # 2) Special Edit Penalty (SEP)
    num_special_edits, special_details = count_special_edits(orig, cand)
    num_special_total = 3  # Key, Door, Goal (we exclude agent here — agent_pos is metadata)
    # raw special penalty: fraction changed
    frac_special_changed = num_special_edits / num_special_total
    # SEP in [0,1]: high value when none changed
    SEP = 1.0 - min(1.0, w_spec * frac_special_changed)
    SEP = max(0.0, min(1.0, SEP))

    # 3) Edit Compactness (EC) using connected components
    mask = changed_cells_mask(orig, cand)
    # consider interior mask only (exclude border)
    interior_mask = [[mask[y][x] if 1 <= x < W-1 and 1 <= y < H-1 else 0 for x in range(W)] for y in range(H)]
    num_comps = count_connected_components(interior_mask)
    if interior_hdist == 0:
        EC = 1.0
    else:
        EC = 1.0 / float(num_comps)  # single component -> 1, two -> 0.5
        # Optionally scale: EC = 1.0 / (1 + log(1+num_comps)) for softer drop
        EC = max(0.0, min(1.0, EC))

    # Combine
    weighted = alpha * NH + beta * SEP + gamma * EC
    norm = alpha + beta + gamma
    score = weighted / norm
    score = max(0.0, min(1.0, score))

    diag = {
        "NH": NH,
        "interior_hamming": interior_hdist,
        "interior_cells": interior,
        "SEP": SEP,
        "num_special_edits": num_special_edits,
        "special_details": special_details,
        "EC": EC,
        "num_components": num_comps,
        "weights": {"alpha":alpha, "beta":beta, "gamma":gamma, "w_spec":w_spec},
        "final_sparsity_score": score
    }

    return score, diag