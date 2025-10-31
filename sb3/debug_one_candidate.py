# debug_one_candidate.py
import json
from stable_baselines3 import PPO
from phase2_at_helpers import rollout_trajectory_improvement, action_change_score
from find_cs import get_true_optimal_path

MODEL = "ppo_doorkey_6x6_mini_2.zip"
POP_FILE = "populations/evaluated_sparsity/plausible_population_with_sparsity.json"
CRIT_FILE = "critical_states.json"
CRIT_IDX = 8

model = PPO.load(MODEL)
pop = json.load(open(POP_FILE))
crits = json.load(open(CRIT_FILE))
crit = crits[CRIT_IDX]
s_star = crit["s_star"]

# pick the i-th candidate (tweak this index, e.g., i = 1)
i = 1 # Use a modified environment, not the original (i=0)
cand = pop[i]
print("--- Testing Candidate:", i, "---")
print("Meta:", cand.get("meta",{}) )
print("Plausibility:", cand.get("plausibility_score"))
print("Sparsity:", cand.get("sparsity_score"))
print("-" * 20)

# A diagnostics
print("Running Action-Change (A) diagnostics...")
a = action_change_score(cand, s_star, model, n_samples=50)
print("A result:", a)
print("-" * 20)

# T diagnostics
print("Running Trajectory-Improvement (T) diagnostics...")
t = rollout_trajectory_improvement(cand, s_star, model, None, max_steps=400)
print("T result:", t)
print("-" * 20)

# Manually verify the true optimal path length
orig_len, _ = get_true_optimal_path(cand["grid"], tuple(s_star["pos"]))
print("Verification: get_true_optimal_path length =", orig_len)