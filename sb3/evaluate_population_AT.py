# evaluate_population_AT.py
"""
Driver: compute A (action-change sampling) for all candidates, then compute T (rollouts) for top-K.

Usage:
python evaluate_population_AT.py \
  --population populations/evaluated_sparsity/plausible_population_with_sparsity.json \
  --episodes-file episode_logs/episodes_full.json \
  --critical-states critical_states.json \
  --critical-index 0 \
  --model-file ppo_doorkey_6x6_mini_2.zip \
  --out-dir populations/evaluated_AT \
  --n_samples 20 \
  --top_k 30 \
  --max_steps 200
"""
import os, json, argparse, math
from phase2_at_helpers import action_change_score, rollout_trajectory_improvement
from phase2_at_helpers import build_env_from_grid_codes
from sparsity import sparsity_score
from plausibility import plausibility_score, find_tile_positions
from find_cs import astar_min_steps  # your existing A* helper
from stable_baselines3 import PPO

def load_json(path):
    with open(path,"r") as f: return json.load(f)
def save_json(obj,path):
    with open(path,"w") as f: json.dump(obj,f,indent=2)

def fetch_original_info(episode_record):
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
    crits = load_json(args.critical_states)
    if args.critical_index >= len(crits): raise IndexError("critical_index out of range")
    crit = crits[args.critical_index]
    ep_map = {ep['episode_id']: ep for ep in episodes}
    ep = ep_map[crit["episode_id"]]
    orig_grid, orig_agent_pos, orig_special = fetch_original_info(ep)
    s_star = crit["s_star"]

    # load model
    model = PPO.load(args.model_file)

    # 1) compute A for all (cheap sampling)
    annotated = []
    print(f"Computing A (action-change sampling) on population (n={len(pop)}) ...")
    for i, cand in enumerate(pop):
        a_res = action_change_score(cand, s_star, model,
                                    env_name=args.env_name,
                                    n_samples=args.n_samples,
                                    threshold_binary=args.a_threshold)
        # also include P and S if available
        cand_copy = dict(cand)
        cand_copy['a_result'] = a_res
        # attach sparsity/plausibility if missing: set to None
        cand_copy.setdefault('plausibility_score', None)
        cand_copy.setdefault('sparsity_score', None)
        annotated.append(cand_copy)
        if (i+1) % 50 == 0: print(f"  processed {i+1}/{len(pop)}")

    # Save intermediate results
    os.makedirs(args.out_dir, exist_ok=True)
    annotated_path = os.path.join(args.out_dir, "population_with_A.json")
    save_json(annotated, annotated_path)
    print(f"Saved A-annotated population to {annotated_path}")

    # 2) select top-K candidates to compute T
    # Order by simple pre-score: combine plausibility, sparsity, a_score
    def pre_score(entry):
        p = entry.get('plausibility_score') or 0.0
        s = entry.get('sparsity_score') or 0.0
        a = entry['a_result'].get('a_score') if entry['a_result'].get('ok') else 0.0
        # linear scoring (you can tune weights)
        return 0.4*p + 0.4*s + 0.2*a

    annotated_sorted = sorted(annotated, key=lambda e: pre_score(e), reverse=True)
    topk = annotated_sorted[:args.top_k]
    print(f"Selected top {len(topk)} candidates for rollouts (T).")

    # 3) compute T for top-K
    final_list = []
    for i, cand in enumerate(topk):
        t_res = rollout_trajectory_improvement(cand, s_star, model, astar_min_steps,
                                               env_name=args.env_name, max_steps=args.max_steps)
        cand2 = dict(cand)
        cand2['t_result'] = t_res
        final_list.append(cand2)
        print(f"  Rollout {i+1}/{len(topk)}: ok={t_res.get('ok')} T={t_res.get('T')} elapsed={t_res.get('elapsed')}")
    final_path = os.path.join(args.out_dir, "topk_with_T.json")
    save_json(final_list, final_path)
    print(f"Saved top-K with T to {final_path}")

    # 4) Compute final fitness for top-K (example weights)
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
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--env-name", type=str, default="MiniGrid-DoorKey-6x6-v0")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--a_threshold", type=float, default=0.5)
    parser.add_argument("--wA", type=float, default=1.0)
    parser.add_argument("--wT", type=float, default=2.0)
    parser.add_argument("--wS", type=float, default=1.0)
    parser.add_argument("--wP", type=float, default=1.0)
    args = parser.parse_args()
    main(args)