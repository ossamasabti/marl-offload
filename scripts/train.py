"""Train IQL, IPPO and MAPPO on the offloading game and compare with baselines.

Usage:
    python scripts/train.py --seeds 0 1 2 3 4 --out results
"""
import argparse
import json
import os
import sys
import multiprocessing as mp
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marl_offload.baselines import GreedyPolicy, all_baselines, evaluate, tune_greedy_toll
from marl_offload.env import EnvConfig, OffloadEnv
from marl_offload.iql import IQLAgent, IQLConfig
from marl_offload.ppo import PPOAgent, PPOConfig

T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}


def ci95(values):
    """Half-width of a Student-t 95 % confidence interval over seeds."""
    n = len(values)
    if n < 2:
        return 0.0
    return float(T95.get(n, 1.96) * np.std(values, ddof=1) / np.sqrt(n))


def run_one(algo, seed, args):
    """Train one (algorithm, seed) pair. Runs in its own process when --jobs > 1."""
    torch.set_num_threads(1)  # small networks: one thread per process is faster than sharing cores
    env, eval_env = OffloadEnv(EnvConfig()), OffloadEnv(EnvConfig())
    if algo == "IQL":
        agent = IQLAgent(env, IQLConfig(episodes=args["iql_episodes"], seed=seed))
        every = max(args["iql_episodes"] // 25, 1)
    else:
        agent = PPOAgent(env, PPOConfig(centralized_critic=(algo == "MAPPO"), updates=args["ppo_updates"], seed=seed))
        every = max(args["ppo_updates"] // 25, 1)
    t0 = time.time()
    tag = f"[{algo} s{seed}]"
    hist = agent.train(eval_fn=lambda ag: evaluate(eval_env, ag, n_episodes=args["eval_episodes"]),
                       eval_every=every, log=lambda msg: print(tag, msg, flush=True))
    print(f"{algo} seed {seed} trained in {time.time() - t0:.0f}s", flush=True)
    agent.save(os.path.join(args["out"], "models", f"{algo}_seed{seed}.pt"))
    return algo, seed, hist, evaluate(eval_env, agent, n_episodes=100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--ppo-updates", type=int, default=400)
    p.add_argument("--iql-episodes", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--algos", nargs="+", default=["IQL", "IPPO", "MAPPO"])
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1)), help="parallel training processes")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)

    eval_env = OffloadEnv(EnvConfig())
    final = {}
    for b in all_baselines():
        final[b.name] = evaluate(eval_env, b, n_episodes=100)
    toll, toll_scores = tune_greedy_toll(eval_env)
    final["greedy+toll"] = evaluate(eval_env, GreedyPolicy(toll), n_episodes=100)
    final["greedy+toll"]["toll"] = toll
    final["greedy+toll"]["toll_tuning"] = toll_scores
    for name, r in final.items():
        print(f"{name:12s} cost/task {r['cost_per_task']:.3f}")

    jobs = [(algo, seed) for algo in args.algos for seed in args.seeds]
    work = [(algo, seed, vars(args)) for algo, seed in jobs]
    if args.jobs > 1:
        with mp.get_context("spawn").Pool(args.jobs) as pool:
            done = pool.starmap(run_one, work)
    else:
        done = [run_one(*w) for w in work]
    results = {(a, s): (h, f) for a, s, h, f in done}

    curves = {}
    for algo in args.algos:
        curves[algo] = [results[(algo, s)][0] for s in args.seeds]
        finals = [results[(algo, s)][1] for s in args.seeds]
        final[algo] = {k: sum(f[k] for f in finals) / len(finals) for k in finals[0]}
        per_seed = [f["cost_per_task"] for f in finals]
        final[algo]["cost_per_task_per_seed"] = per_seed
        final[algo]["cost_per_task_ci95"] = ci95(per_seed)
        final[algo]["seeds"] = args.seeds

    with open(os.path.join(args.out, "curves.json"), "w") as fh:
        json.dump(curves, fh, default=float)
    with open(os.path.join(args.out, "final.json"), "w") as fh:
        json.dump(final, fh, indent=2, default=float)
    for k, v in final.items():
        ci = v.get("cost_per_task_ci95")
        print(f"{k:12s} {v['cost_per_task']:.3f}" + (f" ± {ci:.3f}" if ci is not None else ""))


if __name__ == "__main__":
    main()
