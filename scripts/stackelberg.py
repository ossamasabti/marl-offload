"""Hierarchical (leader-follower) variant.

The edge operator is a leader who posts a price for edge access. Devices are
followers who see the price and react. We proceed in two levels:

1. Followers: one MAPPO policy is trained with a random price drawn at each
   episode, so it learns a price-conditioned response pi(a | o, price).
2. Leader: the operator then learns which price to post, with a UCB bandit
   that only observes its own payoff after each episode (it does not know the
   followers' policy). Its payoff is revenue minus a congestion cost on the
   edge backlog. We compare the price it settles on with an exhaustive sweep,
   and with the price that minimises the social cost. The price itself is a
   transfer from devices to the operator, so the social cost counts latency,
   energy and cloud fees, not the edge price. A positive price can lower it:
   it acts as a congestion toll.

Usage:
    python scripts/stackelberg.py --out results
"""
import argparse
import json
import os

import numpy as np


from marl_offload.baselines import GreedyPolicy, evaluate
from marl_offload.env import EnvConfig, OffloadEnv
from marl_offload.ppo import PPOAgent, PPOConfig

CONGESTION_WEIGHT = 0.5


def leader_payoff(r):
    return r["revenue_per_step"] - CONGESTION_WEIGHT * r["mean_edge_backlog"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--bandit-episodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)

    # 1. price-conditioned followers
    env = OffloadEnv(EnvConfig(random_price=True))
    eval_env = OffloadEnv(EnvConfig())
    followers = PPOAgent(env, PPOConfig(centralized_critic=True, updates=args.updates, seed=args.seed))
    followers.train(eval_fn=lambda ag: evaluate(eval_env, ag, n_episodes=10, price=0.5), eval_every=25)
    followers.save(os.path.join(args.out, "models", "stackelberg_followers.pt"))

    # 2a. exhaustive sweep (reference)
    prices = np.round(np.linspace(0.0, 3.0, 16), 2)
    sweep = {"price": prices.tolist(), "learned": [], "greedy": []}
    for pr in prices:
        for key, pol in (("learned", followers), ("greedy", GreedyPolicy())):
            r = evaluate(eval_env, pol, n_episodes=40, price=float(pr))
            r["leader_payoff"] = leader_payoff(r)
            sweep[key].append(r)
        print(f"price {pr:.2f}  learned: payoff {sweep['learned'][-1]['leader_payoff']:.3f} "
              f"edge share {sweep['learned'][-1]['share_edge']:.2f} device cost {sweep['learned'][-1]['cost_per_task']:.3f} "
              f"social cost {sweep['learned'][-1]['social_cost_per_task']:.3f}")

    # 2b. leader learns the price online with UCB1 (bandit feedback only)
    rng = np.random.default_rng(args.seed)
    K = len(prices)
    counts, means = np.zeros(K), np.zeros(K)
    chosen = []
    for t in range(args.bandit_episodes):
        if t < K:
            k = t
        else:
            k = int(np.argmax(means + 1.0 * np.sqrt(2 * np.log(t + 1) / counts)))
        r = evaluate(eval_env, followers, n_episodes=1, seed=int(rng.integers(1e9)), price=float(prices[k]))
        pay = leader_payoff(r)
        counts[k] += 1
        means[k] += (pay - means[k]) / counts[k]
        chosen.append(float(prices[k]))
    best_bandit = float(prices[int(np.argmax(counts))])
    best_sweep = float(prices[int(np.argmax([r["leader_payoff"] for r in sweep["learned"]]))])
    best_devices = float(prices[int(np.argmin([r["cost_per_task"] for r in sweep["learned"]]))])
    best_social = float(prices[int(np.argmin([r["social_cost_per_task"] for r in sweep["learned"]]))])
    best_social_greedy = float(prices[int(np.argmin([r["social_cost_per_task"] for r in sweep["greedy"]]))])
    summary = {"bandit_price": best_bandit, "sweep_optimal_price": best_sweep,
               "bandit_share_of_plays": chosen.count(best_bandit) / len(chosen),
               "device_optimal_price": best_devices, "social_optimal_price": best_social,
               "social_optimal_price_greedy": best_social_greedy, "congestion_weight": CONGESTION_WEIGHT}
    print(summary)
    with open(os.path.join(args.out, "stackelberg.json"), "w") as fh:
        json.dump({"sweep": sweep, "bandit_choices": chosen, "summary": summary}, fh, default=float)


if __name__ == "__main__":
    main()
