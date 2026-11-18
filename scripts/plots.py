"""Make the figures in results/figures from the JSON files written by
train.py and stackelberg.py."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marl_offload.baselines import GreedyPolicy
from marl_offload.env import EnvConfig, OffloadEnv
from marl_offload.iql import IQLAgent, IQLConfig
from marl_offload.ppo import PPOAgent, PPOConfig

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
COL = {"IQL": "#2a9d8f", "IPPO": "#e76f51", "MAPPO": "#264653", "greedy": "#8d99ae", "random": "#c9ada7",
       "greedy+toll": "#e9c46a"}
N_AGENTS = EnvConfig().n_agents
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def learning_curves():
    curves = json.load(open(os.path.join(RES, "curves.json")))
    final = json.load(open(os.path.join(RES, "final.json")))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for algo, runs in curves.items():
        x = np.array([h["env_steps"] for h in runs[0]]) * N_AGENTS  # agent-steps
        y = np.array([[h["cost_per_task"] for h in r] for r in runs])
        m, s = y.mean(0), y.std(0)
        ax.plot(x, m, label=algo, color=COL[algo])
        ax.fill_between(x, m - s, m + s, color=COL[algo], alpha=0.2)
    for b in ("greedy", "greedy+toll", "random"):
        ax.axhline(final[b]["cost_per_task"], ls="--", color=COL[b], label=f"{b} (baseline)")
    ax.set_xlabel("agent-steps of experience")
    ax.set_ylabel("average cost per task (lower is better)")
    ax.set_ylim(1.3, 2.2)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Learning curves (mean and std over seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "learning_curves.png"), dpi=150)


def final_bars():
    final = json.load(open(os.path.join(RES, "final.json")))
    names = ["all-local", "all-cloud", "random", "greedy", "greedy+toll", "IQL", "IPPO", "MAPPO"]
    lat = [final[n]["latency_per_task"] for n in names]
    en = [final[n]["energy_per_task"] for n in names]
    pay = [final[n]["cost_per_task"] - final[n]["latency_per_task"] - final[n]["energy_per_task"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(names, lat, label="latency", color="#264653")
    ax.bar(names, en, bottom=lat, label="energy", color="#2a9d8f")
    ax.bar(names, pay, bottom=np.array(lat) + np.array(en), label="fees", color="#e9c46a")
    ax.set_ylabel("cost per task")
    ax.set_title("Final cost per task (all-edge: %.1f, off the chart)" % final["all-edge"]["cost_per_task"])
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "final_costs.png"), dpi=150)


def per_agent():
    """Edge share per device, averaged over every trained seed (not just one)."""
    final = json.load(open(os.path.join(RES, "final.json")))
    env = OffloadEnv(EnvConfig())
    policies = {"greedy": [GreedyPolicy()], "greedy+toll": [GreedyPolicy(final["greedy+toll"]["toll"])]}
    for algo in ("IQL", "MAPPO"):
        policies[algo] = []
        for seed in final[algo]["seeds"]:
            pol = IQLAgent(env, IQLConfig()) if algo == "IQL" else PPOAgent(env, PPOConfig(centralized_critic=True))
            pol.load(os.path.join(RES, "models", f"{algo}_seed{seed}.pt"))
            policies[algo].append(pol)
    out = {}
    for name, pols in policies.items():
        edge = np.zeros(env.n); tasks = np.zeros(env.n)
        for pol in pols:
            for ep in range(100):
                obs, _ = env.reset(seed=20_000 + ep)
                done = False
                while not done:
                    obs, _, _, trunc, infos = env.step(pol.act(env, obs))
                    info = next(iter(infos.values()))
                    tasks += info["had_task"]
                    edge += (info["actions"] == 1) * info["had_task"]
                    done = all(trunc.values())
        out[name] = edge / tasks
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.8 / len(out)
    xs = np.arange(env.n)
    for k, (name, v) in enumerate(out.items()):
        ax.bar(xs + (k - (len(out) - 1) / 2) * w, v, w, label=name, color=COL[name])
    ax.set_xticks(xs, [f"dev {i}\nCPU {f}" for i, f in enumerate(env.f_local)])
    ax.set_ylabel("share of tasks sent to the edge")
    ax.set_title("Asymmetric agents: who uses the shared edge server?")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "per_agent_edge_share.png"), dpi=150)
    json.dump({k: v.tolist() for k, v in out.items()}, open(os.path.join(RES, "per_agent.json"), "w"), indent=2)


def stackelberg():
    d = json.load(open(os.path.join(RES, "stackelberg.json")))
    sw, s = d["sweep"], d["summary"]
    p = sw["price"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for key in ("learned", "greedy"):
        lab = "MAPPO followers" if key == "learned" else "greedy followers"
        c = COL["MAPPO"] if key == "learned" else COL["greedy"]
        axes[0].plot(p, [r["leader_payoff"] for r in sw[key]], "o-", ms=3, color=c, label=lab)
        axes[1].plot(p, [r["share_edge"] for r in sw[key]], "o-", ms=3, color=c, label=lab)
        axes[2].plot(p, [r["social_cost_per_task"] for r in sw[key]], "o-", ms=3, color=c, label=lab)
    axes[0].axvline(s["bandit_price"], color="#e76f51", ls="--", label="price learned by UCB leader")
    axes[0].set_title("Leader payoff\n(revenue - congestion cost)")
    axes[1].set_title("Followers' edge usage")
    axes[2].axvline(s["social_optimal_price"], color="#2a9d8f", ls="--", label="social optimum")
    axes[2].set_title("Social cost per task\n(latency + energy + cloud fee)")
    axes[2].legend(frameon=False, fontsize=8)
    for a in axes:
        a.set_xlabel("edge price posted by the leader")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "stackelberg.png"), dpi=150)


def exploitability():
    d = json.load(open(os.path.join(RES, "exploitability.json")))
    names = [n for n in ("greedy", "greedy+toll", "IQL", "IPPO", "MAPPO") if n in d]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.8 / len(names)
    for k, n in enumerate(names):
        rows = d[n]["per_agent"]
        xs = np.arange(len(rows))
        ax.bar(xs + (k - (len(names) - 1) / 2) * w, [r["gain"] / r["cost_before"] for r in rows], w,
               label=f"{n} (NashConv {d[n]['nash_conv']:.2f})", color=COL[n])
    ax.set_xticks(xs, [f"dev {r['agent']}\nCPU {r['cpu']}" for r in rows])
    ax.set_ylabel("cost reduction from deviating")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Exploitability: what a device gains by best-responding to the others")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exploitability.png"), dpi=150)


if __name__ == "__main__":
    for f in (learning_curves, final_bars, per_agent, stackelberg, exploitability):
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
