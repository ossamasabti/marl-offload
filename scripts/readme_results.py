"""Write the Results and Limitations sections of README.md from results/*.json.

The sections live between the markers <!-- results:start --> and <!-- results:end -->,
so every number in them comes from the JSON files. scripts/check_readme.py verifies
that the README is in sync.

Usage:
    python scripts/readme_results.py            # rewrite README.md
    python scripts/readme_results.py --print    # print the generated block only
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
START, END = "<!-- results:start -->", "<!-- results:end -->"
LEARNED = ("IQL", "IPPO", "MAPPO")
RULES = ("greedy", "greedy+toll")


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, d=3):
    return f"{v:.{d}f}"


def pct(v, d=0):
    return f"{100 * v:.{d}f}%"


def build():
    F, S, A, E = load("final.json"), load("stackelberg.json"), load("per_agent.json"), load("exploitability.json")
    n_seeds = len(F["IQL"]["seeds"])
    out = []
    w = out.append

    # ------------------------------------------------------------------ main table
    w("## Results\n")
    w(f"All numbers come from `results/*.json`. Costs are averages over 100 evaluation episodes. "
      f"Learned methods are averaged over {n_seeds} training seeds, with a Student-t 95% confidence interval over seeds. "
      f"Training, tuning and evaluation use disjoint seed ranges.\n")
    w("| policy | cost per task | 95% CI | latency | energy | local / edge / cloud | mean edge backlog |")
    w("|---|---|---|---|---|---|---|")
    order = ["all-edge", "all-local", "all-cloud", "random", "greedy", "greedy+toll", *LEARNED]
    for n in order:
        r = F[n]
        ci = f"± {f(r['cost_per_task_ci95'])}" if r.get("cost_per_task_ci95") is not None else ""
        name = f"greedy+toll (toll {f(r['toll'], 1)})" if n == "greedy+toll" else n
        w(f"| {name} | {f(r['cost_per_task'])} | {ci} | {f(r['latency_per_task'])} | {f(r['energy_per_task'])} | "
          f"{f(r['share_local'], 2)} / {f(r['share_edge'], 2)} / {f(r['share_cloud'], 2)} | {f(r['mean_edge_backlog'], 2)} |")
    w("")
    w("![learning curves](results/figures/learning_curves.png)\n")

    # ------------------------------------------------------------------ findings on cost
    g = F["greedy"]["cost_per_task"]
    costs = {a: F[a]["cost_per_task"] for a in LEARNED}
    lo, hi = 1 - max(costs.values()) / g, 1 - min(costs.values()) / g
    best = min(costs, key=costs.get)
    toll = F["greedy+toll"]["cost_per_task"]
    w(f"**Learners against greedy.** The three learners beat the myopic greedy rule ({f(g)}) by "
      f"{pct(lo)} to {pct(hi)}. Greedy devices react to a stale load signal and herd onto the server. "
      f"The learned policies send fewer tasks to the edge and keep a shorter queue "
      f"(mean backlog {f(min(F[a]['mean_edge_backlog'] for a in LEARNED), 2)} to "
      f"{f(max(F[a]['mean_edge_backlog'] for a in LEARNED), 2)}, against {f(F['greedy']['mean_edge_backlog'], 2)} for greedy).\n")
    ci_best = F[best]["cost_per_task_ci95"]
    if toll <= min(costs.values()) and toll >= costs[best] - ci_best:
        rel = (f"slightly lower than every learner, but inside the 95% confidence interval of the best learner "
               f"({best}, {f(costs[best])} ± {f(ci_best)}), so the toll rule and the best learner are on par")
    elif toll <= min(costs.values()):
        rel = f"lower than every learner (best learner: {best}, {f(costs[best])})"
    else:
        rel = f"within {f(toll - min(costs.values()))} of the best learner ({best}, {f(costs[best])})"
    w(f"**Learners against a tuned toll.** A greedy rule with a virtual congestion toll of {f(F['greedy+toll']['toll'], 1)}, "
      f"tuned on separate seeds, reaches {f(toll)}, {rel}. On average cost, most of the gain over greedy comes from "
      f"internalising congestion, which a one-parameter rule already captures.\n")
    ivs = {a: (costs[a] - F[a]["cost_per_task_ci95"], costs[a] + F[a]["cost_per_task_ci95"]) for a in LEARNED}
    overlap = max(v[0] for v in ivs.values()) <= min(v[1] for v in ivs.values())
    w("**Ranking the learners.** " + (
        "The 95% confidence intervals of IQL, IPPO and MAPPO overlap, so the three learners cannot be ranked. "
        "The centralised critic of MAPPO brings no measurable gain, even though it sees the true current load.\n"
        if overlap else
        f"The 95% confidence intervals do not all overlap; {best} has the lowest cost.\n"))
    m, gr = A["MAPPO"], A["greedy"]
    w(f"**Asymmetry.** All policies give the shared server mostly to the slow devices. With MAPPO (mean over seeds), "
      f"the slowest device (CPU 0.6) sends {pct(m[0])} of its tasks to the edge and the fastest (CPU 1.2) {pct(m[-1])}. "
      f"With greedy the shares are {pct(gr[0])} and {pct(gr[-1])}.\n")
    w("![per-agent edge share](results/figures/per_agent_edge_share.png)\n")

    # ------------------------------------------------------------------ leader-follower
    su, sw = S["summary"], S["sweep"]
    i0, isoc, iop = 0, sw["price"].index(su["social_optimal_price"]), sw["price"].index(su["sweep_optimal_price"])
    soc = lambda i: sw["learned"][i]["social_cost_per_task"]
    w("### Leader-follower pricing\n")
    w("The edge price is a transfer from the devices to the operator, so the social cost counts latency, energy and "
      "cloud fees, but not the edge price. With MAPPO followers:\n")
    w(f"- The operator's payoff is highest at a price of {f(su['sweep_optimal_price'], 1)} in the exhaustive sweep. "
      f"The UCB leader picks the same price ({f(su['bandit_price'], 1)}) and plays it in {pct(su['bandit_share_of_plays'])} "
      f"of {len(S['bandit_choices'])} episodes.")
    w(f"- The social cost is lowest at a price of {f(su['social_optimal_price'], 1)}: {f(soc(isoc))} per task, against "
      f"{f(soc(i0))} at price 0 ({pct(1 - soc(isoc) / soc(i0), 1)} lower). A moderate price acts as a congestion toll.")
    w(f"- At the operator's price, the social cost is {f(soc(iop))}. The operator's price is "
      f"{f(su['sweep_optimal_price'] / su['social_optimal_price'], 1)} times the socially optimal one, so the two goals are not aligned.")
    w(f"- With greedy followers, the socially optimal price is {f(su['social_optimal_price_greedy'], 1)}.\n")
    w("![stackelberg](results/figures/stackelberg.png)\n")

    # ------------------------------------------------------------------ equilibrium
    w("### Distance to equilibrium\n")
    if not E:
        w("The exploitability analysis has not been run yet (`scripts/exploitability.py`).\n")
    else:
        names = [n for n in (*RULES, *LEARNED) if n in E]
        any_e = E[names[0]]
        lseeds = sorted({len(E[n]["seeds"]) for n in LEARNED if n in E})
        one_seed = lseeds == [1]
        if one_seed:
            seed_sentence = (f"Learned policies are analysed on one training seed only (seed "
                             f"{E[next(n for n in LEARNED if n in E)]['seeds'][0]}), for lack of compute time")
        else:
            seed_sentence = "Learned policies are analysed on " + ", ".join(
                n + " seeds " + " and ".join(str(x) for x in E[n]["seeds"]) for n in LEARNED if n in E)
        w(f"For each device, the other five are frozen and the device trains a PPO best response. For learned policies the "
          f"best response starts from the device's current policy; for the rules it starts from scratch. The best checkpoint "
          f"is chosen on validation episodes, and the starting policy is one of the candidates. Before and after are evaluated "
          f"on the same {any_e['eval_episodes']} evaluation seeds. NashConv is the sum over devices of max(0, gain); the raw sum "
          f"is also reported. {seed_sentence}; the rules do not depend on a seed. A gain of +0.000 ± 0.000 means that the "
          f"starting policy itself was selected.\n")
        w("| policy | NashConv | raw sum of gains | mean gain per device | devices where the last iterate did worse |")
        w("|---|---|---|---|---|")
        for n in names:
            e = E[n]
            nc = sorted({a for l in e["not_converged"].values() for a in l})
            w(f"| {n} | {f(e['nash_conv'])} | {f(e['nash_conv_raw'])} | {pct(e['mean_relative_gain'], 1)} | "
              f"{', '.join(str(a) for a in nc) if nc else 'none'} |")
        w("")
        w("Gain per device (cost before minus cost after, with 95% CI over paired episodes):\n")
        w("| policy | " + " | ".join(f"dev {r['agent']} (CPU {r['cpu']})" for r in any_e["per_agent"]) + " |")
        w("|---|" + "---|" * len(any_e["per_agent"]))
        for n in names:
            w(f"| {n} | " + " | ".join(f"{r['gain']:+.3f} ± {f(r['gain_ci95'])}" for r in E[n]["per_agent"]) + " |")
        w("")
        lrn = [E[n]["nash_conv"] for n in LEARNED if n in E]
        rul = {n: E[n]["nash_conv"] for n in RULES if n in E}
        sig = {n: [r["agent"] for r in E[n]["per_agent"] if r["gain"] - r["gain_ci95"] > 0] for n in names}
        sig_items = [f"{n} device {a}" for n in LEARNED if n in E for a in sig[n]]
        sig_l = " and ".join(sig_items) if len(sig_items) <= 2 else ", ".join(sig_items[:-1]) + " and " + sig_items[-1]
        fell_back = [n for n in LEARNED if n in E and any(E[n]["not_converged"].values())]
        max_l = max((r["gain"] / r["cost_before"] for n in LEARNED if n in E for r in E[n]["per_agent"]), default=0.0)
        if lrn and rul and max(lrn) < min(rul.values()):
            w(f"The learned policies are much closer to a Nash equilibrium (NashConv {f(min(lrn))} to {f(max(lrn))}) than "
              f"greedy ({f(rul.get('greedy', float('nan')))}) and greedy+toll ({f(rul.get('greedy+toll', float('nan')))}). "
              f"With a learned policy, a deviating device gains at most {pct(max_l, 1)} of its cost, and the gain is "
              f"significant (95% CI above zero) only for {sig_l if sig_l else 'no device'}. The gain is significant for "
              f"{len(sig.get('greedy+toll', []))} of 6 devices with greedy+toll and {len(sig.get('greedy', []))} of 6 with greedy. "
              f"The toll rule matches the learners on average cost, but it is not stable: single devices gain by ignoring it.\n")
            w("Two caveats. First, for the learned policies the search is local, because the best response starts at the "
              "current policy. "
              + (f"On several {' and '.join(fell_back)} devices, fine-tuning made the last iterate worse and the selection "
                 "fell back to the starting policy or a checkpoint close to it. " if fell_back else "")
              + "A local search that finds no profitable deviation is consistent with a "
              "near-equilibrium, but it does not prove one. Second, all best responses are learned, so every value in this "
              "section is a lower bound.\n")
        else:
            w("The learned policies are not clearly closer to an equilibrium than the rules in this analysis. "
              "Because the best responses are learned, all these values are lower bounds.\n")
        w("![exploitability](results/figures/exploitability.png)\n")

    # ------------------------------------------------------------------ limitations
    w("## Limitations\n")
    w(f"- The cloud option is almost never used at a zero price (cloud share {f(F['MAPPO']['share_cloud'], 3)} for MAPPO). "
      "With these parameters it is dominated by the edge and local options, so in practice the devices choose between two actions.")
    w("- The leader is static (one price per episode), and the follower training does not adapt to the leader's learning.")
    w(f"- The UCB bandit chooses among the same {len(sw['price'])} prices as the exhaustive sweep, so it only confirms the sweep.")
    if E:
        w("- Exploitability uses a learned best response, so it gives lower bounds. "
          + ("Learned policies are analysed on one training seed only. " if one_seed else "")
          + "The evaluation is paired by seed, but the environment draws all its randomness from one stream, so the two "
          "runs of a pair stop sharing random draws once the deviation changes how many devices use the edge.")
        rule_upd = sorted({E[n]["updates"] for n in RULES if n in E})
        if rule_upd:
            line = f"- For the rules, the best response is trained from scratch for {' or '.join(map(str, rule_upd))} PPO updates."
            late = []
            for n in RULES:
                run = load(os.path.join("exploitability", f"{n}_seed0.json"))
                if run:
                    k = sum(r["selected_update"] == run["config"]["updates"] for r in run["per_agent"])
                    if k:
                        late.append(f"{n} ({k} of {len(run['per_agent'])} devices)")
            if late:
                line += (" The best checkpoint was the last one for " + " and ".join(late) + ", so these best responses "
                         "may not have converged, and the NashConv of the rules is a loose lower bound.")
            w(line)
    w("- The model is stylised: one edge server, an abstract unit of work, and simple channel and energy models.\n")
    return "\n".join(out)


def main():
    block = build()
    if "--print" in sys.argv:
        print(block)
        return
    path = os.path.join(ROOT, "README.md")
    s = open(path, encoding="utf-8").read()
    a, b = s.index(START) + len(START), s.index(END)
    s = s[:a] + "\n" + block + "\n" + s[b:]
    open(path, "w", encoding="utf-8").write(s)
    print("README.md results section updated")


if __name__ == "__main__":
    main()
