"""Cross-check every number of README.md against results/*.json and the configuration.

1. The generated Results block must be identical to what scripts/readme_results.py builds
   from the current JSON files (so no number in it is stale).
2. The main results table is parsed and compared with final.json by an independent code path.
3. Every other number in the README must be a known configuration value.

Exit code 0 when everything matches. Usage: python scripts/check_readme.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # to import the sibling generator script
import readme_results as gen  # noqa: E402

from marl_offload.env import EnvConfig  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def main():
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    errors = []

    # 1. generated block in sync with the JSON
    a, b = readme.index(gen.START) + len(gen.START), readme.index(gen.END)
    block = readme[a:b].strip()
    if block != gen.build().strip():
        errors.append("Results block differs from what readme_results.py builds: run scripts/readme_results.py")

    # 2. main table, parsed independently and compared with final.json
    F = json.load(open(os.path.join(ROOT, "results", "final.json")))
    checked = 0
    for line in block.splitlines():
        m = re.match(r"\| (all-edge|all-local|all-cloud|random|greedy\+toll|greedy|IQL|IPPO|MAPPO)\b[^|]* \| ([\d.]+) \| ([^|]*)\| ([\d.]+) \| ([\d.]+) \| ([\d.]+) / ([\d.]+) / ([\d.]+) \| ([\d.]+) \|", line)
        if not m:
            continue
        name = m.group(1)
        r = F[name]
        pairs = [(float(m.group(2)), r["cost_per_task"], 3), (float(m.group(4)), r["latency_per_task"], 3),
                 (float(m.group(5)), r["energy_per_task"], 3), (float(m.group(6)), r["share_local"], 2),
                 (float(m.group(7)), r["share_edge"], 2), (float(m.group(8)), r["share_cloud"], 2),
                 (float(m.group(9)), r["mean_edge_backlog"], 2)]
        ci = re.search(r"([\d.]+)", m.group(3))
        if ci:
            pairs.append((float(ci.group(1)), r["cost_per_task_ci95"], 3))
        for shown, true, d in pairs:
            if abs(shown - round(true, d)) > 10 ** -d / 2 + 1e-12:
                errors.append(f"{name}: README shows {shown}, JSON gives {true:.{d + 2}f}")
            checked += 1
    if checked < 9 * 7:
        errors.append(f"only {checked} table cells parsed; the table format changed")

    # 3. numbers outside the generated block must be configuration values
    cfg = EnvConfig()
    allowed = {cfg.n_agents, cfg.edge_speed, cfg.signal_delay, min(cfg.local_speeds), max(cfg.local_speeds),
               round(cfg.n_agents * cfg.arrival_prob * sum(cfg.task_size) / 2, 1),  # 4.8 units of work per step
               0.5,    # leader congestion weight (scripts/stackelberg.py)
               16,     # prices in the sweep
               5,      # training seeds (scripts/train.py default)
               3.10,   # minimum Python version
               0,      # NashConv of an equilibrium (definition)
               1, 2}   # numbered list items
    outside = readme[:a] + readme[b:]
    for x in NUM.findall(outside):
        if float(x) not in {float(v) for v in allowed}:
            errors.append(f"number {x} outside the generated block is not a known configuration value")

    if "—" in readme:
        errors.append("README contains an em dash")

    if errors:
        print("README check FAILED")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print(f"README check passed: block in sync with the JSON, {checked} table cells verified, "
          f"{len(NUM.findall(outside))} other numbers are configuration values.")


if __name__ == "__main__":
    main()
