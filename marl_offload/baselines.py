"""Hand-written policies used as reference points."""
import numpy as np

from .env import CLOUD, EDGE, LOCAL, OffloadEnv


class FixedPolicy:
    def __init__(self, action):
        self.action = action
        self.name = {LOCAL: "all-local", EDGE: "all-edge", CLOUD: "all-cloud"}[action]

    def act(self, env, obs):
        return {a: self.action for a in env.agents}


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        return {a: int(self.rng.integers(3)) for a in env.agents}


class GreedyPolicy:
    """Myopic best response: each device estimates the cost of the three options
    from its own observation, taking the delayed edge-load signal as the current
    backlog and assuming nobody else offloads in the same step. This is what an
    uncoordinated, non-learning device would do, and it tends to herd onto the
    edge server."""

    def __init__(self, toll=0.0):
        # toll > 0: a virtual surcharge added to the edge estimate. It is not paid to
        # anyone; it only makes devices internalise part of the congestion they cause.
        self.toll = toll
        self.name = "greedy" if toll == 0 else "greedy+toll"

    def act(self, env, obs):
        c = env.cfg
        actions = {}
        lo, hi = c.price_range
        for i, a in enumerate(env.agents):
            o = obs[a]
            size = o[1] * c.task_size[1]
            q = o[2] * 5.0
            f = o[4]
            backlog = o[5] * 10.0
            price = lo + o[6] * (hi - lo)
            rate = c.rate_good if o[3] > 0 else c.rate_bad
            tx = c.data_ratio * size / rate
            cost_local = (q + size) / f + c.energy_weight * c.energy_coef * f ** 2 * size
            cost_edge = tx + (backlog + size) / c.edge_speed + c.energy_weight * c.tx_power * tx + price + self.toll
            cost_cloud = tx + c.cloud_delay + size / c.cloud_speed + c.energy_weight * c.tx_power * tx + c.cloud_fee
            actions[a] = int(np.argmin([cost_local, cost_edge, cost_cloud]))
        return actions


def all_baselines():
    return [FixedPolicy(LOCAL), FixedPolicy(EDGE), FixedPolicy(CLOUD), RandomPolicy(), GreedyPolicy()]


TUNE_SEED = 50_000  # seeds used to tune baseline parameters, distinct from EVAL_SEED


def tune_greedy_toll(env, tolls=None, n_episodes=50):
    """Pick the virtual toll that minimises cost per task on tuning seeds (not the eval seeds)."""
    tolls = np.round(np.arange(0.0, 1.21, 0.1), 2) if tolls is None else tolls
    scores = [evaluate(env, GreedyPolicy(t), n_episodes=n_episodes, seed=TUNE_SEED)["cost_per_task"] for t in tolls]
    return float(tolls[int(np.argmin(scores))]), dict(zip(map(float, tolls), scores))


EVAL_SEED = 10_000  # evaluation episodes use seeds EVAL_SEED, EVAL_SEED + 1, ...


def evaluate(env: OffloadEnv, policy, n_episodes=50, seed=EVAL_SEED, price=None):
    """Average per-task cost, latency, energy, action shares and operator revenue."""
    tot = {"cost": 0.0, "latency": 0.0, "energy": 0.0, "tasks": 0.0, "revenue": 0.0,
           "edge_backlog": 0.0, "steps": 0}
    counts = np.zeros(3)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep, price=price)
        done = False
        while not done:
            actions = policy.act(env, obs)
            obs, rew, _, trunc, infos = env.step(actions)
            info = next(iter(infos.values()))
            had = info["had_task"] > 0
            tot["cost"] += -sum(rew.values())
            tot["latency"] += info["latency"][had].sum()
            tot["energy"] += info["energy"][had].sum()
            tot["tasks"] += had.sum()
            tot["revenue"] += info["revenue"]
            tot["edge_backlog"] += info["edge_backlog"]
            tot["steps"] += 1
            counts += np.bincount(info["actions"][had], minlength=3)
            done = all(trunc.values())
    n = max(tot["tasks"], 1)
    return {
        "cost_per_task": tot["cost"] / n,
        "latency_per_task": tot["latency"] / n,
        "energy_per_task": tot["energy"] / n,
        # The edge price is a transfer from devices to the operator, so it is not a social
        # cost; the cloud fee is paid outside the system and stays in.
        "social_cost_per_task": (tot["cost"] - tot["revenue"]) / n,
        "revenue_per_step": tot["revenue"] / tot["steps"],
        "mean_edge_backlog": tot["edge_backlog"] / tot["steps"],
        "share_local": counts[0] / n, "share_edge": counts[1] / n, "share_cloud": counts[2] / n,
    }
