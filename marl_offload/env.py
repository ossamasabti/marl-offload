"""Multi-agent edge-cloud computation offloading environment.

N heterogeneous devices (the agents) receive computing tasks. At every step,
each device with a task decides where to run it:

    0 = locally, on its own CPU
    1 = on a shared edge server (fast, but congested when many devices use it)
    2 = in the cloud (no congestion, but a long backhaul delay and a fee)

The edge server keeps a backlog across steps, so today's offloading decisions
change tomorrow's latency for everyone: the problem is a general-sum Markov
game. Devices only see local information plus a delayed, public signal of the
edge load (partial observability). Agents are asymmetric: CPU speeds and
channel qualities differ from one device to another.

An optional leader (the edge operator) sets the price of edge access. Followers
see the price in their observation, which gives the hierarchical (Stackelberg)
variant used in scripts/stackelberg.py.

The API follows the PettingZoo parallel style (dicts keyed by agent name), but
the environment has no dependency other than NumPy.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

LOCAL, EDGE, CLOUD = 0, 1, 2
ACTION_NAMES = ("local", "edge", "cloud")


@dataclass
class EnvConfig:
    n_agents: int = 6
    horizon: int = 50
    arrival_prob: float = 0.8
    task_size: tuple = (0.5, 1.5)            # units of work (e.g. Gcycles)
    local_speeds: tuple = (0.6, 0.7, 0.8, 0.9, 1.0, 1.2)  # work per step, per agent
    edge_speed: float = 3.0                  # work per step, shared
    cloud_delay: float = 1.5                 # backhaul round trip, in steps
    cloud_speed: float = 10.0
    cloud_fee: float = 0.3
    data_ratio: float = 0.5                  # data to send per unit of work
    rate_good: float = 2.0                   # uplink rate, good channel
    rate_bad: float = 0.5                    # uplink rate, bad channel
    channel_stay: tuple = (0.9, 0.95, 0.8, 0.85, 0.9, 0.7)  # P(channel keeps its state)
    energy_coef: float = 0.5                 # local energy = k * f^2 * size
    tx_power: float = 0.3                    # transmit energy per step of upload
    energy_weight: float = 1.0
    edge_price: float = 0.0                  # set by the leader in the Stackelberg variant
    price_range: tuple = (0.0, 3.0)          # used to normalise the price in observations
    random_price: bool = False               # resample the price at each reset (follower training)
    signal_delay: int = 2                    # the public load signal is this many steps old (0 = exact)
    seed: int | None = None


class OffloadEnv:
    OBS_DIM = 9

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        c = self.cfg
        assert len(c.local_speeds) >= c.n_agents and len(c.channel_stay) >= c.n_agents
        self.agents = [f"device_{i}" for i in range(c.n_agents)]
        self.n = c.n_agents
        self.f_local = np.array(c.local_speeds[: self.n], dtype=float)
        self.stay = np.array(c.channel_stay[: self.n], dtype=float)
        self.rng = np.random.default_rng(c.seed)
        self.price = c.edge_price
        self.observation_dim = self.OBS_DIM
        self.state_dim = self.OBS_DIM * self.n + 1
        self.n_actions = 3

    # ------------------------------------------------------------------ helpers
    def _draw_tasks(self):
        has = self.rng.random(self.n) < self.cfg.arrival_prob
        size = self.rng.uniform(*self.cfg.task_size, size=self.n)
        return has.astype(float), np.where(has, size, 0.0)

    def _obs_array(self):
        c = self.cfg
        lo, hi = c.price_range
        price = (self.price - lo) / max(hi - lo, 1e-8)
        obs = np.stack([
            self.has_task,
            self.task / c.task_size[1],
            np.minimum(self.q_local / 5.0, 2.0),
            self.channel_good,
            self.f_local,
            np.full(self.n, min(self.backlog_hist[0] / 10.0, 2.0)),  # delayed public signal
            np.full(self.n, price),
            np.full(self.n, self.t / c.horizon),
            np.arange(self.n) / max(self.n - 1, 1),                    # agent index
        ], axis=1)
        return obs.astype(np.float32)

    def _obs(self):
        arr = self._obs_array()
        return {a: arr[i] for i, a in enumerate(self.agents)}

    def global_state(self):
        """Full state for a centralised critic (MAPPO)."""
        return np.concatenate([self._obs_array().ravel(), [self.edge_backlog / 10.0]]).astype(np.float32)

    # ---------------------------------------------------------------------- API
    def reset(self, seed: int | None = None, price: float | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        if price is not None:
            self.price = price
        elif c.random_price:
            self.price = float(self.rng.uniform(*c.price_range))
        else:
            self.price = c.edge_price
        self.t = 0
        self.q_local = np.zeros(self.n)
        self.edge_backlog = 0.0
        # backlog_hist[0] is the edge backlog at the start of step t - signal_delay
        self.backlog_hist = deque([0.0] * (c.signal_delay + 1), maxlen=c.signal_delay + 1)
        self.channel_good = (self.rng.random(self.n) < 0.7).astype(float)
        self.has_task, self.task = self._draw_tasks()
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        c = self.cfg
        act = np.array([int(actions[a]) for a in self.agents])
        latency = np.zeros(self.n)
        energy = np.zeros(self.n)
        payment = np.zeros(self.n)
        rate = np.where(self.channel_good > 0, c.rate_good, c.rate_bad)
        tx_time = c.data_ratio * self.task / rate

        # local execution: FIFO behind the device's own backlog
        loc = (self.has_task > 0) & (act == LOCAL)
        latency[loc] = (self.q_local[loc] + self.task[loc]) / self.f_local[loc]
        energy[loc] = c.energy_coef * self.f_local[loc] ** 2 * self.task[loc]
        self.q_local[loc] += self.task[loc]

        # edge execution: shared FIFO queue, random service order within a step
        edge_idx = np.where((self.has_task > 0) & (act == EDGE))[0]
        self.rng.shuffle(edge_idx)
        backlog = self.edge_backlog
        for i in edge_idx:
            backlog += self.task[i]
            latency[i] = tx_time[i] + backlog / c.edge_speed
            energy[i] = c.tx_power * tx_time[i]
            payment[i] = self.price

        # cloud execution: no congestion, fixed backhaul delay and fee
        cl = (self.has_task > 0) & (act == CLOUD)
        latency[cl] = tx_time[cl] + c.cloud_delay + self.task[cl] / c.cloud_speed
        energy[cl] = c.tx_power * tx_time[cl]
        payment[cl] = c.cloud_fee

        cost = latency + c.energy_weight * energy + payment
        rewards = -cost

        # dynamics
        self.edge_backlog = max(0.0, backlog - c.edge_speed)
        self.backlog_hist.append(self.edge_backlog)
        self.q_local = np.maximum(0.0, self.q_local - self.f_local)
        flip = self.rng.random(self.n) > self.stay
        self.channel_good = np.where(flip, 1.0 - self.channel_good, self.channel_good)
        n_edge = len(edge_idx)
        revenue = self.price * n_edge
        info_common = {
            "latency": latency, "energy": energy, "payment": payment, "actions": act,
            "had_task": self.has_task.copy(), "edge_backlog": self.edge_backlog,
            "revenue": revenue, "n_edge": n_edge,
        }
        self.t += 1
        self.has_task, self.task = self._draw_tasks()
        done = self.t >= c.horizon
        obs = self._obs()
        rew = {a: float(rewards[i]) for i, a in enumerate(self.agents)}
        term = {a: False for a in self.agents}
        trunc = {a: done for a in self.agents}
        infos = {a: info_common for a in self.agents}
        return obs, rew, term, trunc, infos
