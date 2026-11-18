"""Independent Q-learning (deep, with parameter sharing).

Each device treats the others as part of the environment and learns
Q(o_i, a) with a shared network, a replay buffer and a target network.
This is the simplest MARL baseline: the environment is non-stationary from
each agent's point of view, which is exactly what makes coordination hard.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .env import OffloadEnv
from .ppo import TRAIN_SEED, mlp


@dataclass
class IQLConfig:
    episodes: int = 2000
    buffer: int = 100_000
    batch: int = 256
    gamma: float = 0.95
    lr: float = 5e-4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.6
    target_every: int = 500
    train_every: int = 2
    reward_scale: float = 1 / 3
    seed: int = 0


class IQLAgent:
    name = "IQL"

    def __init__(self, env: OffloadEnv, cfg: IQLConfig):
        self.env, self.cfg = env, cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.q = mlp(env.observation_dim, env.n_actions)
        self.target = mlp(env.observation_dim, env.n_actions)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        d = env.observation_dim
        self.B = {"o": np.zeros((cfg.buffer, d), np.float32), "a": np.zeros(cfg.buffer, np.int64),
                  "r": np.zeros(cfg.buffer, np.float32), "o2": np.zeros((cfg.buffer, d), np.float32),
                  "done": np.zeros(cfg.buffer, np.float32)}
        self.ptr = self.size = 0

    @torch.no_grad()
    def act(self, env, obs, eps=0.0):
        o = torch.as_tensor(np.stack([obs[a] for a in env.agents]))
        a = self.q(o).argmax(-1).numpy()
        rand = self.rng.random(len(a)) < eps
        a[rand] = self.rng.integers(3, size=rand.sum())
        return {ag: int(a[i]) for i, ag in enumerate(env.agents)}

    def _store(self, o, a, r, o2, done):
        # Steps without a task are stored too (reward 0, action irrelevant). Skipping
        # them would leave Q untrained on no-task observations, yet those observations
        # appear as next states o2 and are used in the bootstrap target.
        for i in range(len(a)):
            p = self.ptr
            self.B["o"][p], self.B["a"][p], self.B["r"][p], self.B["o2"][p], self.B["done"][p] = o[i], a[i], r[i], o2[i], done
            self.ptr = (p + 1) % self.cfg.buffer
            self.size = min(self.size + 1, self.cfg.buffer)

    def _learn(self):
        c = self.cfg
        j = self.rng.integers(self.size, size=c.batch)
        o, a, r = map(torch.as_tensor, (self.B["o"][j], self.B["a"][j], self.B["r"][j]))
        o2, done = torch.as_tensor(self.B["o2"][j]), torch.as_tensor(self.B["done"][j])
        with torch.no_grad():
            # double DQN target
            a2 = self.q(o2).argmax(-1, keepdim=True)
            y = r + c.gamma * (1 - done) * self.target(o2).gather(1, a2).squeeze(1)
        qa = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(qa, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def train(self, eval_fn=None, eval_every=100, log=print):
        c, env = self.cfg, self.env
        history, step = [], 0
        decay_eps = c.episodes * c.eps_decay_frac
        for ep in range(c.episodes):
            eps = max(c.eps_end, c.eps_start - (c.eps_start - c.eps_end) * ep / decay_eps)
            obs, _ = env.reset(seed=TRAIN_SEED + c.seed * 1_000_000 + ep)
            done = False
            while not done:
                actions = self.act(env, obs, eps)
                o = np.stack([obs[a] for a in env.agents])
                obs, rew, _, trunc, _ = env.step(actions)
                done = all(trunc.values())
                o2 = np.stack([obs[a] for a in env.agents])
                r = np.array([rew[a] for a in env.agents]) * c.reward_scale
                self._store(o, np.array([actions[a] for a in env.agents]), r, o2, float(done))
                step += 1
                if self.size > c.batch and step % c.train_every == 0:
                    self._learn()
                if step % c.target_every == 0:
                    self.target.load_state_dict(self.q.state_dict())
            if eval_fn is not None and (ep % eval_every == 0 or ep == c.episodes - 1):
                r = eval_fn(self)
                r["update"] = ep
                r["env_steps"] = (ep + 1) * env.cfg.horizon
                history.append(r)
                log(f"[IQL] episode {ep:5d}  eps {eps:.2f}  cost/task {r['cost_per_task']:.3f}  "
                    f"shares L/E/C {r['share_local']:.2f}/{r['share_edge']:.2f}/{r['share_cloud']:.2f}")
        return history

    def save(self, path):
        torch.save({"q": self.q.state_dict()}, path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, weights_only=True)["q"])
        self.target.load_state_dict(self.q.state_dict())
