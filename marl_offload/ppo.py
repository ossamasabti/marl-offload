"""IPPO and MAPPO with parameter sharing.

Both use one actor shared by all devices (the agent index is part of the
observation, so the policy can still differ between asymmetric devices).
They differ only in the critic:

* IPPO: each agent's critic sees its own observation, V(o_i).
* MAPPO: centralised training, decentralised execution. The critic sees the
  global state (all observations plus the true edge backlog), V(s, i); the
  actor still only uses o_i at execution time.

Steps where a device has no task are kept in the trajectory (they carry the
dynamics) but masked out of the policy loss, since the action has no effect.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .env import OffloadEnv

# Training episodes use seeds from TRAIN_SEED upwards, far from the evaluation seeds
# (baselines.EVAL_SEED), so that no evaluation episode is ever seen in training.
TRAIN_SEED = 100_000_000


def mlp(inp, out, hidden=64):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))


@dataclass
class PPOConfig:
    centralized_critic: bool = False
    episodes_per_update: int = 8
    updates: int = 250
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.95
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    reward_scale: float = 1 / 3
    seed: int = 0


class PPOAgent:
    def __init__(self, env: OffloadEnv, cfg: PPOConfig):
        self.env, self.cfg = env, cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)  # minibatch shuffling, seeded for reproducibility
        self.n = env.n
        self.actor = mlp(env.observation_dim, env.n_actions)
        crit_in = env.state_dim + self.n if cfg.centralized_critic else env.observation_dim
        self.critic = mlp(crit_in, 1)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr)
        self.name = "MAPPO" if cfg.centralized_critic else "IPPO"

    # ----------------------------------------------------------- acting
    def _obs_tensor(self, obs):
        return torch.as_tensor(np.stack([obs[a] for a in self.env.agents]))

    def _critic_input(self, obs_t, state):
        if not self.cfg.centralized_critic:
            return obs_t
        s = torch.as_tensor(state).unsqueeze(0).expand(self.n, -1)
        return torch.cat([s, torch.eye(self.n)], dim=1)

    @torch.no_grad()
    def act(self, env, obs, greedy=True):
        logits = self.actor(self._obs_tensor(obs))
        a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        return {ag: int(a[i]) for i, ag in enumerate(env.agents)}

    # ---------------------------------------------------------- training
    def collect(self, rng_seed):
        env, c = self.env, self.cfg
        buf = {k: [] for k in ("obs", "cin", "act", "logp", "rew", "val", "mask")}
        for ep in range(c.episodes_per_update):
            obs, _ = env.reset(seed=int(TRAIN_SEED + rng_seed * 1000 + ep))
            done = False
            ep_buf = {k: [] for k in buf}
            while not done:
                o = self._obs_tensor(obs)
                cin = self._critic_input(o, env.global_state())
                with torch.no_grad():
                    dist = torch.distributions.Categorical(logits=self.actor(o))
                    a = dist.sample()
                    v = self.critic(cin).squeeze(-1)
                actions = {ag: int(a[i]) for i, ag in enumerate(env.agents)}
                obs, rew, _, trunc, _ = env.step(actions)
                ep_buf["obs"].append(o); ep_buf["cin"].append(cin); ep_buf["act"].append(a)
                ep_buf["logp"].append(dist.log_prob(a)); ep_buf["val"].append(v)
                ep_buf["rew"].append(torch.tensor([rew[ag] for ag in env.agents], dtype=torch.float32) * c.reward_scale)
                ep_buf["mask"].append(o[:, 0] > 0)
                done = all(trunc.values())
            for k in buf:
                buf[k].append(torch.stack(ep_buf[k]))  # (T, N, ...)
        return {k: torch.stack(v) for k, v in buf.items()}  # (E, T, N, ...)

    def gae(self, rew, val):
        c = self.cfg
        E, T, N = rew.shape
        adv = torch.zeros_like(rew)
        last = torch.zeros(E, N)
        for t in reversed(range(T)):
            next_v = val[:, t + 1] if t + 1 < T else torch.zeros(E, N)  # finite horizon
            delta = rew[:, t] + c.gamma * next_v - val[:, t]
            last = delta + c.gamma * c.lam * last
            adv[:, t] = last
        return adv, adv + val

    def update(self, batch):
        c = self.cfg
        adv, ret = self.gae(batch["rew"], batch["val"])
        flat = lambda x: x.reshape(-1, *x.shape[3:])
        obs, cin, act, logp_old = flat(batch["obs"]), flat(batch["cin"]), flat(batch["act"]), flat(batch["logp"])
        adv, ret, mask = flat(adv), flat(ret), flat(batch["mask"]).float()
        adv = (adv - adv[mask > 0].mean()) / (adv[mask > 0].std() + 1e-8)
        idx = np.arange(len(obs))
        mb = len(obs) // c.minibatches
        for _ in range(c.epochs):
            self.rng.shuffle(idx)
            for s in range(0, len(obs), mb):
                j = torch.as_tensor(idx[s:s + mb])
                dist = torch.distributions.Categorical(logits=self.actor(obs[j]))
                logp = dist.log_prob(act[j])
                ratio = torch.exp(logp - logp_old[j])
                m = mask[j]
                pg = -torch.min(ratio * adv[j], ratio.clamp(1 - c.clip, 1 + c.clip) * adv[j])
                pg = (pg * m).sum() / m.sum().clamp(min=1)
                ent = (dist.entropy() * m).sum() / m.sum().clamp(min=1)
                vloss = ((self.critic(cin[j]).squeeze(-1) - ret[j]) ** 2).mean()
                loss = pg + c.value_coef * vloss - c.entropy_coef * ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.opt.step()

    def train(self, eval_fn=None, eval_every=10, log=print):
        history = []
        for u in range(self.cfg.updates):
            batch = self.collect(rng_seed=self.cfg.seed * 100_000 + u)
            self.update(batch)
            if eval_fn is not None and (u % eval_every == 0 or u == self.cfg.updates - 1):
                r = eval_fn(self)
                r["update"] = u
                r["env_steps"] = (u + 1) * self.cfg.episodes_per_update * self.env.cfg.horizon
                history.append(r)
                log(f"[{self.name}] update {u:4d}  cost/task {r['cost_per_task']:.3f}  "
                    f"shares L/E/C {r['share_local']:.2f}/{r['share_edge']:.2f}/{r['share_cloud']:.2f}")
        return history

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, weights_only=True)
        self.actor.load_state_dict(d["actor"]); self.critic.load_state_dict(d["critic"])
