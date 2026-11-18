"""Smoke tests: the learners run end to end, save and reload, and are reproducible."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marl_offload.baselines import evaluate
from marl_offload.env import EnvConfig, OffloadEnv
from marl_offload.iql import IQLAgent, IQLConfig
from marl_offload.ppo import PPOAgent, PPOConfig


def test_iql_runs_and_reloads(tmp_path):
    env = OffloadEnv(EnvConfig(horizon=10))
    agent = IQLAgent(env, IQLConfig(episodes=40, batch=32, seed=0))
    agent.train(log=lambda *_: None)
    assert agent.size == 40 * 10 * env.n  # every agent-step is stored, with or without a task
    agent.save(tmp_path / "iql.pt")
    other = IQLAgent(env, IQLConfig(seed=1))
    other.load(tmp_path / "iql.pt")
    assert evaluate(env, agent, 2) == evaluate(env, other, 2)


def test_ppo_is_reproducible(tmp_path):
    def run(centralized):
        env = OffloadEnv(EnvConfig(horizon=10))
        agent = PPOAgent(env, PPOConfig(centralized_critic=centralized, updates=3, episodes_per_update=2, seed=0))
        agent.train(log=lambda *_: None)
        return agent
    for centralized in (False, True):
        a, b = run(centralized), run(centralized)
        for pa, pb in zip(a.actor.parameters(), b.actor.parameters()):
            assert torch.equal(pa, pb)
    a.save(tmp_path / "ppo.pt")
    a.load(tmp_path / "ppo.pt")
