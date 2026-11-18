import numpy as np

from marl_offload.baselines import FixedPolicy, GreedyPolicy, evaluate
from marl_offload.env import CLOUD, EDGE, LOCAL, EnvConfig, OffloadEnv


def test_shapes_and_horizon():
    env = OffloadEnv(EnvConfig(seed=0))
    obs, _ = env.reset(seed=0)
    assert set(obs) == set(env.agents)
    assert all(o.shape == (env.observation_dim,) for o in obs.values())
    assert env.global_state().shape == (env.state_dim,)
    for t in range(env.cfg.horizon):
        obs, rew, term, trunc, _ = env.step({a: LOCAL for a in env.agents})
    assert all(trunc.values())


def test_reset_is_reproducible():
    env = OffloadEnv()
    a, _ = env.reset(seed=3)
    b, _ = env.reset(seed=3)
    assert all(np.allclose(a[k], b[k]) for k in a)


def test_edge_congestion_grows_with_users():
    """More devices on the edge in the same step means a higher latency for the last one."""
    cfg = EnvConfig(arrival_prob=1.0, task_size=(1.0, 1.0))
    env = OffloadEnv(cfg)
    env.reset(seed=0)
    env.channel_good[:] = 1.0  # same channel for everybody, so upload times are equal
    _, _, _, _, info = env.step({a: EDGE for a in env.agents})
    lat = np.sort(next(iter(info.values()))["latency"])
    tx = cfg.data_ratio * 1.0 / cfg.rate_good
    expected = tx + np.arange(1, env.n + 1) * 1.0 / cfg.edge_speed  # k-th in line waits for k tasks
    assert np.allclose(lat, expected)


def test_cloud_has_no_congestion():
    cfg = EnvConfig(arrival_prob=1.0, task_size=(1.0, 1.0))
    env = OffloadEnv(cfg)
    env.reset(seed=0)
    env.channel_good[:] = 1.0
    _, _, _, _, info = env.step({a: CLOUD for a in env.agents})
    lat = next(iter(info.values()))["latency"]
    assert np.ptp(lat) < 1e-9


def test_load_signal_is_delayed():
    """The public signal shows the backlog from `signal_delay` steps ago, not the current one."""
    for delay in (0, 1, 3):
        env = OffloadEnv(EnvConfig(signal_delay=delay, arrival_prob=1.0))
        env.reset(seed=0)
        history = [env.edge_backlog]
        for _ in range(6):
            obs, *_ = env.step({a: EDGE for a in env.agents})
            history.append(env.edge_backlog)
            expected = history[max(len(history) - 1 - delay, 0)]
            assert np.isclose(obs["device_0"][5], min(expected / 10.0, 2.0))


def test_price_resets_to_config_default():
    env = OffloadEnv(EnvConfig(edge_price=0.0))
    env.reset(seed=0, price=2.0)
    assert env.price == 2.0
    env.reset(seed=0)
    assert env.price == 0.0


def test_social_cost_excludes_edge_price():
    env = OffloadEnv()
    r = evaluate(env, FixedPolicy(EDGE), n_episodes=2, price=1.0)
    assert np.isclose(r["cost_per_task"] - r["social_cost_per_task"], 1.0)


def test_greedy_beats_trivial_policies():
    env = OffloadEnv()
    g = evaluate(env, GreedyPolicy(), n_episodes=10)["cost_per_task"]
    for a in (LOCAL, EDGE, CLOUD):
        assert g < evaluate(env, FixedPolicy(a), n_episodes=10)["cost_per_task"]
