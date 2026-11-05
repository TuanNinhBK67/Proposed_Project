import math

import pytest
import torch

from simfas import fas
from simfas.action_maps import phase_from_latent, physical_from_unit, power_from_latent, unit_action_dim
from simfas.config import load_config
from simfas.env import SimFasEnv


def test_softmax_power_is_feasible_and_can_leave_power_unused():
    pmax = 0.1
    z = torch.cat([torch.randn(500, 3, 5) * 20, torch.full((1, 3, 5), 80.0), torch.full((1, 3, 5), -80.0)])
    p = power_from_latent(z, pmax)
    assert p.shape == (502, 3, 4) and (p >= 0).all() and (p.sum(-1) <= pmax * (1 + 1e-12)).all()
    unused = torch.zeros(1, 1, 5)
    unused[..., -1] = 30.0                                   # all the mass on the unused share
    assert power_from_latent(unused, pmax).sum() < 1e-10 * pmax
    full = torch.zeros(1, 1, 5)
    full[..., -1] = -30.0
    assert power_from_latent(full, pmax).sum() > pmax * (1 - 1e-10)


def test_phase_map_in_range():
    z = torch.cat([torch.randn(100, 32) * 10, torch.full((1, 32), 1e4), torch.full((1, 32), -1e4)])
    phi = phase_from_latent(z, 2, 16)
    assert phi.shape == (102, 2, 16) and (phi >= 0).all() and (phi <= 2 * math.pi).all()


def test_unit_actions_map_to_valid_env_actions():
    cfg = load_config("configs/smoke.yaml")
    env = SimFasEnv(cfg, 6, 0)
    env.reset()
    s = cfg.system
    a = torch.rand(6, unit_action_dim(s.L, s.U, s.M, s.N, s.K, s.Np)) * 2 - 1
    a[0], a[1] = 1.0, -1.0                                   # extreme corners
    p, phase, ports = physical_from_unit(a, cfg, env.pmax, cfg.offpolicy.power_scale)
    env.validate_action(p, torch.remainder(phase, 2 * math.pi), ports)
    fas.validate_ports(ports, env.slot_valid, s.Np)
    assert (phase >= 0).all() and (phase <= 2 * math.pi).all()
    env.step(p, phase, ports)



def test_linear_power_map_is_per_feed_feasible_and_rescales_only_above_the_budget():
    pmax = 0.2
    p = power_from_latent(torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 1.0], [5.0, -7.0]]), pmax, "linear")
    want = torch.tensor([[0.0, 0.0], [0.0, pmax], [pmax / 2, 0.0], [pmax / 2, pmax / 2], [pmax, 0.0]], dtype=p.dtype)
    assert torch.allclose(p, want)                           # (a+1)/2 P_max per feed, clipped; [1, 1] -> rescaled
    z = torch.randn(1000, 6, 2) * 2
    p = power_from_latent(z, pmax, "linear")
    assert (p >= 0).all() and (p.sum(-1) <= pmax * (1 + 1e-12)).all()
    raw = pmax * (z.double().clamp(-1, 1) + 1) / 2
    under = raw.sum(-1) <= pmax                              # below the budget the map is the plain linear one
    assert torch.allclose(p[under], raw[under])
    assert torch.allclose(p[~under].sum(-1), torch.full_like(p[~under].sum(-1), pmax))
    assert torch.allclose((p / p.sum(-1, keepdim=True))[~under], (raw / raw.sum(-1, keepdim=True))[~under])


@pytest.mark.parametrize("algo", ["mappo", "ppo", "ddpg", "random"])
def test_linear_power_map_reaches_the_environment(algo):
    """env.power_map = linear: every learner and the random baseline use U per-feed levels and stay within P_max."""
    from simfas.baselines import RandomPolicy
    from simfas.mappo.trainer import OnPolicyTrainer, active_mask, to_env_action
    from simfas.offpolicy.trainer import OffPolicyTrainer
    cfg = load_config("configs/smoke.yaml", ["env.power_map=linear"])
    s = cfg.system
    if algo in ("mappo", "ppo"):
        tr = OnPolicyTrainer(cfg, algo, None, "cpu")
        env, o = tr.env, tr.env.reset()
        x = tr.state_norm(env.global_state(), update=False)[:, None] if algo == "ppo" else tr.obs_norm(o["obs"], update=False)
        a = tr.actor.act(x, active_mask(env, algo == "ppo", True))
        assert a["z_power"].shape[-1] == s.U
        p = to_env_action(a, env, cfg, algo == "ppo")[0]
    elif algo == "ddpg":
        assert unit_action_dim(s.L, s.U, s.M, s.N, s.K, s.Np, "linear") == unit_action_dim(s.L, s.U, s.M, s.N, s.K, s.Np) - s.L
        tr = OffPolicyTrainer(cfg, algo, None, "cpu")
        env = tr.env
        env.reset()
        p = physical_from_unit(torch.rand(env.E, tr.act_dim) * 2 - 1, cfg, env.pmax, cfg.offpolicy.power_scale)[0]
    else:
        env = SimFasEnv(cfg, 4, 0)
        p = RandomPolicy(0)(env, env.reset())[0]
    assert (p >= 0).all() and (p.sum(-1) <= env.pmax * (1 + 1e-9)).all()
    env.validate_action(p, torch.zeros(env.E, s.L, s.M, s.N), env.ports)
