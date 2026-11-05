import math

import pytest
import torch

from simfas import fas
from simfas.association import check_association
from simfas.baselines import RandomPolicy
from simfas.config import dims, load_config
from simfas.env import SimFasEnv
from simfas.evaluate import evaluate_policy
from simfas.reward import evaluate_link


def make_env(E=3, seed=0, overrides=()):
    cfg = load_config("configs/smoke.yaml", list(overrides))
    return SimFasEnv(cfg, E, seed), cfg


def rand_action(env, g):
    return RandomPolicy(int(torch.randint(0, 10 ** 6, (1,), generator=g)))(env, None)


@pytest.mark.parametrize("enc", ["gain_direction", "raw_snr"])
def test_dimensions_and_masks(enc):
    env, cfg = make_env(overrides=[f"env.channel_encoding={enc}"])
    o = env.reset()
    d, s = dims(cfg), cfg.system
    assert o["obs"].shape == (3, s.L, d["obs"]) and env.global_state().shape == (3, d["state"])
    assert (o["ctrl_mask"].sum(1) == 1).all()
    check_association(env.F)


def test_actor_obs_is_local_but_state_is_global():
    env, _ = make_env()
    o1 = env.reset()["obs"]
    s1 = env.global_state()
    env.Hbar = env.Hbar.clone()
    env.Hbar[:, 1:] = env.Hbar[:, 1:] * 3.0 + 1e-4            # change the CSI of APs 1..L-1 only
    o2, s2 = env.actor_obs(), env.global_state()
    assert torch.equal(o1[:, 0], o2[:, 0]) and not torch.equal(o1[:, 1], o2[:, 1])
    assert not torch.equal(s1, s2)


def test_actor_obs_hides_unconnected_ues():
    env, _ = make_env(E=16)
    o1 = env.reset()["obs"]
    conn = env.F.any(2)                                           # (E, L, K) AP l serves UE k
    assert not conn.all()                                         # some (AP, UE) pairs are unconnected
    assert torch.equal(o1[..., -env.K:], conn.float())            # connection flags at the end
    env.Hbar = env.Hbar.clone()
    env.Hbar[~conn] = env.Hbar[~conn] * 3.0 + 1e-4                # change only the CSI of unconnected pairs
    env.xi_prev = env.xi_prev + 1.0                               # and the QoS feedback of every UE
    o2 = env.actor_obs()
    for e in range(env.E):
        for l in range(env.L):
            unaware = ~conn[e, l]
            if unaware.any():                                     # UEs this AP serves still look the same
                d = o2[e, l] - o1[e, l]
                C = dims(env.cfg)["obs_channel"][1]
                assert (d[:C].view(env.K, -1)[unaware] == 0).all()


def test_obs_scope_all_keeps_old_layout():
    env, cfg = make_env(overrides=["env.obs_ue_scope=all"])
    assert env.reset()["obs"].shape[-1] == dims(cfg)["obs"]
    cfg2 = load_config("configs/smoke.yaml")
    assert dims(cfg2)["obs"] == dims(cfg)["obs"] + cfg.system.K


def test_control_ap_always_serves_its_ue():
    env, _ = make_env(E=256)
    env.reset()
    serve = env.F.any(2)
    assert torch.gather(serve, 1, env.ctrl[:, None, :])[:, 0].all()


@pytest.mark.parametrize("scope", ["connected", "all"])
def test_ue_feature_index_picks_each_ues_features(scope):
    from simfas.config import ue_feature_index
    env, cfg = make_env(E=4, overrides=[f"env.obs_ue_scope={scope}"])
    o = env.reset()["obs"]
    K, Np, U, L = env.K, env.Np, env.U, env.L
    C = env.channel_features().float()                                       # (E, L, K, ch)
    hot = fas.multi_hot(env.ports, env.slot_valid, Np)
    xi = (env.xi_prev / (env.r_min + 1e-6)).float()
    conn = env.F.any(2)
    idx = torch.tensor(ue_feature_index(cfg, central=False))
    ch = C.shape[-1]
    for k in range(K):
        f = o[..., idx[k]]                                                    # (E, L, d_ue)
        own = env.ctrl_mask[:, :, k]
        seen = conn[:, :, k] if scope == "connected" else torch.ones_like(own)
        assert torch.allclose(f[..., :ch], C[:, :, k] * seen[..., None])
        assert torch.equal(f[..., ch:ch + U], env.F[:, :, :, k].float())
        assert torch.equal(f[..., ch + U:ch + U + Np], hot[:, None, k] * own[..., None].float())
        assert torch.equal(f[..., ch + U + Np], own.float())
        assert torch.allclose(f[..., ch + U + Np + 1], xi[:, None, k] * seen.float())
        if scope == "connected":
            assert torch.equal(f[..., -1], conn[:, :, k].float())
    s = env.global_state()
    idx = torch.tensor(ue_feature_index(cfg, central=True))
    for k in range(K):
        f = s[:, idx[k]]
        assert torch.allclose(f[:, :L], (env.D[:, :, k] / env.d_max).float())
        assert torch.allclose(f[:, L:L + L * ch], C[:, :, k].reshape(env.E, -1))
        assert torch.equal(f[:, -Np - 2:-2], hot[:, k])
        assert torch.allclose(f[:, -1], xi[:, k])


LEAN = ["env.channel_encoding=raw_snr", "env.phase_encoding=raw", "env.obs_ctrl_mask=false", "env.obs_qos=false",
        "env.state_distance=false"]


@pytest.mark.parametrize("port_gain,conn_flag", [(False, False), (False, True), (True, True)])
def test_lean_layout_blocks(port_gain, conn_flag):
    """Real/imag-only CSI, raw phases phi/2pi, no control mask / xi in the actor obs, no D in the state."""
    from simfas.config import _blocks, _offsets, ue_feature_index
    env, cfg = make_env(E=6, overrides=LEAN + [f"env.port_gain_obs={str(port_gain).lower()}",
                                               f"env.obs_conn_flag={str(conn_flag).lower()}"])
    o = env.reset()["obs"]
    s = env.global_state()
    K, L, U, Np = env.K, env.L, env.U, env.Np
    ob, st = dict(_blocks(cfg, False)), dict(_blocks(cfg, True))
    oo, so = _offsets(_blocks(cfg, False)), _offsets(_blocks(cfg, True))
    assert "ctrl" not in ob and "xi" not in ob and "D" not in st and so["csi"] == 0
    assert ("conn" in ob) == conn_flag and ("port_gain" in ob) == port_gain
    assert o.shape[-1] == dims(cfg)["obs"] and s.shape[-1] == dims(cfg)["state"]
    ch = 2 * Np * env.N
    assert ob["csi"] == K * ch and ob["phase"] == env.M * env.N
    conn = env.F.any(2)
    H = (env.Hbar * env.h_scale).reshape(env.E, L, K, -1)
    C = torch.cat([H.real, H.imag], -1).float()
    assert torch.allclose(o[..., :K * ch].view(env.E, L, K, ch), C * conn[..., None])
    ph = (env.phase / (2 * math.pi)).float()
    assert torch.allclose(o[..., oo["phase"]:oo["phase"] + ob["phase"]], ph.reshape(env.E, L, -1))
    assert torch.allclose(s[:, so["phase"]:so["phase"] + st["phase"]], ph.reshape(env.E, -1))
    assert (ph >= 0).all() and (ph < 1).all()
    env.xi_prev = env.xi_prev + 1.0                                 # QoS feedback reaches the state only
    assert torch.equal(env.actor_obs(), o) and not torch.equal(env.global_state(), s)
    for central in (False, True):
        idx = torch.tensor(ue_feature_index(cfg, central))
        x = s if central else o
        hot = fas.multi_hot(env.ports, env.slot_valid, Np)
        for k in range(K):
            f = x[..., idx[k]]
            if not central:
                assert torch.allclose(f[..., :ch], C[:, :, k] * conn[:, :, k, None])
                own = env.ctrl_mask[:, :, k, None].float()
                assert torch.equal(f[..., ch + U:ch + U + Np], hot[:, None, k] * own)
                if conn_flag:
                    assert torch.equal(f[..., ch + U + Np], conn[:, :, k].float())
                else:
                    assert f.shape[-1] == ch + U + Np + (Np + 1 if port_gain else 0)
            else:
                assert torch.allclose(f[:, :L * ch], C[:, :, k].reshape(env.E, -1))
                assert torch.equal(f[:, L * ch + L * U:L * ch + L * U + Np], hot[:, k])


def test_port_block_only_for_controlled_ues():
    env, _ = make_env()
    o = env.reset()["obs"]
    K, Np, N = env.K, env.Np, env.N
    off = dims(env.cfg)["obs_channel"][1] + env.U * K + env.U + 2 * env.M * N
    block = o[..., off:off + K * Np].view(env.E, env.L, K, Np)
    hot = fas.multi_hot(env.ports, env.slot_valid, Np)
    for l in range(env.L):
        own = env.ctrl_mask[:, l]
        assert torch.equal(block[:, l][own], hot[own]) and (block[:, l][~own] == 0).all()


@pytest.mark.parametrize("port_gain", ["false", "true"])
def test_state_time_flag_removes_only_the_step_index(port_gain):
    """env.state_time=false drops t/T and nothing else; the blocks after it and ue_feature_index stay aligned."""
    from simfas.config import _blocks, _offsets, ue_feature_index
    ov = [f"env.port_gain_obs={port_gain}"]
    on, cfg_on = make_env(seed=5, overrides=ov)
    off, cfg_off = make_env(seed=5, overrides=ov + ["env.state_time=false"])
    assert "t" in dict(_blocks(cfg_on, True)) and "t" not in dict(_blocks(cfg_off, True))
    assert dims(cfg_off)["state"] == dims(cfg_on)["state"] - 1 and dims(cfg_off)["obs"] == dims(cfg_on)["obs"]
    i = _offsets(_blocks(cfg_on, True))["t"]
    idx_on, idx_off = torch.tensor(ue_feature_index(cfg_on, True)), torch.tensor(ue_feature_index(cfg_off, True))
    o_on, o_off = on.reset()["obs"], off.reset()["obs"]
    g = torch.Generator().manual_seed(1)
    for t in range(3):
        s_on, s_off = on.global_state(), off.global_state()
        assert s_off.shape[-1] == dims(cfg_off)["state"]
        assert torch.allclose(s_on[:, i], torch.full((on.E,), t / on.T))
        assert torch.equal(s_off, torch.cat([s_on[:, :i], s_on[:, i + 1:]], -1))
        assert torch.equal(s_on[:, idx_on], s_off[:, idx_off])
        assert torch.equal(o_on, o_off)                                     # the actor obs never had t / T
        p, phase, ports = rand_action(on, g)
        o_on, o_off = on.step(p, phase, ports)[0]["obs"], off.step(p, phase, ports)[0]["obs"]


def test_benchmark_yamls_have_no_step_index_in_the_state():
    from simfas.config import _blocks, load_config
    for path, size in (("configs/bench_L6U2K4.yaml", 19632), ("configs/bench_L6U2K4_perhead.yaml", 19632),
                       ("configs/default.yaml", 9882)):
        cfg = load_config(path)
        assert "t" not in dict(_blocks(cfg, True)) and dims(cfg)["state"] == size


def test_time_indexing_and_fixed_channel_within_episode():
    env, cfg = make_env(seed=5)
    g = torch.Generator().manual_seed(1)
    env.reset()
    H0, F0, D0 = env.Hbar.clone(), env.F.clone(), env.D.clone()
    for t in range(env.T):
        assert torch.equal(env.Hbar, H0) and torch.equal(env.F, F0)          # fixed during the episode
        assert torch.allclose(env.global_state()[:, -1], torch.full((env.E,), t / env.T))
        p, phase, ports = rand_action(env, g)
        ref = evaluate_link(env.Hbar, ports, env.slot_valid, phase, p, env.F, env.Q1, env.Qinter, env.sigma2,
                            env.r_min, cfg.env.qos_penalty)
        _, info = env.step(p, phase, ports)
        assert torch.allclose(info["reward"], ref.reward, rtol=1e-5)      # r_t on the episode channel
        assert bool(info["terminated"].all()) == (t == env.T - 1)
        if t < env.T - 1:
            assert torch.allclose(env.xi_prev, ref.xi.double())           # feedback of step t for o_{t+1}
            assert torch.equal(env.ports, ports)                          # installed configuration = a_t
    assert not torch.equal(env.Hbar, H0) and not torch.equal(env.D, D0)   # new channel and geometry
    assert (env.t == 0).all()


def test_effective_channel_changes_with_phase_and_ports():
    env, cfg = make_env()
    env.reset()
    base = env._evaluate(env.p, env.phase, env.ports).sum_se
    # a common offset on every atom of every AP is only a global phase (SINR unchanged): perturb AP 0 only
    ph = env.phase.clone()
    ph[:, 0] = torch.remainder(ph[:, 0] + torch.rand(ph[:, 0].shape, dtype=ph.dtype) * 3, 2 * math.pi)
    rot = env._evaluate(env.p, ph, env.ports).sum_se
    common = env._evaluate(env.p, torch.remainder(env.phase + 1.0, 2 * math.pi), env.ports).sum_se
    assert torch.allclose(base, common, rtol=1e-4)
    other = env._evaluate(env.p, env.phase, fas.random_ports(env.E, env.nk, env.Np, env.n_max,
                                                             torch.Generator().manual_seed(9), "cpu")).sum_se
    assert not torch.allclose(base, rot) and not torch.allclose(base, other)


def test_infeasible_actions_rejected():
    env, _ = make_env()
    env.reset()
    p, phase, ports = rand_action(env, torch.Generator().manual_seed(0))
    with pytest.raises(ValueError):
        env.step(p * 1e3 + env.pmax, phase, ports)
    bad = ports.clone()
    bad[0, 2, 1] = bad[0, 2, 0]
    with pytest.raises(ValueError):
        env.step(p, phase, bad)


def test_merge_takes_only_controller_proposals():
    env, _ = make_env()
    env.reset()
    E, L, K, n = env.E, env.L, env.K, env.n_max
    prop = torch.full((E, L, K, n), 99, dtype=torch.long)
    want = fas.random_ports(E, env.nk, env.Np, n, torch.Generator().manual_seed(3), "cpu")
    for e in range(E):
        for k in range(K):
            prop[e, env.ctrl[e, k], k] = want[e, k]
    assert torch.equal(fas.merge_port_proposals(prop, env.ctrl), want)


def test_reproducible_scenarios_independent_of_policy():
    env1, _ = make_env(seed=7)
    env2, _ = make_env(seed=7)
    assert torch.equal(env1.reset()["obs"], env2.reset()["obs"])
    assert not torch.equal(make_env(seed=8)[0].reset()["obs"], env1.actor_obs())
    _, cfg = make_env()
    seen = []

    class Rec:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, env, o):
            if int(env.t[0]) == 0:
                seen.append(env.Hbar.clone())
            return self.inner(env, o)

    evaluate_policy(cfg, Rec(RandomPolicy(1)), 6, 11)
    first = list(seen)
    seen.clear()
    evaluate_policy(cfg, Rec(lambda env, o: (env.p, env.phase, env.ports)), 6, 11)
    assert all(torch.equal(a, b) for a, b in zip(first, seen))


def test_gain_direction_encoding_is_lossless():
    env, _ = make_env()
    env.reset()
    C = env.channel_features().double()                        # (E, L, K, 2 Np N + 1 + Np)
    Np, N = env.Np, env.N
    Hn = torch.complex(C[..., :Np * N], C[..., Np * N:2 * Np * N]).view(*C.shape[:3], Np, N)
    rms = torch.sqrt(10 ** (C[..., 2 * Np * N] * 10 / 10) * env.sigma2 / env.pmax)
    assert torch.allclose(Hn * rms[..., None, None], env.Hbar.to(torch.complex128), rtol=1e-4, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_scenarios_identical_on_cpu_and_cuda():
    cfg = load_config("configs/smoke.yaml")
    a, b = SimFasEnv(cfg, 3, 4, "cpu"), SimFasEnv(cfg, 3, 4, "cuda")
    a.reset(), b.reset()
    assert torch.allclose(a.Hbar, b.Hbar.cpu(), rtol=1e-5, atol=1e-12) and torch.equal(a.ports, b.ports.cpu())



@pytest.mark.parametrize("central", [False, True])
def test_port_gain_observation_block(central):
    """env.port_gain_obs: actor rows carry the post-SIM per-port desired-signal gains of the UEs the AP controls
    (zeros otherwise), the critic state carries them for every UE, and ue_feature_index points at them."""
    import math
    from simfas.config import dims, ue_feature_index
    from simfas.phy import desired_port_power
    cfg = load_config("configs/smoke.yaml", ["env.port_gain_obs=true"])
    env = SimFasEnv(cfg, 3, 0)
    o = env.reset()
    d = dims(cfg)
    obs, st = o["obs"], env.global_state()
    assert obs.shape[-1] == d["obs"] and st.shape[-1] == d["state"]
    Np, K = env.Np, env.K
    g = desired_port_power(env.Hbar, env.Q1, env.Qinter, env.phase, env.p, env.F).double().clamp_min(env.sigma2 * 1e-6)
    ref = torch.cat([torch.log10(g / g.mean(-1, keepdim=True)), torch.log10(g.mean(-1, keepdim=True) / env.sigma2)], -1)
    blk = obs[..., -K * (Np + 1):].reshape(env.E, env.L, K, Np + 1).double()
    own = env.ctrl_mask[..., None].double()
    assert torch.allclose(blk, (ref[:, None] * own).float().double(), atol=1e-5)
    assert torch.allclose(st[:, -K * (Np + 1):].reshape(env.E, K, Np + 1).double(), ref.float().double(), atol=1e-5)
    x = st[:, None] if central else obs
    idx = ue_feature_index(cfg, central)
    for k in range(K):
        tail = x[..., idx[k][-(Np + 1):]]
        want = ref[:, None, k] if central else ref[:, None, k] * own[:, :, k]
        assert torch.allclose(tail.double(), want.float().double().expand_as(tail), atol=1e-5)


@pytest.mark.parametrize("scope", ["controlled", "served"])
def test_port_block_scope(scope):
    """obs_ports_scope: AP l sees the active ports of the UEs whose FAS it controls, or of every UE it serves."""
    from simfas.config import _blocks, _offsets
    env, cfg = make_env(E=16, overrides=LEAN + [f"env.obs_ports_scope={scope}", "env.obs_ctrl_mask=true"])
    o = env.reset()["obs"]
    K, Np = env.K, env.Np
    off, size = _offsets(_blocks(cfg, False)), dict(_blocks(cfg, False))
    block = o[..., off["ports"]:off["ports"] + size["ports"]].view(env.E, env.L, K, Np)
    hot = fas.multi_hot(env.ports, env.slot_valid, Np)
    seen = env.ctrl_mask if scope == "controlled" else env.F.any(2)
    for l in range(env.L):
        assert torch.equal(block[:, l][seen[:, l]], hot[seen[:, l]]) and (block[:, l][~seen[:, l]] == 0).all()
    assert torch.equal(o[..., off["ctrl"]:off["ctrl"] + K], env.ctrl_mask.float())
    if scope == "served":
        assert (env.F.any(2) & ~env.ctrl_mask).any()           # some UEs are served but not controlled by an AP


def test_served_port_scope_needs_the_control_mask():
    """With obs_ports_scope = served the port block shows UEs the AP does not control: the ctrl mask must be on."""
    with pytest.raises(ValueError):
        load_config("configs/smoke.yaml", ["env.obs_ports_scope=served", "env.obs_ctrl_mask=false"])
    load_config("configs/smoke.yaml", ["env.obs_ports_scope=controlled", "env.obs_ctrl_mask=false"])
    load_config("configs/smoke.yaml", ["env.obs_ports_scope=served", "env.obs_ctrl_mask=true"])


def bench_obs_flags(path="configs/bench_L6U2K4.yaml"):
    """The observation / state flags of a benchmark YAML as overrides for the smoke sizes (no physics values)."""
    import yaml
    env = yaml.safe_load(open(path))["env"]
    skip = {"episode_steps", "qos_penalty", "complex_dtype"}
    return [f"env.{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in env.items() if k not in skip]


def test_actor_obs_is_local_with_the_bench_obs_flags():
    """Bench layout (served ports + ctrl mask, raw_snr CSI, raw phases, no xi, no conn flags): changing AP 0's
    Hbar, SIM phases or powers changes only row 0 of actor_obs; the critic state sees every change."""
    ov = bench_obs_flags()
    assert "env.obs_ports_scope=served" in ov and "env.phase_encoding=raw" in ov
    env, cfg = make_env(E=8, overrides=ov)
    o = env.reset()["obs"]
    s = env.global_state()
    assert o.shape[-1] == dims(cfg)["obs"]
    for name, change in (("Hbar", lambda x: x * 3.0 + 1e-4), ("phase", lambda x: torch.remainder(x + 0.7, 2 * math.pi)),
                         ("p", lambda x: x * 0.5)):
        old = getattr(env, name)
        new = old.clone()
        new[:, 0] = change(old[:, 0])
        setattr(env, name, new)
        o2, s2 = env.actor_obs(), env.global_state()
        assert not torch.equal(o2[:, 0], o[:, 0]), name
        assert torch.equal(o2[:, 1:], o[:, 1:]), name             # other APs' observations unchanged
        assert not torch.equal(s2, s), name
        setattr(env, name, old)
    assert torch.equal(env.actor_obs(), o)
