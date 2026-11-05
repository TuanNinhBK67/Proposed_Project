import os
import sys

import pytest
import torch

from simfas.config import load_config
from simfas.evaluate import evaluate_policy, load_policy
from simfas.mappo.trainer import OnPolicyTrainer, gae
from simfas.offpolicy.trainer import OffPolicyTrainer, ReplayBuffer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_gae_by_hand_with_terminal():
    r = torch.tensor([[1.0], [2.0], [3.0]])
    v = torch.tensor([[0.5], [0.4], [0.3]])
    nv = torch.tensor([[0.4], [0.3], [9.0]])                   # V(s_3) must be ignored: terminal at t=2
    term = torch.tensor([[0.0], [0.0], [1.0]])
    g, lam = 0.9, 0.8
    adv, ret = gae(r, v, nv, term, g, lam)
    d2 = 3.0 - 0.3
    d1 = 2.0 + g * 0.3 - 0.4
    d0 = 1.0 + g * 0.4 - 0.5
    a2, a1 = d2, d1 + g * lam * d2
    assert torch.allclose(adv[:, 0], torch.tensor([d0 + g * lam * a1, a1, a2]))
    assert torch.allclose(ret, adv + v)


@pytest.mark.parametrize("algo,ov", [("mappo", []), ("mappo", ["ppo.shared_actor=false"]), ("ppo", []),
                                     ("mappo", ["ppo.fas_mode=fixed_random", "ppo.learn_power=false"]),
                                     ("mappo", ["ppo.per_head_clip=true"]), ("mappo", ["ppo.fas_mode=strongest"])])
def test_on_policy_update_finite_and_changes_parameters(algo, ov):
    cfg = load_config("configs/smoke.yaml", ov)
    tr = OnPolicyTrainer(cfg, algo, None, "cpu")
    before = [p.detach().clone() for p in tr.actor.parameters()]
    tr.collect()
    # stored actions re-score to the stored log-probabilities before any update (ratio = 1)
    b = tr.buf
    f = lambda x: x.reshape((-1,) + x.shape[2:])
    ev = tr.actor.evaluate(f(b["x"]), f(b["active"]), f(b["z_power"]), f(b["z_phase"]), f(b["draws"]))
    assert torch.allclose(ev["logp"], f(b["logp"]), atol=1e-4)
    assert torch.allclose(ev["logp_parts"], f(b["logp_parts"]), atol=1e-4)
    assert torch.allclose(f(b["logp_parts"]).sum(-1), f(b["logp"]), atol=1e-4)
    st = tr.update()
    assert all(torch.isfinite(torch.tensor(v)) for k, v in st.items() if k != "explained_var")
    assert any(not torch.equal(a, p) for a, p in zip(before, tr.actor.parameters()))


def test_replay_next_state_alignment_and_newest_row_excluded():
    rb = ReplayBuffer(capacity=12, E=2, state_dim=1, act_dim=1, device="cpu")
    for t in range(9):                                         # wraps around (6 rows of 2 envs)
        rb.add(torch.tensor([[t], [100 + t]], dtype=torch.float32), torch.zeros(2, 1), torch.zeros(2),
               torch.zeros(2))
    s, _, _, _, s2 = rb.sample(500, torch.Generator().manual_seed(0))
    assert torch.equal(s2 - s, torch.ones_like(s))             # successor = next row of the same env
    assert (s[s < 100] < 8).all() and (s[s >= 100] < 108).all()   # newest row never sampled as s


@pytest.mark.parametrize("algo", ["ddpg", "td3"])
def test_off_policy_updates_finite(algo):
    cfg = load_config("configs/smoke.yaml")
    tr = OffPolicyTrainer(cfg, algo, None, "cpu")
    fin = []
    for _ in range(40):
        tr.env_step(fin)
    assert fin and tr.replay.size == 40
    before = [p.detach().clone() for p in tr.actor.parameters()]
    for _ in range(4):
        out = tr.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values())
    assert any(not torch.equal(a, p) for a, p in zip(before, tr.actor.parameters()))


@pytest.mark.parametrize("algo", ["mappo", "td3", "random", "rp_ep", "rp_ep_sp", "mappo_ep", "mappo_rp", "mappo_rport",
                                  "mappo_sp", "mappo_sp_ep", "mappo_phc"])
def test_smoke_training_writes_outputs_and_best_model_reproduces(tmp_path, algo):
    import train
    train.main(["--config", "configs/smoke.yaml", "--algo", algo, "--out", str(tmp_path)])
    rd = tmp_path / algo / "seed0"
    for f in ("config.yaml", "metrics.csv", "best_model.pt", "final_model.pt", "final.json"):
        assert (rd / f).exists(), f
    if algo not in ("random", "rp_ep", "rp_ep_sp"):
        import csv
        rows = list(csv.DictReader(open(rd / "episodes.csv")))
        T = load_config("configs/smoke.yaml").env.episode_steps
        assert len(rows) == int(rows[-1]["episode"]) and int(rows[-1]["env_steps"]) == len(rows) * T
        assert {"ep_sum_se", "ep_reward", "ep_qos_satisfied", "final_sum_se"} <= set(rows[0])
    # the run redraws the benchmark figure of its output root at the end (LivePlot)
    from simfas.plotting import METHODS
    labels = dict((k, lb) for k, lb, _ in METHODS)
    if algo in labels:                                       # rp_ep / rp_ep_sp runs are not drawn
        fig = tmp_path / "figures"
        assert (fig / "convergence_sumrate.png").exists() and (fig / "convergence_sumrate.pdf").exists()
        import csv
        table = list(csv.reader(open(fig / "convergence_sumrate.csv")))
        assert table[0] == ["steps", labels[algo]] and float(table[-1][1]) > 0
    policy, cfg, d = load_policy(str(rd / "best_model.pt"))
    ev = evaluate_policy(cfg, policy, cfg.training.eval_episodes, cfg.seed + cfg.training.eval_seed_offset)
    assert abs(ev["final_reward"] - d["eval"]["final_reward"]) < 1e-6


def test_strongest_ports_policy_picks_the_top_desired_power_ports():
    from simfas.baselines import StrongestPortsPolicy, desired_port_power
    from simfas.env import SimFasEnv
    cfg = load_config("configs/smoke.yaml")
    env = SimFasEnv(cfg, 4, 0, "cpu")
    o = env.reset()
    p, phase, ports = StrongestPortsPolicy()(env, o)
    assert torch.equal(p, env.p) and torch.equal(phase, env.phase)
    scores = desired_port_power(env, env.p, env.phase)
    for e in range(env.E):
        for k, nk in enumerate(env.nk):
            chosen = ports[e, k, :nk]
            assert (ports[e, k, nk:] == -1).all() and len(set(chosen.tolist())) == nk
            assert scores[e, k, chosen].min() >= scores[e, k].sort(descending=True).values[nk - 1] - 1e-12
    # the desired-signal power of the chosen ports matches the full PHY chain (interference-free)
    from simfas.phy import effective_channel, stream_channels
    d = stream_channels(effective_channel(env.Hbar, ports, env.slot_valid, phase, env.Q1, env.Qinter), p, env.F)
    own = d[:, torch.arange(env.K), torch.arange(env.K)].abs() ** 2                # (E, K, n_max)
    ref = torch.gather(scores, 2, ports.clamp_min(0)) * env.slot_valid
    assert torch.allclose(own.double(), ref.double(), rtol=1e-4, atol=1e-30)


@pytest.mark.parametrize("flag,fixed", [("ppo.learn_power=false", "p"), ("ppo.learn_phase=false", "phase"),
                                        ("ppo.fas_mode=fixed_random", "ports")])
def test_mappo_variant_keeps_its_fixed_variable_at_the_episode_start_value(flag, fixed):
    """MAPPO variants: the fixed variable stays at the episode's initial value (equal power, random phases,
    random ports) for all T steps, while the policy changes the other two."""
    cfg = load_config("configs/smoke.yaml", [flag])
    tr = OnPolicyTrainer(cfg, "mappo", None, "cpu")
    env = tr.env
    from simfas.mappo.trainer import to_env_action, active_mask
    x = tr.obs_norm(env.reset()["obs"], update=False)
    init = {"p": env.p.clone(), "phase": env.phase.clone(), "ports": env.ports.clone()}
    for _ in range(env.T - 1):
        a = tr.actor.act(x, active_mask(env, False, tr.fas_learn))
        act = dict(zip(("p", "phase", "ports"), to_env_action(a, env, cfg, False)))
        assert torch.equal(act[fixed], init[fixed])
        assert any(not torch.equal(act[k], init[k]) for k in act if k != fixed)
        o, _ = env.step(act["p"], act["phase"], act["ports"])
        x = tr.obs_norm(o["obs"], update=False)


def test_per_head_clip_matches_joint_ratio_gradient_at_ratio_one():
    """At the old policy (all ratios 1) the per-head surrogate and the joint surrogate have the same gradient."""
    cfg = load_config("configs/smoke.yaml")
    grads = []
    for phc in (False, True):
        tr = OnPolicyTrainer(load_config("configs/smoke.yaml", [f"ppo.per_head_clip={str(phc).lower()}"]), "mappo",
                             None, "cpu")
        tr.collect()
        b = tr.buf
        f = lambda x: x.reshape((-1,) + x.shape[2:])
        ev = tr.actor.evaluate(f(b["x"]), f(b["active"]), f(b["z_power"]), f(b["z_phase"]), f(b["draws"]))
        a = torch.randn(ev["logp"].shape[0], 1, generator=torch.Generator().manual_seed(1))
        if phc:
            loss = -(a[..., None] * (ev["logp_parts"] - f(b["logp_parts"])).exp()).sum(-1).mean()
        else:
            loss = -(a * (ev["logp"] - f(b["logp"])).exp()).mean()
        tr.actor.zero_grad()
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in tr.actor.parameters() if p.grad is not None]))
    assert cfg.ppo.per_head_clip is False
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_strongest_port_mode_uses_the_port_gains_of_the_configuration_in_effect():
    from simfas.baselines import desired_port_power
    from simfas.fas import topk_ports
    from simfas.mappo.trainer import active_mask, to_env_action
    cfg = load_config("configs/smoke.yaml", ["ppo.fas_mode=strongest"])
    tr = OnPolicyTrainer(cfg, "mappo", None, "cpu")
    env = tr.env
    o = env.reset()
    for _ in range(3):
        act = active_mask(env, False, tr.fas_learn)
        assert not act.any()                                   # the port head makes no draws
        a = tr.actor.act(tr.obs_norm(o["obs"], update=False), act)
        p, phase, ports = to_env_action(a, env, cfg, False)
        assert torch.equal(ports, topk_ports(desired_port_power(env, env.p, env.phase), env.nk, env.n_max))
        o, _ = env.step(p, phase, ports)


def test_linear_std_schedule_sets_the_std_and_keeps_it_out_of_the_optimizer():
    import math
    cfg = load_config("configs/smoke.yaml", ["ppo.std_schedule=linear", "ppo.final_std=0.05"])
    tr = OnPolicyTrainer(cfg, "mappo", None, "cpu")
    s0 = math.exp(cfg.ppo.init_log_std)
    assert abs(tr.scheduled_std(0.0) - s0) < 1e-12 and abs(tr.scheduled_std(1.0) - 0.05) < 1e-12
    assert abs(tr.scheduled_std(0.5) - (s0 + 0.05) / 2) < 1e-12
    tr._iteration(0.5)
    ps, fs = tr.actor.mean_std()
    assert abs(ps - (s0 + 0.05) / 2) < 1e-6 and abs(fs - (s0 + 0.05) / 2) < 1e-6   # unchanged by the update


def test_head_loss_weights_and_head_target_kl_options_are_validated():
    for ov in (["ppo.head_loss_weights=[1.0,2.0,1.0]"], ["ppo.head_target_kl=[0.02,0.02,0.02]"],
               ["ppo.per_head_clip=true", "ppo.head_loss_weights=[1.0,-1.0,1.0]"],
               ["ppo.per_head_clip=true", "ppo.head_target_kl=[0.02,0.0,0.02]"],
               ["ppo.per_head_clip=true", "ppo.head_target_kl=[0.02,0.02]"]):
        with pytest.raises(ValueError):
            load_config("configs/smoke.yaml", ov)


def test_ppo_and_gae_hyper_parameters_are_range_checked():
    """Eq. (35) delta in [0, 1), Eq. (37) epsilon > 0, and no empty rollout / minibatch / epoch loop."""
    for ov in ("ppo.discount=1.0", "ppo.discount=-0.1", "offpolicy.discount=1.0", "ppo.gae_lambda=1.5",
               "ppo.gae_lambda=-0.1", "ppo.clip=0", "ppo.ppo_epochs=0", "ppo.rollout_steps=0",
               "ppo.minibatch_size=0", "ppo.minibatch_size=1", "ppo.num_envs=0", "offpolicy.num_envs=0"):
        with pytest.raises(ValueError):
            load_config("configs/smoke.yaml", [ov])
    for ov in ("ppo.discount=0.0", "ppo.gae_lambda=1.0", "ppo.gae_lambda=0.0", "ppo.minibatch_size=2"):
        load_config("configs/smoke.yaml", [ov])                # range edges are accepted
    import glob
    for path in glob.glob("configs/*.yaml"):                   # every shipped config passes
        load_config(path)


@pytest.mark.parametrize("mb,sizes", [(31, [32]), (10, [10, 10, 10, 2]), (16, [16, 16]), (64, [32])])
def test_one_sample_tail_minibatch_is_merged_into_the_previous_one(mb, sizes):
    """n = 4 envs x 8 steps = 32: a 1-sample tail (no advantage std -> NaN) joins the previous minibatch; every
    other split is the plain range(0, n, mb) one."""
    tr = OnPolicyTrainer(load_config("configs/smoke.yaml", [f"ppo.minibatch_size={mb}", "ppo.target_kl=null"]),
                         "mappo", None, "cpu")
    tr.collect()
    seen, evaluate = [], tr.actor.evaluate
    tr.actor.evaluate = lambda x, *a: seen.append(x.shape[0]) or evaluate(x, *a)
    st = tr.update()
    assert seen == sizes * tr.cfg.ppo.ppo_epochs
    assert all(torch.isfinite(torch.tensor(v)) for k, v in st.items() if k != "explained_var")


def test_head_loss_weight_scales_only_that_heads_gradient():
    """A zero weight removes one head's surrogate: its own head parameters get no policy gradient."""
    grads = {}
    for w in ("[1.0,1.0,1.0]", "[1.0,0.0,1.0]"):
        cfg = load_config("configs/smoke.yaml", ["ppo.per_head_clip=true", f"ppo.head_loss_weights={w}",
                                                 "ppo.ppo_epochs=1", "ppo.max_grad_norm=1e9"])
        tr = OnPolicyTrainer(cfg, "mappo", None, "cpu")
        tr.collect()
        torch.manual_seed(0)
        before = {n: p.detach().clone() for n, p in tr.actor.named_parameters()}
        tr.update()
        grads[w] = {n: (p.detach() - before[n]).abs().max().item() for n, p in tr.actor.named_parameters()}
    z = grads["[1.0,0.0,1.0]"]
    assert z["nets.0.phase_mean.weight"] == 0 and z["nets.0.phase_log_std"] == 0
    assert grads["[1.0,1.0,1.0]"]["nets.0.phase_mean.weight"] > 0 and z["nets.0.power_mean.weight"] > 0


def test_head_target_kl_stops_only_the_head_that_exceeds_it():
    cfg = load_config("configs/smoke.yaml", ["ppo.per_head_clip=true", "ppo.target_kl=null", "ppo.ppo_epochs=4",
                                             "ppo.head_target_kl=[1e9,1e-12,1e9]"])
    tr = OnPolicyTrainer(cfg, "mappo", None, "cpu")
    tr.collect()
    st = tr.update()
    assert st["ppo_epochs_phase"] == 1 and st["ppo_epochs_power"] == 4 and st["ppo_epochs_fas"] == 4
    assert st["ppo_epochs_run"] == 4 and st["approx_kl_phase"] > 0
    cfg = load_config("configs/smoke.yaml", ["ppo.per_head_clip=true", "ppo.target_kl=null", "ppo.ppo_epochs=4",
                                             "ppo.head_target_kl=[1e-12,1e-12,1e-12]"])
    tr = OnPolicyTrainer(cfg, "mappo", None, "cpu")
    tr.collect()
    assert tr.update()["ppo_epochs_run"] == 1                  # every learned head stopped -> update ends


def test_strongest_new_port_mode_uses_the_port_gains_of_the_new_action():
    from simfas.baselines import desired_port_power
    from simfas.fas import topk_ports
    from simfas.mappo.trainer import active_mask, to_env_action
    cfg = load_config("configs/smoke.yaml", ["ppo.fas_mode=strongest_new"])
    tr = OnPolicyTrainer(cfg, "mappo", None, "cpu")
    env = tr.env
    o = env.reset()
    for _ in range(3):
        act = active_mask(env, False, tr.fas_learn)
        assert not act.any()                                   # the port head makes no draws
        a = tr.actor.act(tr.obs_norm(o["obs"], update=False), act)
        p, phase, ports = to_env_action(a, env, cfg, False)
        assert torch.equal(ports, topk_ports(desired_port_power(env, p, phase), env.nk, env.n_max))
        o, _ = env.step(p, phase, ports)


def test_strongest_local_port_mode_uses_only_the_control_aps_own_contribution():
    from simfas.fas import topk_ports
    from simfas.mappo.trainer import active_mask, to_env_action
    from simfas.phy import desired_port_contributions, desired_port_power, desired_port_power_local
    cfg = load_config("configs/smoke.yaml", ["ppo.fas_mode=strongest_local"])
    tr = OnPolicyTrainer(cfg, "mappo", None, "cpu")
    env = tr.env
    o = env.reset()
    for _ in range(3):
        act = active_mask(env, False, tr.fas_learn)
        assert not act.any()
        a = tr.actor.act(tr.obs_norm(o["obs"], update=False), act)
        p, phase, ports = to_env_action(a, env, cfg, False)
        g = desired_port_power_local(env.Hbar, env.Q1, env.Qinter, phase, p, env.F, env.ctrl)
        assert torch.equal(ports, topk_ports(g, env.nk, env.n_max))
        c = desired_port_contributions(env.Hbar, env.Q1, env.Qinter, phase, p, env.F)
        for e in range(env.E):
            for k in range(env.K):
                assert torch.allclose(g[e, k], c[e, env.ctrl[e, k], k].abs() ** 2)
        single = env.F.any(2).sum(1) == 1                      # one serving AP: local == coherent
        assert torch.allclose(g[single], desired_port_power(env.Hbar, env.Q1, env.Qinter, phase, p, env.F)[single],
                              rtol=1e-4)
        o, _ = env.step(p, phase, ports)


def test_head_stop_freeze_keeps_trunk_and_stopped_head_after_their_stop():
    """Phase stops after epoch 1: with head_stop_freeze the trunk and phase head end the update exactly as after a
    1-epoch update, while the power and port heads keep learning."""
    import copy
    ov = ["ppo.per_head_clip=true", "ppo.target_kl=null", "ppo.ppo_epochs=4", "ppo.head_target_kl=[1e9,1e-12,1e9]",
          "ppo.head_stop_freeze=true"]
    tr = OnPolicyTrainer(load_config("configs/smoke.yaml", ov), "mappo", None, "cpu")
    tr.collect()
    actor0, opt0 = copy.deepcopy(tr.actor.state_dict()), copy.deepcopy(tr.opt_actor.state_dict())
    torch.manual_seed(0)
    st = tr.update()
    assert st["ppo_epochs_phase"] == 1 and st["ppo_epochs_fas"] == 4
    after4 = {n: q.detach().clone() for n, q in tr.actor.named_parameters()}
    tr.actor.load_state_dict(actor0)
    tr.opt_actor.load_state_dict(opt0)
    tr.cfg.ppo.ppo_epochs = 1
    torch.manual_seed(0)
    tr.update()
    after1 = {n: q.detach().clone() for n, q in tr.actor.named_parameters()}
    for n in after1:
        if ".body." in n or "phase_" in n:
            assert torch.allclose(after4[n], after1[n], atol=1e-7), n
    assert any(not torch.allclose(after4[n], after1[n]) for n in after1 if "fas_logits" in n)
    assert not torch.allclose(after4["nets.0.power_mean.weight"], after1["nets.0.power_mean.weight"])
    with pytest.raises(ValueError):
        load_config("configs/smoke.yaml", ["ppo.head_stop_freeze=true"])


def test_ppo_sp_variant_runs_centralized_with_port_gain_ports(tmp_path):
    import train
    assert train.main(["--config", "configs/smoke.yaml", "--algo", "ppo_sp", "--seed", "1", "--steps", "600",
                       "--out", str(tmp_path), "--device", "cpu"])
    from simfas.config import load_config as lc
    cfg = lc(str(tmp_path / "ppo_sp" / "seed1" / "config.yaml"))
    assert cfg.ppo.fas_mode == "strongest"
    d = torch.load(tmp_path / "ppo_sp" / "seed1" / "best_model.pt", weights_only=False)
    assert d["algo"] == "ppo"


def _one_ap_controls_all(D, serve=None):
    """Control assignment for tests: AP 0 controls the FAS of every UE, the other APs control none."""
    E, L, K = D.shape
    ctrl = torch.zeros(E, K, dtype=torch.long, device=D.device)
    return ctrl, ctrl[:, None, :] == torch.arange(L, device=D.device).view(1, L, 1)


def _update_stats(ov, active_only):
    cfg = load_config("configs/smoke.yaml", ov + [f"ppo.fas_stats_active_only={str(active_only).lower()}"])
    tr = OnPolicyTrainer(cfg, "mappo", None, "cpu")
    tr.collect()
    torch.manual_seed(0)
    return tr, tr.update()


def test_fas_stats_active_only_averages_the_port_head_over_controlling_agents(monkeypatch):
    """AP 0 controls every UE, AP 1 none (L = 2). The port head of AP 1 has ratio 1 and KL 0, so the plain agent
    mean halves the port KL / entropy; fas_stats_active_only gives AP 0's own values. No port term in the loss
    (joint ratio, entropy_coef_fas 0), so both runs train identically."""
    import simfas.geometry
    monkeypatch.setattr(simfas.geometry, "nearest_ap_control", _one_ap_controls_all)
    ov = ["ppo.target_kl=null", "ppo.entropy_coef_fas=0.0"]
    tr, st0 = _update_stats(ov, False)
    act = tr.buf["active"].flatten(3).any(-1)                  # (T, E, A)
    assert act[..., 0].all() and not act[..., 1].any()
    _, st1 = _update_stats(ov, True)
    assert st1["approx_kl_fas"] > 0 and st1["ent_fas"] > 0
    for k in ("approx_kl_fas", "ent_fas", "clipfrac_fas"):
        assert abs(st1[k] - 2 * st0[k]) <= 1e-6 * max(abs(st1[k]), 1e-12), k
    for k in ("approx_kl", "approx_kl_power", "approx_kl_phase", "ent_power", "pg_loss", "v_loss"):
        assert st1[k] == st0[k], k                               # power / phase / joint terms unchanged
    # head_target_kl uses the same masked port KL. The masked port surrogate is twice the plain one, so a port
    # weight of 0.5 keeps the gradient (and the training) the same. With 1 epoch the stats hold the epoch-1 port
    # KL e1 (2 minibatches); a ports threshold of 1.5 e1 stops the ports head only for the masked mean (~2 e1)
    ph = ["ppo.per_head_clip=true"]
    e1 = _update_stats(ov + ph + ["ppo.ppo_epochs=1"], False)[1]["approx_kl_fas"]
    assert e1 > 0
    stop = ov + ph + ["ppo.ppo_epochs=2", f"ppo.head_target_kl=[1e9,1e9,{1.5 * e1}]"]
    assert _update_stats(stop + ["ppo.head_loss_weights=[1.0,1.0,1.0]"], False)[1]["ppo_epochs_fas"] == 2
    assert _update_stats(stop + ["ppo.head_loss_weights=[1.0,1.0,0.5]"], True)[1]["ppo_epochs_fas"] == 1
    with pytest.raises(ValueError):
        load_config("configs/smoke.yaml", ["ppo.fas_stats_active_only=maybe"])
    assert load_config("configs/smoke.yaml").ppo.fas_stats_active_only is False


def test_every_learning_algo_is_drawn_with_its_own_color():
    """Each train.py algo (MAPPO_VARIANTS, PPO_VARIANTS, the four learners, random) has a METHODS entry with a
    label and a distinct color; rp_ep / rp_ep_sp are not drawn on purpose."""
    import train
    from simfas.baselines import BASELINE_ALGOS
    from simfas.plotting import METHODS
    keys = [k for k, _, _ in METHODS]
    want = {"mappo", "ppo", "ddpg", "td3", *train.MAPPO_VARIANTS, *train.PPO_VARIANTS, *BASELINE_ALGOS}
    want -= {"rp_ep", "rp_ep_sp"}
    assert want <= set(keys) and len(keys) == len(set(keys))
    assert len({c.lower() for _, _, c in METHODS}) == len(METHODS) and all(lb for _, lb, _ in METHODS)


def test_gae_bootstraps_with_the_critic_value_at_a_mid_episode_rollout_boundary(monkeypatch):
    """rollout_steps = 6, T = 4: the rollout ends at step 2 of the second episode (not terminal), so the last GAE
    target uses V(s_next) of the state the next rollout starts from (last_value); inside the rollout the next
    value is the stored V(s_{t+1}) and an episode end at t = 3 stops the bootstrap."""
    import simfas.mappo.trainer as mt
    cfg = load_config("configs/smoke.yaml", ["ppo.rollout_steps=6", "ppo.discount=0.9", "ppo.gae_lambda=0.8"])
    tr = OnPolicyTrainer(cfg, "mappo", None, "cpu")
    tr.collect()
    b, env = tr.buf, tr.env
    assert (env.t == 2).all() and b["terminated"][-1].eq(0).all() and b["terminated"][3].eq(1).all()
    with torch.no_grad():
        v_next = tr.vnorm.denormalize(tr.critic(tr.state_norm(env.global_state(), update=False)))
    assert torch.allclose(tr.last_value, v_next)
    seen = {}
    gae0 = mt.gae
    monkeypatch.setattr(mt, "gae", lambda r, v, nv, term, g, lam: seen.update(nv=nv) or gae0(r, v, nv, term, g, lam))
    adv_ref, _ = gae0(b["reward"].clone(), b["value"].clone(), torch.cat([b["value"][1:], v_next[None]]),
                      b["terminated"].clone(), 0.9, 0.8)
    r, v = b["reward"].clone(), b["value"].clone()
    tr.update()
    assert torch.equal(seen["nv"][-1], tr.last_value) and torch.equal(seen["nv"][:-1], v[1:])
    g, lam = 0.9, 0.8
    d = lambda t, nv: r[t] + g * nv - v[t]
    a5 = d(5, v_next)                                          # boundary: bootstrap with V(s_6)
    a4 = d(4, v[5]) + g * lam * a5
    a3 = r[3] - v[3]                                           # terminal: no bootstrap, recursion cut
    assert torch.allclose(adv_ref[5], a5) and torch.allclose(adv_ref[4], a4) and torch.allclose(adv_ref[3], a3)


def test_port_head_stats_with_agents_that_control_no_ue():
    """Natural control assignment with L = 3 APs, K = 3 UEs: about 30% of the agents control no UE. Their port
    log-prob and entropy are 0 (ratio 1, KL 0). With one minibatch per epoch the plain agent mean is the
    active-agent mean times the fraction of agents with port draws; the training is identical (joint ratio,
    entropy_coef_fas 0)."""
    ov = ["system.L=3", "ppo.minibatch_size=32", "ppo.target_kl=null", "ppo.entropy_coef_fas=0.0"]
    tr, st0 = _update_stats(ov, False)
    b = tr.buf
    has = b["active"].flatten(3).any(-1)                       # (T, E, A)
    frac = has.float().mean().item()
    assert 0 < frac < 1                                        # some agents control no UE
    f = lambda x: x.reshape((-1,) + x.shape[2:])
    with torch.no_grad():
        ev = tr.actor.evaluate(f(b["x"]), f(b["active"]), f(b["z_power"]), f(b["z_phase"]), f(b["draws"]))
    idle = ~f(has)
    assert (ev["logp_parts"][..., 2][idle] == 0).all() and (ev["ent_fas"][idle] == 0).all()
    _, st1 = _update_stats(ov, True)
    for k in ("approx_kl_fas", "ent_fas", "clipfrac_fas"):
        assert abs(st0[k] - frac * st1[k]) <= 1e-5 * max(abs(st0[k]), 1e-12), k
    assert st1["approx_kl_fas"] > st0["approx_kl_fas"] > 0
    assert st1["approx_kl_power"] == st0["approx_kl_power"] and st1["approx_kl"] == st0["approx_kl"]
