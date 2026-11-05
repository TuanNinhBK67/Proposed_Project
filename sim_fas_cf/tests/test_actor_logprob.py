import itertools
import math

import pytest
import torch

from simfas.config import load_config
from simfas.env import SimFasEnv
from simfas.mappo.actor import build_actor
from simfas.mappo.distributions import fas_evaluate, fas_sample
from simfas.mappo.trainer import active_mask


def test_sequential_logprob_matches_hand_formula_np3_nk2():
    logits = torch.tensor([[[0.3, -1.2, 0.8]]])
    active = torch.ones(1, 1, 2, dtype=torch.bool)
    draws = torch.tensor([[[2, 0]]])
    lp, _ = fas_evaluate(logits, active, draws)
    e = logits[0, 0].exp()
    hand = math.log(e[2] / e.sum()) + math.log(e[0] / (e[0] + e[1]))
    assert math.isclose(lp.item(), hand, rel_tol=1e-6)
    lp_rev, _ = fas_evaluate(logits, active, torch.tensor([[[0, 2]]]))
    assert not math.isclose(lp.item(), lp_rev.item())        # ordered-sequence likelihood: order matters


def test_ordered_draw_probabilities_sum_to_one():
    logits = torch.randn(1, 1, 4)
    active = torch.ones(1, 1, 3, dtype=torch.bool)
    total = sum(fas_evaluate(logits, active, torch.tensor([[list(seq)]]))[0].exp().item()
                for seq in itertools.permutations(range(4), 3))
    assert math.isclose(total, 1.0, rel_tol=1e-5)


def test_samples_distinct_and_inactive_slots_contribute_nothing():
    logits = torch.randn(256, 3, 5)
    active = torch.tensor([[True, True, False], [True, False, False], [False, False, False]]).expand(256, 3, 3)
    draws, lp, ent = fas_sample(logits, active)
    assert (draws[:, 0, 2] == -1).all() and (draws[:, 1, 1:] == -1).all() and (draws[:, 2] == -1).all()
    assert (draws[:, 0, 0] != draws[:, 0, 1]).all()
    none = torch.zeros(4, 3, 3, dtype=torch.bool)            # an agent controlling no UE
    _, lp0, ent0 = fas_sample(torch.randn(4, 3, 5), none)
    assert (lp0 == 0).all() and (ent0 == 0).all()


LEAN = ["env.channel_encoding=raw_snr", "env.phase_encoding=raw", "env.obs_ctrl_mask=false", "env.obs_qos=false",
        "env.state_distance=false"]


@pytest.mark.parametrize("lean", [False, True])
@pytest.mark.parametrize("head", ["per_ue", "flat"])
@pytest.mark.parametrize("algo,shared", [("mappo", True), ("mappo", False), ("ppo", True)])
def test_stored_actions_rescore_to_ratio_one(algo, shared, head, lean):
    cfg = load_config("configs/smoke.yaml", [f"ppo.shared_actor={shared}", f"ppo.fas_head={head}"] + (LEAN if lean else []))
    env = SimFasEnv(cfg, 5, 1)
    o = env.reset()
    central = algo == "ppo"
    actor = build_actor(cfg, algo, env.state_dim if central else env.obs_dim)
    x = env.global_state()[:, None] if central else o["obs"]
    act = active_mask(env, central, True)
    a = actor.act(x, act)
    ev = actor.evaluate(x, act, a["z_power"], a["z_phase"], a["draws"])
    assert torch.allclose(ev["logp"], a["logp"], atol=1e-5)          # PPO log-ratio 0, ratio 1
    d1 = actor.act(x, act, deterministic=True)
    d2 = actor.act(x, act, deterministic=True)
    assert torch.equal(d1["draws"], d2["draws"]) and torch.equal(d1["z_power"], d2["z_power"])
    (ev["logp"].sum() - ev["ent_fas"].sum()).backward()
    for name, p in actor.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_fas_entropy_surrogate_vs_exact_sequence_entropy():
    """Np = 4, n_k = 3. The surrogate (sum of the conditional entropies along the realised prefix) is not the
    sequence entropy for one draw, but by the chain rule its expectation over the ordered draws is exactly
    H(q_1, q_2, q_3) = -sum_seq P(seq) log P(seq)."""
    logits = torch.tensor([[[1.2, -0.4, 0.3, -1.5]]], dtype=torch.float64)
    active = torch.ones(1, 1, 3, dtype=torch.bool)
    seqs = list(itertools.permutations(range(4), 3))
    lp, ent = zip(*(fas_evaluate(logits, active, torch.tensor([[list(q)]])) for q in seqs))
    P = torch.cat(lp).exp()
    exact = -(P * P.log()).sum()
    surrogate = torch.cat(ent)
    assert math.isclose(P.sum().item(), 1.0, rel_tol=1e-12)
    assert math.isclose((P * surrogate).sum().item(), exact.item(), rel_tol=1e-10)
    assert (surrogate - exact).abs().max() > 1e-3                # a single realisation is only a surrogate
    # n_k = 1: the surrogate is the categorical entropy itself
    _, e1 = fas_evaluate(logits, active[..., :1], torch.tensor([[[2]]]))
    p = logits.softmax(-1)
    assert math.isclose(e1.item(), -(p * p.log()).sum().item(), rel_tol=1e-12)


@pytest.mark.parametrize("bench", [False, True])
def test_shared_actor_obs_dim_and_weights_do_not_depend_on_L(bench):
    """Sec. III-E: the shared AP actor processes AP-local inputs only, so its input size is the same for L = 3 and
    L = 6 and a state_dict trained at one L loads (strict) and acts at the other."""
    from simfas.config import dims
    flags = []
    if bench:
        import yaml
        e = yaml.safe_load(open("configs/bench_L6U2K4.yaml"))["env"]
        flags = [f"env.{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in e.items()
                 if k not in ("episode_steps", "qos_penalty", "complex_dtype")] + ["ppo.fas_head=flat"]
    cfgs = {L: load_config("configs/smoke.yaml", flags + [f"system.L={L}"]) for L in (3, 6)}
    assert dims(cfgs[3])["obs"] == dims(cfgs[6])["obs"]
    assert dims(cfgs[3])["state"] != dims(cfgs[6])["state"]     # the critic state does grow with L
    a3 = build_actor(cfgs[3], "mappo", dims(cfgs[3])["obs"])
    a6 = build_actor(cfgs[6], "mappo", dims(cfgs[6])["obs"])
    a6.load_state_dict(a3.state_dict())
    env = SimFasEnv(cfgs[6], 2, 0)
    o = env.reset()["obs"]
    out = a6.act(o, active_mask(env, False, True), deterministic=True)
    assert out["z_power"].shape[:2] == (2, 6) and torch.isfinite(out["logp"]).all()
    ref = a3.act(o, active_mask(env, False, True), deterministic=True)   # same weights -> same actions at L = 6
    assert torch.equal(out["draws"], ref["draws"]) and torch.allclose(out["z_phase"], ref["z_phase"])
