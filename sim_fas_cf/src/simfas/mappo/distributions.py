"""Policy distributions of the hybrid action.

Continuous heads: diagonal Gaussians over raw latents; log-probabilities are those of the stored latent z
(the environment maps z deterministically, so PPO ratios on z are exact).

FAS head (Plackett-Luce): for every UE k with active slots, n_k sequential draws from ONE logit vector over the
Np ports; ports already drawn are masked. log pi = sum of the draw log-probabilities IN SAMPLE ORDER, so the
ordered draw sequence is stored for PPO. This is the likelihood of the ordered sequence, not the marginal
probability of the unordered port subset. The entropy term is the sum of the conditional entropies along the
realised draw prefix (a tractable surrogate of the sequence entropy). Inactive slots (j >= n_k, or UEs not
controlled by the agent) draw nothing (stored as -1) and contribute zero log-probability and entropy.
"""
import math
import torch
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
NEG_INF = -1e9

def gaussian_log_prob(z, mean, log_std):
    """Sum over the last two dims (n_ap, dim) -> (...)."""
    return (-0.5 * ((z - mean) / log_std.exp()) ** 2 - log_std - HALF_LOG_2PI).sum((-1, -2))

def gaussian_entropy(log_std):
    """log_std (..., n_ap, dim) -> (...) sum of per-dimension entropies."""
    return (0.5 + HALF_LOG_2PI + log_std).sum((-1, -2))

def _draw_loop(logits, active, draws=None, deterministic=False, generator=None):
    """logits (B, K, Np), active (B, K, n) bool -> draws (B, K, n) long (-1 inactive), log-prob (B,), entropy (B,)."""
    B, K, Np = logits.shape
    n = active.shape[-1]
    chosen = torch.zeros(B, K, Np, dtype=torch.bool, device=logits.device)
    logp = torch.zeros(B, K, dtype=logits.dtype, device=logits.device)
    ent = torch.zeros(B, K, dtype=logits.dtype, device=logits.device)
    out = torch.full((B, K, n), -1, dtype=torch.long, device=logits.device)
    for j in range(n):
        act = active[..., j]
        lg = logits.masked_fill(chosen, NEG_INF)
        logpj = torch.log_softmax(lg, dim=-1)
        pj = logpj.exp()
        if draws is None:
            if deterministic:
                a = lg.argmax(-1)
            else:
                a = torch.multinomial(pj.reshape(-1, Np), 1, generator=generator).view(B, K)
        else:
            a = draws[..., j].clamp_min(0)
        lp = torch.gather(logpj, -1, a[..., None])[..., 0]
        h = -(torch.where(chosen, torch.zeros_like(pj), pj * logpj)).sum(-1)
        logp = logp + torch.where(act, lp, torch.zeros_like(lp))
        ent = ent + torch.where(act, h, torch.zeros_like(h))
        out[..., j] = torch.where(act, a, torch.full_like(a, -1))
        chosen = chosen | (torch.nn.functional.one_hot(a, Np).bool() & act[..., None])
    return out, logp.sum(-1), ent.sum(-1)

def fas_sample(logits, active, deterministic=False, generator=None):
    return _draw_loop(logits, active, None, deterministic, generator)

def fas_evaluate(logits, active, draws):
    _, logp, ent = _draw_loop(logits, active, draws)
    return logp, ent
