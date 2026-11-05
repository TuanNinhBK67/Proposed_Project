"""Vectorised PHY vs. a transparent NumPy loop reference (complex128) with heterogeneous n_k."""
import numpy as np
import pytest
import torch

from simfas import fas
from simfas.phy import effective_channel, explicit_sinr, mmse_combiner, mmse_sinr, stream_channels
from simfas.reward import evaluate_link


def reference(Hbar, ports, nk, phase, p, F, Q1, Qi, sigma2):
    """Literal Eq.(6)-(9), (15), (18)-(20) with true n_k-dimensional quantities."""
    E, L, K, Np, N = Hbar.shape
    gam = np.zeros((E, K))
    for e in range(E):
        Heff = {}
        for l in range(L):
            B = np.diag(np.exp(1j * phase[e, l, 0]))
            for m in range(1, phase.shape[2]):
                B = np.diag(np.exp(1j * phase[e, l, m])) @ Qi @ B
            for k in range(K):
                A = np.zeros((Np, nk[k]))
                for j in range(nk[k]):
                    A[ports[e, k, j], j] = 1.0
                Heff[l, k] = A.T @ Hbar[e, l, k] @ B @ Q1                     # (n_k, U)
        for k in range(K):
            d = [sum(Heff[l, k] @ (np.sqrt(p[e, l]) * F[e, l, :, i]) for l in range(L)) for i in range(K)]
            C = sigma2 * np.eye(nk[k], dtype=complex)
            for i in range(K):
                if i != k:
                    C += np.outer(d[i], d[i].conj())
            gam[e, k] = np.real(d[k].conj() @ np.linalg.solve(C, d[k]))
    return gam


def random_case(L, U, K, Np, N, M, nk, seed):
    rng = np.random.default_rng(seed)
    E = 3
    Hbar = (rng.standard_normal((E, L, K, Np, N)) + 1j * rng.standard_normal((E, L, K, Np, N))) * 1e-5
    Q1 = (rng.standard_normal((N, U)) + 1j * rng.standard_normal((N, U))) * 0.3
    Qi = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * 0.2
    n_max = max(nk)
    ports = -np.ones((E, K, n_max), dtype=np.int64)
    for e in range(E):
        for k in range(K):
            ports[e, k, :nk[k]] = rng.choice(Np, nk[k], replace=False)
    phase = rng.uniform(0, 2 * np.pi, (E, L, M, N))
    w = rng.uniform(0, 1, (E, L, U + 1))
    p = 0.1 * (w / w.sum(-1, keepdims=True))[..., :U]
    F = np.zeros((E, L, U, K))
    for e in range(E):                                     # random exclusive association with coverage
        owner = rng.integers(0, K, L * U)
        owner[:K] = rng.permutation(K)
        F[e].reshape(L * U, K)[np.arange(L * U), owner] = 1.0
    return Hbar, Q1, Qi, ports, phase, p, F


@pytest.mark.parametrize("L,U,K,Np,N,M,nk", [(1, 1, 1, 2, 4, 1, [1]), (2, 2, 3, 4, 4, 2, [2, 1, 3]),
                                              (3, 2, 4, 6, 9, 3, [3, 3, 1, 2]), (2, 4, 2, 4, 4, 2, [4, 2])])
def test_vectorised_phy_matches_loop_reference(L, U, K, Np, N, M, nk):
    Hbar, Q1, Qi, ports, phase, p, F = random_case(L, U, K, Np, N, M, nk, seed=L + 7 * K)
    sigma2 = 1e-12
    ref = reference(Hbar, ports, nk, phase, p, F, Q1, Qi, sigma2)
    t = lambda x: torch.as_tensor(x, dtype=torch.complex128)
    sv = fas.slot_valid_mask(nk, max(nk))
    Pt, Ft = torch.as_tensor(p), torch.as_tensor(F)
    Heff = effective_channel(t(Hbar), torch.as_tensor(ports), sv, torch.as_tensor(phase), t(Q1), t(Qi))
    assert Heff.shape == (3, L, K, max(nk), U)
    assert (Heff[:, :, ~sv] == 0).all()                    # unused slots are zero rows
    d = stream_channels(Heff, Pt, Ft)
    gamma = mmse_sinr(d, sigma2)
    np.testing.assert_allclose(gamma.numpy(), ref, rtol=1e-8, atol=1e-12)
    # Eq.(17) with the unnormalised Eq.(19) combiner gives the same SINR as Eq.(20)
    w = mmse_combiner(d, sigma2)
    np.testing.assert_allclose(explicit_sinr(d, w, sigma2).numpy(), ref, rtol=1e-7)
    res = evaluate_link(t(Hbar), torch.as_tensor(ports), sv, torch.as_tensor(phase), Pt, Ft, t(Q1), t(Qi),
                        sigma2, torch.full((K,), 0.5, dtype=torch.float64), 2.0)
    R = np.log2(1 + ref)
    np.testing.assert_allclose(res.reward.numpy(), R.sum(-1) - 2.0 * np.maximum(0.5 - R, 0).sum(-1), rtol=1e-8)


def test_coherent_sum_across_aps():
    """d_{k,i} adds complex AP contributions: rotating AP 1's contribution changes |d| (not a power sum)."""
    Heff = torch.zeros(1, 2, 1, 1, 1, dtype=torch.complex128)
    Heff[0, 0, 0, 0, 0], Heff[0, 1, 0, 0, 0] = 1.0, 1.0
    p = torch.ones(1, 2, 1, dtype=torch.float64)
    F = torch.ones(1, 2, 1, 1, dtype=torch.bool)
    assert torch.isclose(stream_channels(Heff, p, F).abs()[0, 0, 0, 0], torch.tensor(2.0, dtype=torch.float64))
    Heff[0, 1, 0, 0, 0] = -1.0
    assert stream_channels(Heff, p, F).abs()[0, 0, 0, 0] < 1e-12


def test_zero_desired_signal_gives_zero_rate_and_finite_values():
    d = torch.zeros(1, 2, 2, 2, dtype=torch.complex128)
    d[0, 0, 1] = torch.tensor([1e-3, 2e-3])                # UE 0 sees only interference
    g = mmse_sinr(d, 1e-12)
    assert g[0, 0] == 0 and torch.isfinite(g).all() and (g >= 0).all()
    assert (mmse_sinr(torch.zeros(1, 2, 2, 2, dtype=torch.complex128), 1e-12) == 0).all()


def test_single_user_is_matched_filter_snr():
    d = torch.tensor([[[[1e-6 + 2e-6j, -3e-6j]]]], dtype=torch.complex128)   # K = 1, n = 2
    g = mmse_sinr(d, 1e-12)
    assert torch.isclose(g[0, 0], (d.abs() ** 2).sum() / 1e-12)
