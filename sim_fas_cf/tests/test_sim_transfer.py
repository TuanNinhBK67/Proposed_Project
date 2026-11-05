import math

import numpy as np
import pytest
import torch

from simfas import sim_transfer
from simfas.config import Config


def test_rs_coefficient_hand_value():
    lam = 0.01
    # d = dz = lambda, S = (lambda/2)^2: S dz / d^2 (1/(2 pi d) - j/lambda) e^{j 2 pi} = 1/(8 pi) - j/4
    w = sim_transfer.rs_coefficient(lam, lam, (lam / 2) ** 2, lam)
    assert np.isclose(w, 1 / (8 * np.pi) - 0.25j, rtol=1e-12)


def test_eq5_feed_to_first_layer_distances():
    cfg = Config()
    s = cfg.system
    s.U, s.N = 3, 9
    lam, da = s.wavelength, s.meta_atom_spacing_lambda * s.wavelength
    dz = s.feed_to_first_layer_spacing_lambda * lam
    D = sim_transfer.feed_layer_distances(cfg)
    ns = 3
    for n in range(9):
        nx, ny = n % ns + 1, n // ns + 1                         # 1-based indices of Eq.(5)
        for u in range(1, 4):
            ref = math.sqrt(((nx - (ns + 1) / 2) * da - (u - (s.U + 1) / 2) * lam / 2) ** 2
                            + ((ny - (ns + 1) / 2) * da) ** 2 + dz ** 2)
            assert math.isclose(D[n, u - 1], ref, rel_tol=1e-12)


@pytest.mark.parametrize("M", [1, 2, 3])
def test_cascade_matches_explicit_product(M):
    rng = np.random.default_rng(M)
    N, U = 4, 2
    Qi = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Q1 = rng.standard_normal((N, U)) + 1j * rng.standard_normal((N, U))
    phase = rng.uniform(0, 2 * np.pi, (M, N))
    B_ref = np.diag(np.exp(1j * phase[0]))
    for m in range(1, M):                                        # B = Phi_M Q Phi_{M-1} ... Q Phi_1
        B_ref = np.diag(np.exp(1j * phase[m])) @ Qi @ B_ref
    t = lambda a: torch.as_tensor(a, dtype=torch.complex128)
    B = sim_transfer.sim_cascade(torch.as_tensor(phase), t(Qi)).numpy()
    np.testing.assert_allclose(B, B_ref, rtol=1e-12, atol=1e-12)
    T = sim_transfer.sim_apply(torch.as_tensor(phase), t(Qi), t(Q1)).numpy()
    np.testing.assert_allclose(T, B_ref @ Q1, rtol=1e-12, atol=1e-12)
    # zero phases: B = Q^{M-1};  2 pi periodicity
    B0 = sim_transfer.sim_cascade(torch.zeros(M, N, dtype=torch.float64), t(Qi)).numpy()
    np.testing.assert_allclose(B0, np.linalg.matrix_power(Qi, M - 1), rtol=1e-12, atol=1e-12)
    B2 = sim_transfer.sim_cascade(torch.as_tensor(phase + 2 * np.pi), t(Qi)).numpy()
    np.testing.assert_allclose(B2, B_ref, rtol=1e-9, atol=1e-9)


def test_physical_sim_matrices_shapes_and_symmetry():
    cfg = Config()
    Q1, Qi = sim_transfer.build_sim_matrices(cfg, "cpu", torch.complex128)
    assert Q1.shape == (cfg.system.N, cfg.system.U) and Qi.shape == (cfg.system.N, cfg.system.N)
    assert torch.allclose(Qi, Qi.T)                              # reciprocal propagation between equal layers
    assert torch.isfinite(torch.view_as_real(Q1)).all()
