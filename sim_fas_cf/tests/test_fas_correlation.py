import numpy as np
import pytest
import torch

from simfas import fas
from simfas.channels import correlation_from_points, psd_sqrt, spherical_j0
from simfas.config import Config


def test_spherical_j0_matches_scipy_and_is_not_cylindrical():
    special = pytest.importorskip("scipy.special")
    x = np.linspace(0.0, 20.0, 201)
    np.testing.assert_allclose(spherical_j0(x), special.spherical_jn(0, x), atol=1e-12)
    assert spherical_j0(0.0) == 1.0
    assert not np.allclose(spherical_j0(x[1:]), special.j0(x[1:]))


@pytest.mark.parametrize("Nx,Ny,W", [(4, 4, 1.0), (3, 5, 2.0), (1, 4, 1.0), (6, 1, 0.5)])
def test_fas_correlation_symmetric_unit_diag_psd(Nx, Ny, W):
    lam = 0.0107
    J = correlation_from_points(fas.port_coordinates(Nx, Ny, W, W, lam), lam)
    assert np.allclose(J, J.T) and np.allclose(np.diag(J), 1.0)
    assert np.linalg.eigvalsh(J).min() > -1e-9
    S = psd_sqrt(J)
    np.testing.assert_allclose(S @ S.conj().T, J, atol=1e-9)


def test_eq1_port_coordinates_and_index_convention():
    lam = 0.01
    q = fas.port_coordinates(4, 3, 1.5, 2.0, lam, centered=False)
    # p = nx + Nx * ny, x = nx/(Nx-1) Wx lambda, y = ny/(Ny-1) Wy lambda
    for p in range(12):
        nx, ny = p % 4, p // 4
        np.testing.assert_allclose(q[p], [nx / 3 * 1.5 * lam, ny / 2 * 2.0 * lam])
    qc = fas.port_coordinates(4, 3, 1.5, 2.0, lam, centered=True)
    d = lambda a: np.linalg.norm(a[:, None] - a[None], axis=-1)
    np.testing.assert_allclose(d(q), d(qc), atol=1e-15)          # centring keeps pairwise distances
    single = fas.port_coordinates(1, 1, 1.0, 1.0, lam, centered=False)
    assert np.all(single == 0.0)                                  # no division by Nx-1 = 0


def test_default_config_correlation_shapes():
    from simfas.channels import correlation_matrices
    cfg = Config()
    J, R = correlation_matrices(cfg)
    assert J.shape == (cfg.system.Np, cfg.system.Np) and np.allclose(R, np.eye(cfg.system.N))   # R = I default
    cfg.system.sim_corr_model = "j0"
    _, R2 = correlation_matrices(cfg)
    assert np.allclose(np.diag(R2), 1.0) and np.linalg.eigvalsh(R2).min() > -1e-9


def test_port_helpers_distinct_and_masked():
    nk = [1, 3, 2]
    sv = fas.slot_valid_mask(nk, 3)
    g = torch.Generator().manual_seed(0)
    q = fas.random_ports(50, nk, 5, 3, g, "cpu")
    fas.validate_ports(q, sv, 5)
    assert (q[:, 0, 1:] == -1).all() and (q[:, 2, 2] == -1).all()
    hot = fas.multi_hot(q, sv, 5)
    assert torch.equal(hot.sum(-1), torch.tensor(nk, dtype=torch.float32).expand(50, 3))
    bad = q.clone()
    bad[0, 1, 1] = bad[0, 1, 0]
    with pytest.raises(ValueError):
        fas.validate_ports(bad, sv, 5)
