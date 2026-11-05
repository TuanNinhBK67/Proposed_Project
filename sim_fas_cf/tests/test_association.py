import pytest
import torch

from simfas.association import aga, check_association
from simfas.config import config_from_dict
from simfas.geometry import nearest_ap_control


@pytest.mark.parametrize("L,U,K", [(3, 4, 4), (2, 2, 4), (1, 5, 5), (4, 1, 3), (3, 2, 5)])
def test_aga_constraints_random_geometries(L, U, K):
    g = torch.Generator().manual_seed(L * 100 + U * 10 + K)
    D = torch.rand(64, L, U, K, generator=g, dtype=torch.float64) * 100
    F = aga(D)
    check_association(F)
    assert F.shape == (64, L, U, K) and F.dtype == torch.bool


def test_aga_rejects_too_few_feeds():
    with pytest.raises(ValueError):
        aga(torch.rand(1, 1, 2, 3))
    with pytest.raises(ValueError):
        config_from_dict({"system": {"L": 1, "U": 2, "K": 3, "nk": 1, "min_se": 0.5}})


def test_aga_hand_example_stage1_greedy_then_stage2_ranked():
    # one AP with 3 feeds, 2 UEs. Distances (u, k):
    #   feed0: 1, 2   feed1: 3, 1.5   feed2: 0.5, 4
    D = torch.tensor([[[[1.0, 2.0], [3.0, 1.5], [0.5, 4.0]]]], dtype=torch.float64)
    F = aga(D)[0, 0].int().tolist()
    # stage 1: (feed2, UE0) at 0.5, then UE1 is the only unserved UE: closest free feed is feed1 (1.5)
    # stage 2: AP ranking by mean feed distance UE0 (1.5) < UE1 (2.5): the single free feed0 -> UE0
    assert F == [[1, 0], [0, 1], [1, 0]]


def test_aga_stage2_spreads_free_feeds_over_ranked_ues():
    # one AP with 5 feeds, 2 UEs; UE0 is closer to every feed
    D = torch.tensor([1.0, 2.0], dtype=torch.float64).expand(1, 1, 5, 2).clone()
    F = aga(D)[0, 0].int().tolist()
    # stage 1: feed0 -> UE0, feed1 -> UE1; stage 2: free feeds 2, 3, 4 -> UE0, UE1, UE0 (cyclic over the ranking),
    # not all to the nearest UE0
    assert F == [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]


def test_aga_tie_break_lowest_l_u_k():
    D = torch.ones(1, 2, 1, 2, dtype=torch.float64)            # all distances equal
    F = aga(D)[0].int()
    assert F[0, 0].tolist() == [1, 0] and F[1, 0].tolist() == [0, 1]


def test_control_delegation_independent_of_association():
    # AP0 is the nearest AP of UE0 (control), but AGA may give UE0's stream to AP1's feed: both coexist
    Dap = torch.tensor([[[1.0], [2.0]]], dtype=torch.float64)  # (E=1, L=2, K=1)
    ctrl, mask = nearest_ap_control(Dap)
    assert ctrl.tolist() == [[0]] and mask[0, :, 0].tolist() == [True, False]
    Dfeed = torch.tensor([[[[5.0]], [[0.5]]]], dtype=torch.float64)   # (E, L=2, U=1, K=1)
    F = aga(Dfeed)
    assert F[0, 1, 0, 0] and F[0, 0, 0, 0]                    # stage 2 gives the free feed to the only UE


def test_control_tie_goes_to_less_loaded_ap():
    # UE1 is equally close to AP0 and AP1; AP0 already controls UE0, UE2, UE3, UE4, AP1 controls UE5 -> AP1 wins
    D = torch.tensor([[[1.0, 2.0, 1.0, 1.0, 1.0, 9.0],
                       [9.0, 2.0, 9.0, 9.0, 9.0, 1.0]]], dtype=torch.float64)
    ctrl, mask = nearest_ap_control(D)
    assert ctrl.tolist() == [[0, 1, 0, 0, 0, 1]]
    assert mask[0].sum(-1).tolist() == [4, 2]


def test_control_restricted_to_serving_aps():
    # AP0 is nearer to UE0 but has no feed for it -> AP1 (serving) controls UE0
    D = torch.tensor([[[1.0, 1.0], [2.0, 3.0]]], dtype=torch.float64)   # (E=1, L=2, K=2)
    serve = torch.tensor([[[False, True], [True, True]]])
    ctrl, _ = nearest_ap_control(D, serve)
    assert ctrl.tolist() == [[1, 0]]


def test_control_ties_update_load_sequentially():
    # both UEs tie between AP0 and AP1: UE0 -> AP0 (equal load, lower index), then UE1 -> AP1 (AP0 now loaded)
    ctrl, _ = nearest_ap_control(torch.ones(1, 2, 2, dtype=torch.float64))
    assert ctrl.tolist() == [[0, 1]]
