# Equation-to-code map

Equation numbers follow `demo_test.pdf`; where the revised manuscript renumbered an equation, both are given.
Port indices are 0-based in the code (manuscript port p = code port p-1).

| Manuscript | Meaning | Code | Test |
|---|---|---|---|
| Eq.(1) | FAS candidate-port coordinates | `fas.port_coordinates` | `test_fas_correlation::test_eq1_port_coordinates_and_index_convention` |
| Eq.(2)-(3) | J_k = j0(2 pi d / lambda) (spherical Bessel) | `channels.correlation_from_points`, `channels.spherical_j0` | `test_fas_correlation` |
| A_k, active set Q_k | port selection | `fas.gather_selected` (index select), `fas.selection_matrix` (reference) | `test_phy_mmse` |
| Eq.(4) | Rayleigh-Sommerfeld coefficient | `sim_transfer.rs_coefficient`, `build_sim_matrices` | `test_sim_transfer::test_rs_coefficient_hand_value` |
| Eq.(5) | feed -> first-layer distance | `sim_transfer.feed_layer_distances` | `test_sim_transfer::test_eq5_feed_to_first_layer_distances` |
| Eq.(6) | B_l = Phi_M Q_M ... Q_2 Phi_1 | `sim_transfer.sim_cascade`, `sim_apply` | `test_sim_transfer::test_cascade_matches_explicit_product` |
| Eq.(7) | T_l = B_l Q_{l,1} | `sim_transfer.sim_apply(phase, Qinter, Q1)` | same |
| Eq.(8) | Hbar_{l,k} = sqrt(beta) J^{1/2} G R^{1/2} | `channels.draw_hbar`, `channels.large_scale_gain` | `test_fas_correlation`, `test_env_transitions` |
| Eq.(9) | H_{l,k} = A_k^H Hbar_{l,k} B_l Q_{l,1} | `phy.effective_channel` | `test_phy_mmse` (loop reference) |
| Eq.(10)-(11) | association F_l, constraints | `association.aga`, `check_association` | `test_association` |
| Eq.(12) | x_l = P_l F_l s | `phy.stream_channels` (P_l F_l) | `test_phy_mmse` |
| Eq.(13)-(16) | received signal, d_{k,i} = sum_l H_{l,k} P_l f_{l,i} | `phy.stream_channels` (complex sum over l) | `test_phy_mmse::test_coherent_sum_across_aps` |
| Eq.(17) | post-combining SINR | `phy.explicit_sinr` | `test_phy_mmse` (cross-check) |
| Eq.(18) | C_k | `phy.mmse_sinr` | `test_phy_mmse` |
| Eq.(19) | MMSE combiner (unnormalised) | `phy.mmse_combiner` | `test_phy_mmse` |
| Eq.(20) | gamma_k = d^H C^{-1} d (0 if d_kk = 0) | `phy.mmse_sinr` | `test_phy_mmse::test_zero_desired_signal...` |
| Eq.(21)-(22) | R_k, R_sum | `reward.spectral_efficiency`, `LinkResult.sum_se` | `test_phy_mmse` |
| (P1) Eq.(24) | hard-constrained problem | constraints (24c)-(24g) enforced by the action maps; (24b) soft | `test_action_mapping`, `test_env_transitions` |
| (P2), reward Eq.(33) [revised: (25)] | r = R_sum - lambda sum [R_min - R]^+ | `reward.evaluate_link`, `reward.penalised_reward` | `test_phy_mmse` |
| Eq.(25) [AGA problem] | distance-based association | `association.aga` | `test_association::test_aga_hand_example...` |
| Eq.(26) | d_{k,i} with the AGA association | `phy.stream_channels` | `test_phy_mmse` |
| Eq.(27) [revised: (28)] | actor observation o_{l,t} (layout `config._blocks`; active ports of the controlled or, `env.obs_ports_scope: served`, of the served UEs; `served` needs `obs_ctrl_mask`) | `env.actor_obs`, `config.validate` | `test_env_transitions::test_actor_obs_is_local...`, `::test_actor_obs_is_local_with_the_bench_obs_flags`, `::test_port_block_scope`, `::test_served_port_scope_needs_the_control_mask` |
| Eq.(28)-(29) [revised: (30)] | power map: U+1 softmax with an unused share (`env.power_map: softmax`, code default); per-feed linear p = P_max (a+1)/2, rescaled to P_max only if the AP sum exceeds it (`linear`, the benchmark YAMLs; documented deviation) | `action_maps.power_from_latent` (both branches), `action_maps.physical_from_unit` (DDPG/TD3) | `test_action_mapping::test_softmax_power_is_feasible...`, `::test_linear_power_map_is_per_feed_feasible...`, `::test_linear_power_map_reaches_the_environment` |
| Eq.(30) | phase map 2 pi sigmoid | `action_maps.phase_from_latent` | `test_action_mapping` |
| Plackett-Luce port draws | sequential masked categorical; FAS entropy = sum of conditional entropies along the stored prefix | `mappo/distributions._draw_loop` | `test_actor_logprob`, `::test_fas_entropy_surrogate_vs_exact_sequence_entropy` |
| Eq.(31) | joint decentralised policy | `mappo/actor.ActorGroup.act` | `test_actor_logprob` |
| Eq.(32) | joint latent log-likelihood | `ActorGroup.evaluate`, `gaussian_log_prob`, `fas_evaluate` | `test_actor_logprob::test_sequential_logprob...` |
| Eq.(34) | critic global state (+ t/T only with `env.state_time`, off in the benchmark YAMLs) | `env.global_state` | `test_env_transitions`, `::test_state_time_flag_removes_only_the_step_index` |
| Eq.(35) | discounted objective, delta in [0, 1) | `mappo/trainer.gae` (discount `ppo.discount`), range check in `config.validate` | `test_mappo_update::test_gae_by_hand_with_terminal`, `::test_ppo_and_gae_hyper_parameters_are_range_checked` |
| Eq.(36)-(37) | PPO ratio and clipped surrogate (epsilon > 0); per-head variant `ppo.per_head_clip`; port-head means over the agents with port draws (`ppo.fas_stats_active_only`, default off) | `OnPolicyTrainer.update`, `mappo.trainer.head_means` | `test_mappo_update`, `::test_fas_stats_active_only_averages_the_port_head_over_controlling_agents`, `::test_port_head_stats_with_agents_that_control_no_ue`, `::test_one_sample_tail_minibatch_is_merged_into_the_previous_one` |
| Eq.(38) | GAE (terminal at the episode end, V(s_next) at a mid-episode rollout boundary) | `mappo/trainer.gae`, `OnPolicyTrainer.update` (`next_values` from `last_value`) | `test_mappo_update::test_gae_by_hand_with_terminal`, `::test_gae_bootstraps_with_the_critic_value_at_a_mid_episode_rollout_boundary` |
| Sec. III-E | shared AP actor on AP-local inputs: input size and weights independent of L | `config._blocks` (actor obs), `mappo/actor.build_actor` | `test_actor_logprob::test_shared_actor_obs_dim_and_weights_do_not_depend_on_L` |

## Deviations from the manuscript

The benchmark YAMLs depart from the equations above in eleven recorded places (user decisions or documented
implementation choices; full table in `docs/ASSUMPTIONS.md`, Section 4): (1) Eq. (30) power map -> `power_map:
linear`; (2) Eq. (35) delta = 0; (3) Eq. (28) CSI of the served UEs only (`obs_ue_scope: connected`); (4) Eq. (28)
no xi_{t-1} in the actor obs (`obs_qos: false`); (5) Eq. (28) ports of the served UEs + control mask
(`obs_ports_scope: served`); (6) FAS control by the nearest SERVING AP; (7) Eq. (34) state without D, with
phi/2pi, `raw_snr` CSI, and a flat K x Np actor port head; (8) Eq. (31)-(32) ordered Plackett-Luce likelihood and
prefix-entropy surrogate; (9) one common minibatch-normalised advantage; (10) joint-KL early stop in bench, per-head
PPO only in `_perhead` / `mappo_phc`; (11) ideal CSI, the initial configuration, T = 20, and the paper's Eq. (33)
typo for (25).
