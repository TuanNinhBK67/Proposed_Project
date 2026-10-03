import numpy as np
from cf_mmimo_env import overallEnv


class RunningMeanStd:
    # Parallel (Welford-style) running mean/variance in float64.
    # Unlike gymnasium's version it starts from count = 0 (no fake var = 1 prior),
    # because the raw channel entries are ~1e-5 and a var = 1 prior would swamp them.
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.zeros(shape, dtype=np.float64)
        self.count = 0.0

    def update(self, x):  # x: (batch, *shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        m2 = (self.var * self.count + batch_var * batch_count
              + np.square(delta) * self.count * batch_count / total)
        self.mean = self.mean + delta * batch_count / total
        self.var = m2 / total
        self.count = total

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])


class CFMultiAgentEnv:
    """Cooperative multi-agent view of overallEnv: one agent per AP.

    The physics, reward and episode logic are those of overallEnv; this class only
    re-slices the joint action / observation per AP.

      agent l acts    : [power (K) | SIM phases (M*N) | FA ports (U)]   in [-1, 1]
      agent l observes (obs_mode):
        "full_csi"  : [Re,Im H_l (N x U) | Re,Im h_sim_ue[l] (K x N) | phases/2pi (M x N) | port idx/(Np-1) (U)]
                      i.e. the underlying channels of AP l plus its current SIM phases and FA ports
        "effective" : [Re, Im] of the effective channel h_user[l] (K x U), the observation of overallEnv
      global state    : concatenation of all agents' (normalised) observations
      reward          : shared team reward = sum-rate (0 if any FA pair is infeasible)
    """

    def __init__(self, env_kwargs: dict, obs_mode: str = "full_csi",
                 obs_clip: float = 10.0, norm_eps: float = 1e-30):
        assert obs_mode in ("full_csi", "effective"), obs_mode
        self.obs_mode = obs_mode
        self.env = overallEnv(**env_kwargs)
        e = self.env
        self.n_agents = e.ap_nums
        self.n_users = e.user_equipment_nums
        self.n_ant = e.antenna_nums
        self.n_layers = e.layer_nums
        self.n_elem = e.total_element_per_layers

        K, U, M, N = self.n_users, self.n_ant, self.n_layers, self.n_elem
        if obs_mode == "effective":
            self.obs_dim = 2 * K * U
        else:
            self.obs_dim = 2 * N * U + 2 * K * N + M * N + U
        self.state_dim = self.n_agents * self.obs_dim
        self.act_dim = self.n_users + self.n_layers * self.n_elem + self.n_ant
        assert self.n_agents * self.act_dim == e.action_dim

        # Statistics are kept per agent and per feature: the channel scales differ by orders of
        # magnitude between APs (path loss) and between blocks (H_l ~ O(1-10), h_sim_ue ~ 1e-5).
        self.rms = RunningMeanStd((self.n_agents, self.obs_dim))
        self.obs_clip = obs_clip
        self.norm_eps = norm_eps
        self.training = True  # False freezes the normaliser (evaluation)

    def _raw_local_obs(self, obs):
        e = self.env
        if self.obs_mode == "effective":
            # overallEnv obs layout: [Re (L,K,U) | Im (L,K,U)] -> per-agent [Re | Im]
            local = obs.astype(np.float64).reshape(2, self.n_agents, -1).transpose(1, 0, 2)
            return local.reshape(self.n_agents, self.obs_dim)

        n_ports = e.num_available_fa_ports
        rows = []
        for l in range(self.n_agents):
            H = e.H_l[l]                     # (N, U)  FA -> first SIM layer (depends on FA ports)
            H_su = e.h_sim_ue[l]             # (K, N)  last SIM layer -> UEs (fixed within an episode)
            phases = e.phase_shift_matrix[l] / (2.0 * np.pi)                        # (M, N) in [0, 1)
            # FA_position holds port coordinates; recover each antenna's port index
            dist = np.linalg.norm(
                e.FA_position[l].T[:, None, :] - e.available_fa_ports[None, :, :], axis=-1)  # (U, Np)
            ports = dist.argmin(axis=1) / (n_ports - 1)
            rows.append(np.concatenate([
                H.real.ravel(), H.imag.ravel(),
                H_su.real.ravel(), H_su.imag.ravel(),
                phases.ravel(), ports,
            ]))
        return np.stack(rows)                # (L, obs_dim)

    def _process(self, obs):
        local = self._raw_local_obs(obs)
        if self.training:
            self.rms.update(local[None])
        local = (local - self.rms.mean) / np.sqrt(self.rms.var + self.norm_eps)
        local = np.clip(local, -self.obs_clip, self.obs_clip).astype(np.float32)
        return local, local.reshape(-1)  # (L, obs_dim), (L * obs_dim,)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        local, state = self._process(obs)
        return local, state, info

    def step(self, actions):
        # actions: (L, act_dim), one row per agent
        actions = np.asarray(actions, dtype=np.float32)
        K, M, N = self.n_users, self.n_layers, self.n_elem
        power = actions[:, :K]
        phase = actions[:, K:K + M * N]
        fa = actions[:, K + M * N:]
        # overallEnv expects [power (L,K) | phase (L,M,N) | fa (L,U)], each flattened row-major
        joint = np.concatenate([power.reshape(-1), phase.reshape(-1), fa.reshape(-1)])
        obs, reward, terminated, truncated, info = self.env.step(joint)
        local, state = self._process(obs)
        return local, state, float(reward), bool(terminated), bool(truncated), info

    def normalizer_state(self):
        return self.rms.state_dict()

    def load_normalizer_state(self, state):
        self.rms.load_state_dict(state)
