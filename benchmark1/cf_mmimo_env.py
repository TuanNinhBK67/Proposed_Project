import numpy as np
import math 
import gymnasium as gym 
from typing import Optional
from utils import (
    generate_ap_locations, 
    generate_ue_locations,
    generate_path_angles,
    generate_path_angles_for_all_AP,
    generate_path_response_matrix
) 

class overallEnv(gym.Env):
    #Class that simulates the downlink of a system with FA-SIM in cell-free massive MIMO system
    def __init__(self,
                 ap_nums: int = 2,
                 antenna_nums : int = 3,
                 user_equipment_nums : int = 3, 
                 layer_nums : int = 2, 
                 num_elements_side : int = 3, #--> total_elements_side
                 FA_region_size_factor: float = 2.0, #A = FA_region_size_factor * self.light_lambda
                 num_possible_position_horizontal: int = 4,
                 num_possible_position_vertical: int = 5,
                 frequency_hz : float = 28e9,
                 bandwidth_hz : float = 10e6,
                 area_size : float = 200.0,
                 noise_psd_dbm_hz: float = -174.0,
                 noise_figure_db: float = 0.0,
                 ap_transmit_power_dBm: float = 10,
                 ap_height_m: float = 15.0,
                 ue_height_m: float = 1.65,
                 path_loss_exp: float = 3.5,
                 ref_distance_m: float = 1.0, 
                 transmit_path_num: int = 4, 
                 receive_path_num: int = 4, 
                 rician_factor_db: float = 10.0, #?
                 max_episode_steps: int = 10):
        self.ap_nums = ap_nums
        self.antenna_nums = antenna_nums
        self.user_equipment_nums = user_equipment_nums
        self.layer_nums = layer_nums
        self.num_elements_side = num_elements_side
        self.total_element_per_layers = self.num_elements_side ** 2
        self.num_possible_position_horizontal = num_possible_position_horizontal
        self.num_possible_position_vertical = num_possible_position_vertical
        self.frequency_hz = frequency_hz
        self.bandwidth_hz = bandwidth_hz
        self.area_size = area_size
        self.light_lambda = 3e8 / self.frequency_hz
        self.D_t = 5 * self.light_lambda #the thickness of SIM 
        self.atom_dis = self.light_lambda / 2
        self.layer_dis = self.D_t / self.layer_nums
        
        self.noise_psd_dbm_hz = noise_psd_dbm_hz
        self.noise_figure_db = noise_figure_db
        # Total noise power over bandwidth
        self.noise_power_dbm = (self.noise_psd_dbm_hz + 10.0 * np.log10(self.bandwidth_hz) + self.noise_figure_db) #?
        self.noise_power_watts = (10.0 ** (self.noise_power_dbm / 10.0) * 1e-3)
        self.ap_transmit_power_dBm = ap_transmit_power_dBm
        self.ap_transmit_power_watts = 10 ** (self.ap_transmit_power_dBm / 10.0) * 1e-3
        
        # Equal maximum-power allocation
        self.allocated_powers = np.full((self.ap_nums, self.user_equipment_nums),self.ap_transmit_power_watts/ self.user_equipment_nums, dtype=np.float64)
        self.consumed_power_per_ap = np.sum(self.allocated_powers, axis=1)
        self.consumed_power = np.sum(self.consumed_power_per_ap)
        
        self.ap_height_m = ap_height_m
        self.ue_height_m = ue_height_m
        self.path_loss_exp = path_loss_exp
        self.ref_distance_m = ref_distance_m
        self.transmit_path_num = transmit_path_num
        self.receive_path_num = receive_path_num
        self.C_0 = -35 #(self.light_lambda / (4 * math.pi * self.ref_distance_m)) ** 2
        self.ap_sim_dis = self.D_t / self.layer_nums # 5*lambda / M 
        self.rician_factor_db = rician_factor_db #?
        self.rician_factor = 10 ** (self.rician_factor_db / 10.0)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        self.reward = 0
        self.h_user = None 
        self.phase_shift_matrix = None
        self.B_l = None
        self.consumed_power = 0 
        self.user_data_rates = None
        self.num_paths = self.transmit_path_num #self.receive_path_num
        self.fa_feasible = True 
        self.fa_violation_count = 0
        self.fa_min_pair_distance = None
        self.user_sinr = None
        
        #Observation space :
        self.observation_space = gym.spaces.Box(
            low = -np.inf, high = np.inf,
            shape = ((2 * self.ap_nums * self.user_equipment_nums * self.antenna_nums),),
            dtype = np.float32
        )
        
        #Action space : 
        self.power_action_dim = 0
        self.phase_action_dim = (self.ap_nums * self.layer_nums * self.total_element_per_layers) # SIM phase shifts phi_{l,m,n}
        self.fa_action_dim = (self.ap_nums * self.antenna_nums) # FA positions (x_{l,u}, y_{l,u})
        self.action_dim = (self.phase_action_dim + self.fa_action_dim)
        
        self.phase_action_start = 0
        self.phase_action_end = (self.phase_action_dim)

        self.fa_action_start = self.phase_action_end
        self.fa_action_end = (
            self.fa_action_start
            + self.fa_action_dim
        )
        assert self.fa_action_end == self.action_dim
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        
        self.np_random = None
        self.ap_location = generate_ap_locations(self.ap_nums, self.area_size)
        self.ue_location = generate_ue_locations(self.area_size, self.user_equipment_nums)
        self.element_position_matrix = self.calculate_element_position_matrix()
        
        # Movement region for FA
        self.FA_region_size_factor = FA_region_size_factor 
        self.FA_region_size = self.FA_region_size_factor * self.light_lambda # (x, y) in (-FA_region_size / 2, FA_region_size)
        self.FA_min_distance = 0.5 * self.light_lambda #Minimum Distance between pair of FA (D)
        self.available_fa_ports = self.generate_available_fa_ports()
        self.num_available_fa_ports = self.available_fa_ports.shape[0]
        
        # Deterministic, well-separated initial port assignment.
        base_fa_position = self.initialize_fa_positions() #(L, U, 2)
        
        self.initial_FA_position = np.repeat(
            base_fa_position[None, :, :],
            self.ap_nums,
            axis=0
        ) #(L, 2, U)
        self.FA_position = self.initial_FA_position.copy() 
        
        #Fading channels - initialized in reset()
        self.g_in_user = None 

        self.path_loss_beta = self._calculate_pathloss()      
        
        (self.tx_elevation, #L x P_t
         self.tx_azimuth,
         self.rx_elevation,
         self.rx_azimuth) = generate_path_angles_for_all_AP(num_aps=self.ap_nums,
                                                            num_tx_paths=self.transmit_path_num,
                                                            num_rx_paths=self.receive_path_num)

        #L x P_r x P_t  
        self.path_response_matrix = np.zeros((self.ap_nums, self.receive_path_num, self.transmit_path_num), dtype=np.complex128)
        prm_rng = np.random.default_rng()
        for l in range(self.ap_nums):
            self.path_response_matrix[l] = (
                generate_path_response_matrix(
                    num_paths=self.transmit_path_num,
                    distance=self.ap_sim_dis,
                    pathloss_ref_db=self.C_0,
                    reference_distance=self.ref_distance_m,
                    pathloss_exponent=self.path_loss_exp,
                    rician_factor=self.rician_factor,
                    rng=prm_rng
                    )
                )
        self.H_l = self.calculate_path_between_ap_sim(self.FA_position, self.element_position_matrix)
        self.w_t, self.corr_t = self._calculate_transmission_matrix()
        self.h_sim_ue = self._generate_sim_user_channel()
    
    def generate_available_fa_ports(self):
        A = self.FA_region_size
        Nx = self.num_possible_position_horizontal
        Ny = self.num_possible_position_vertical

        if Nx == 1:
            x_coordinates = np.array([0.0], dtype=np.float64)
        else:
            x_coordinates = np.linspace(-A / 2.0, A / 2.0, Nx)

        if Ny == 1:
            y_coordinates = np.array([0.0], dtype=np.float64)
        else:
            y_coordinates = np.linspace(-A / 2.0, A / 2.0, Ny)

        x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates, indexing="xy")
        ports = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
        return ports.astype(np.float64)
    
    def initialize_fa_positions(self):
        U = self.antenna_nums
        ports = self.available_fa_ports
        num_ports = ports.shape[0]

        # Choose approximately evenly distributed initial ports
        selected_indices = np.linspace(0, num_ports - 1, U, dtype=int)
        selected_positions = ports[selected_indices]

        # selected_positions: (U, 2)
        # Need return shape: (2, U)
        return selected_positions.T.copy()  
    
    def check_fa_feasibility(self, fa_positions: np.ndarray):
        L = self.ap_nums
        U = self.antenna_nums
        violation_count = 0
        minimum_distance = np.inf

        for l in range(L):
            for u in range(U):
                for u_prime in range(u + 1, U):
                    distance = np.linalg.norm(fa_positions[l, :, u] - fa_positions[l, :, u_prime])
                    minimum_distance = min(minimum_distance, distance)
                    if distance < self.FA_min_distance:
                        violation_count += 1
        feasible = (violation_count == 0)
        return (feasible, violation_count, minimum_distance)
    
    def calculate_element_position_matrix(self):
        element_position_matrix = np.zeros((self.total_element_per_layers, 2), dtype = np.float64)
        for mm1_idx in range(self.total_element_per_layers):
            m_row = np.floor(mm1_idx / self.num_elements_side)
            m_col = mm1_idx % self.num_elements_side
            m_x_centered = (m_col - (self.num_elements_side - 1) / 2) * self.atom_dis
            m_y_centered = (m_row - (self.num_elements_side - 1) / 2) * self.atom_dis
            element_position_matrix[mm1_idx, :] = [
                m_x_centered,
                m_y_centered
            ]
        return element_position_matrix

    def calculate_path_between_ap_sim(self, FA_position_matrix: np.ndarray, element_position_matrix: np.ndarray):
        L = self.ap_nums
        U = self.antenna_nums
        N = self.total_element_per_layers
        Pt = self.transmit_path_num
        Pr = self.receive_path_num

        # Final AP -> SIM channels
        H = np.zeros((L, N, U), dtype=np.complex128)
        for l in range(L):
            # 1. Transmit FRM G_l
            # FA_position_matrix[l]: (2, U)
            x_fa = FA_position_matrix[l, 0, :]
            y_fa = FA_position_matrix[l, 1, :]
            tx_x = (np.sin(self.tx_elevation[l]) * np.cos(self.tx_azimuth[l]))
            tx_y = np.cos(self.tx_elevation[l])
            rho_t = (tx_x[:, None] * x_fa[None, :] + tx_y[:, None] * y_fa[None, :]) # rho_t: (Pt, U)
            G_l = np.exp(1j* (2.0 * np.pi / self.light_lambda)* rho_t) # G_l: (Pt, U)

            # 2. Receive FRM F_l
            rx_x = (np.sin(self.rx_elevation[l])* np.cos(self.rx_azimuth[l]))
            x_element = element_position_matrix[:, 0] # SIM element coordinates
            rho_r = (rx_x[:, None] * x_element[None, :]) # rho_r: (Pr, N)
            F_l = np.exp(1j * (2.0 * np.pi / self.light_lambda) * rho_r) # F_l: (Pr, N)

            # 3. Path-response matrix Sigma_l
            Sigma_l = self.path_response_matrix[l] # (P_r, P_t)

            # 4. AP to first SIM layer channel
            # H_l = F_l^H Sigma_l G_l
            H[l] = (F_l.conj().T @ Sigma_l @ G_l) # (N, U)
        return H #(L, N, U)

    def _calculate_transmission_matrix(self):
        w_t = np.zeros((self.total_element_per_layers, self.total_element_per_layers), dtype = complex)
        corr_t = np.zeros((self.total_element_per_layers, self.total_element_per_layers), dtype = complex)
            
        for mm1_idx in range(self.total_element_per_layers):
            m_row = np.floor(mm1_idx / self.num_elements_side)
            m_col = mm1_idx % self.num_elements_side
            m_x_centered = (m_col - (self.num_elements_side - 1) / 2) * self.atom_dis
            m_y_centered = (m_row - (self.num_elements_side - 1) / 2) * self.atom_dis
                
            for mm2_idx in range(self.total_element_per_layers):
                n_row = np.floor(mm2_idx / self.num_elements_side)
                n_col = mm2_idx % self.num_elements_side
                n_x_centered = (n_col - (self.num_elements_side - 1) / 2) * self.atom_dis
                n_y_centered = (n_row - (self.num_elements_side - 1) / 2) * self.atom_dis
                    
                d_temp = np.sqrt((m_x_centered - n_x_centered) ** 2 + (m_y_centered - n_y_centered) ** 2)
                d_temp2 = np.sqrt(self.layer_dis ** 2 + d_temp ** 2)
                    
                # w_t[mm2_idx, mm1_idx] = (self.light_lambda ** 2 / (4 * np.pi)) * (self.layer_dis / (d_temp2 ** 2)) * \
                #                         (1 / (d_temp2) - 1j * 2 * np.pi / self.light_lambda) *\
                #                         np.exp(1j * 2 * np.pi * d_temp2 / self.light_lambda)
                w_t[mm2_idx, mm1_idx] = ((self.atom_dis ** 2)* self.layer_dis/ (d_temp2 ** 2) * (1.0 / (2.0 * np.pi * d_temp2)- 1j / self.light_lambda)* np.exp(1j * 2.0 * np.pi * d_temp2 / self.light_lambda))
                corr_t[mm2_idx, mm1_idx] = np.sinc(2 * d_temp / self.light_lambda)
                
        return w_t, corr_t
    
    def calculate_sim_transfer_matrix(self, phase_shift_matrix: np.ndarray): 
        #Calculate the SIM transfer matrix B_l for all APs
        L = self.ap_nums
        M = self.layer_nums
        N = self.total_element_per_layers
        # expected_shape = (L, M, N)
        B = np.zeros((L, N, N), dtype=np.complex128)
        # At the moment all APs/layers use the same
        # inter-layer propagation matrix because
        # their SIM geometries are identical.
        Q = self.w_t
        
        for l in range(L):
            # First metasurface layer
            phi_1 = np.exp(1j * phase_shift_matrix[l, 0, :])
            B_l = np.diag(phi_1)

            # Remaining layers
            for m in range(1, M):
                phi_m = np.exp(1j * phase_shift_matrix[l, m, :])
                Phi_m = np.diag(phi_m) #(N, N)
                B_l = (Phi_m @ Q @ B_l)
            B[l] = B_l
        return B #(L, N, N)

    #Path loss from SIM to UE  
    def _calculate_pathloss(self):
        pathloss_beta = np.zeros((self.ap_nums, self.user_equipment_nums))
        # heigh_difference = self.ap_height_m - self.ue_height_m
        # Consider the location of (AP-UE) == (SIM=UE)
        dx = (self.ap_location[:, 0][:, None] - self.ue_location[:, 0][None, :])
        dy = (self.ap_location[:, 1][:, None] - self.ue_location[:, 1][None, :])
        dz = self.ap_height_m - self.ue_height_m
        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        pathloss_beta = (10 ** (self.C_0 / 10)) * ((distance / self.ref_distance_m) ** (-self.path_loss_exp))
        return pathloss_beta #(L, K)

    def _generate_sim_user_channel(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        L = self.ap_nums
        K = self.user_equipment_nums
        N = self.total_element_per_layers

        # Matrix square root R^(1/2)
        eigenvalues, eigenvectors = np.linalg.eigh(self.corr_t)
        eigenvalues = np.clip(eigenvalues,0.0,None)
        corr_sqrt = (eigenvectors@ np.diag(np.sqrt(eigenvalues))@ eigenvectors.conj().T)
        # Channel:
        # h_SIM[l,k] ~ CN(0, beta[l,k] R)
        h_sim_ue = np.zeros((L, K, N),dtype=np.complex128)
        for l in range(L):
            for k in range(K):
                z = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2.0) # z ~ CN(0, I_N)
                correlated_fading = (corr_sqrt @ z) # Spatially correlated fading
                h_sim_ue[l, k, :] = (np.sqrt(self.path_loss_beta[l, k])* correlated_fading) # Apply large-scale channel gain

        return h_sim_ue #(L, K, N)
    
    def get_channel(self):
        L = self.ap_nums
        K = self.user_equipment_nums
        U = self.antenna_nums
        N = self.total_element_per_layers
        # Equivalent channel
        overall_channel = np.zeros((L, K, U), dtype=np.complex128)
        
        for l in range(L):
            BH = (self.B_l[l]@ self.H_l[l])
            for k in range(K):
                overall_channel[l, k, :] = (self.h_sim_ue[l, k, :].conj()@ BH)
        return overall_channel

    def _calculate_rates(self):
        L = self.ap_nums
        K = self.user_equipment_nums
        U = self.antenna_nums

        sinr = np.zeros(K, dtype=np.float64)
        rates = np.zeros(K, dtype=np.float64)

        for k in range(K):
            desired_amplitude = np.sum(self.h_user[:, k, k] * np.sqrt(self.allocated_powers[:, k]))
            desired_power = (np.abs(desired_amplitude) ** 2)

            interference_power = 0.0
            for j in range(K):
                if j == k:
                    continue
                interference_amplitude = np.sum(self.h_user[:, k, j]* np.sqrt(self.allocated_powers[:, j]))
                interference_power += (np.abs(interference_amplitude) ** 2)

            denominator = (interference_power + self.noise_power_watts) #nosie + interference
            sinr[k] = (desired_power/ denominator)
            rates[k] = np.log2(1.0 + sinr[k])
        sum_rate = np.sum(rates)

        return (sinr, rates, sum_rate)

    def _get_info(self):
        return {
            "current_reward": self.reward,
            "current_step": self.current_step,
            "user_sinr": self.user_sinr,
            "user_data_rates": self.user_data_rates,
            "sum_rate":(np.sum(self.user_data_rates) if self.user_data_rates is not None else None),
            "consumed_power_watts": self.consumed_power,
            "fa_feasible": self.fa_feasible,
            "fa_violation_count": self.fa_violation_count,
            "fa_min_pair_distance": self.fa_min_pair_distance,
        }  
            
    def _get_obs(self, h_user: np.ndarray):
        obs = np.concatenate([
            h_user.real.flatten(),
            h_user.imag.flatten()
        ])
        return obs.astype(np.float32)

    def _decode_action(self, action: np.ndarray):
        L = self.ap_nums
        K = self.user_equipment_nums
        M = self.layer_nums
        N = self.total_element_per_layers
        U = self.antenna_nums
        action = np.asarray(action, dtype=np.float32) # Basic check
        action = np.clip(action, -1.0, 1.0) # Safety clipping
        
        # power action
        # power_action = action[self.power_action_start:self.power_action_end]
        # power_action = power_action.reshape(L,K)
        # power_fraction = (power_action + 1.0) / 2.0 # Map [-1, 1] -> [0, 1]
        # allocated_powers = np.zeros((L, K), dtype=np.float64)

        # for l in range(L):
        #     power_l = power_fraction[l].copy()
        #     total_fraction = np.sum(power_l)

        #     if total_fraction > 1.0:
        #         power_l = (power_l /total_fraction)

        #     allocated_powers[l] = (self.ap_transmit_power_watts * power_l)

        #Sim phase action
        phase_action = action[self.phase_action_start:self.phase_action_end]
        phase_action = phase_action.reshape(L, M, N)
        phase_shift_matrix = (np.pi * (phase_action + 1.0)) # Map [-1, 1] -> [0, 2*pi)
        phase_shift_matrix = np.mod(phase_shift_matrix, 2.0 * np.pi)
        
        #Fa position action
        fa_action = action[self.fa_action_start:self.fa_action_end]
        fa_action = fa_action.reshape(L, U)
        num_ports = self.num_available_fa_ports
                
        #Map [-1, 1] --> [0, num_ports - 1]
        normalized = (fa_action + 1.0) / 2.0
        port_indices = np.rint(normalized * (num_ports - 1)).astype(np.int64)
        port_indices = np.clip(port_indices, 0, num_ports - 1)
        
        # FA positions q = round [(a + 1) / 2 * (Np - 1)]
        fa_positions = np.zeros((L, 2, U), dtype=np.float64)
        for l in range(L):
            for u in range(U):
                port_idx = port_indices[l, u]
                fa_positions[l, :, u] = (self.available_fa_ports[port_idx])

        return (phase_shift_matrix, fa_positions)

    def step(self, action: np.ndarray):
        self.current_step += 1
        (self.phase_shift_matrix, proposed_fa_positions) = self._decode_action(action)
        # self.consumed_power_per_ap = np.sum(self.allocated_powers,axis=1)
        # self.consumed_power = np.sum(self.consumed_power_per_ap)
        (self.fa_feasible, self.fa_violation_count, self.fa_min_pair_distance) = self.check_fa_feasibility(proposed_fa_positions)
        if self.fa_feasible:
            self.FA_position = proposed_fa_positions
        self.H_l = (self.calculate_path_between_ap_sim(self.FA_position, self.element_position_matrix)) #FA position change
        self.B_l = (self.calculate_sim_transfer_matrix(self.phase_shift_matrix)) #phase-shift change
        self.h_user = self.get_channel()
        observation = self._get_obs(h_user=self.h_user)
        (self.user_sinr, self.user_data_rates, total_data_rates) = self._calculate_rates()
        #Condition about FA position
        if self.fa_feasible:
            self.reward = float(total_data_rates)
        else:
            self.reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        
        info = self._get_info()# print("info", info)
        return observation, self.reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.reward = 0.0
        self.consumed_power = 0.0
        self.user_sinr = None
        self.user_data_rates = None
        self.fa_feasible = True
        self.fa_violation_count = 0
        self.fa_min_pair_distance = None
        # self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.FA_position = (
            self.initial_FA_position.copy()
        )
        self.h_sim_ue = self._generate_sim_user_channel(rng=self.np_random) #SIM-UE fading
        self.H_l = (
            self.calculate_path_between_ap_sim(
                self.FA_position,
                self.element_position_matrix
            )
        )
        self.phase_shift_matrix = self.np_random.uniform(
            low=0.0, high=2.0 * np.pi,
            size=(self.ap_nums, self.layer_nums, self.total_element_per_layers) #(L, M, N)
        )
        self.B_l = self.calculate_sim_transfer_matrix(self.phase_shift_matrix)
        self.h_user = self.get_channel()
        observation = self._get_obs(h_user = self.h_user)
        info = self._get_info()
        
        return observation, info