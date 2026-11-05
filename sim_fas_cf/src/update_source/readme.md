RULE 
1. Cho phép import 1 chiều: 
_ physics.py - không được import file nào của project 
_ env.py - không được import mappo, train
_ mappo.py - không được import physic, env
_ train.py - import cả file 

2. Nội dung file: 
_ physics.py - Chỉ gọi các hàm thuần túy: Geometry, Channel, SIM, FAS/AGA, Link
_ env.py - chỉ chứa SimFasEnv - chứa decode (action)
_ mappo.py - MLP, gaussian, Plackett-Luce, Actor, Critic, ValueNorm, Buffer
_ train.py - đọc config, baseline (heurisitc, other model), Runner(warmup, collect, insert, compute, train, eval)

3. Quy ước: 
_ env.py: 
obs - local observation (E, L, obs_dim) - float32
share_obs - global state (E, L, state_dim) - float32 --maybe (E, 1, state_dim)
reward - (E, L, 1) - float32 
done - (E, L) - bool --maybe (E, 1)
active(mặt nạ UE mà mỗi AP điều khiển) - (E, L, K) - bool
kích thước action gửi cho mappo - int -- (n_power = U + 1, n_phase = M.N, K, N_p)

4. Config vào 1 file config.yaml