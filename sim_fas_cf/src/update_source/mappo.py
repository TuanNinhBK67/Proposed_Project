# mappo model
# # ===== 1. MLP =====          copy mlp.py: LayerNorm đầu vào, (Linear → act → LayerNorm) × n
# # ===== 2. Phân phối =====
# def gaussian_logp(z, mean, log_std) -> (B,)
# def gaussian_entropy(log_std)       -> (B,)
# def pl_sample(logits, active)       -> draws, logp, ent       # Plackett–Luce, lấy từ simfas cũ
# def pl_evaluate(logits, active, draws) -> logp, ent
# # ===== 3. Actor / Critic =====
# class Actor:  obs → MLP → {mean_power (U+1), mean_phase (M·N), logits_port (K, Np)} + log_std riêng
#               act(obs, active, deterministic) / evaluate(obs, active, z_p, z_f, draws)
# class Critic: share_obs → MLP → V
# # ===== 4. ValueNorm =====     copy valuenorm.py (không đổi)
# # ===== 5. Buffer =====        copy SharedReplayBuffer, CHỈ giữ feed_forward_generator
# # ===== 6. Trainer =====       copy r_mappo.py: train() + ppo_update() + cal_value_loss()
