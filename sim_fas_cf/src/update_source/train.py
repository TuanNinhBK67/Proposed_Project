# training.py 
# def main():  cfg → env, eval_env → actor, critic, trainer, buffer → run()
# def run():
#     warmup()
#     for it in range(total_steps // (T * E)):
#         for step in range(T):
#             out = collect(step)                    # actor.act trên (E·L, obs_dim)
#             p, phase, ports = to_env_action(out)   # latent → vật lý, gộp port theo ctrl
#             obs, share_obs, r, done, info = env.step(p, phase, ports)
#             insert(...)
#         compute()                                  # GAE
#         trainer.train(buffer); buffer.after_update()
#         log / save / eval (deterministic)
