import os
import json
import math
import time
import random
import argparse
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from cf_mmimo_marl_env import CFMultiAgentEnv
from ppo_classical_agent import PPOConfig, layer_init

try:
    import wandb
except ImportError:
    wandb = None

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass
class MAPPOConfig(PPOConfig):
    # Inherits every PPO / environment hyper-parameter of ppo_classical_agent.PPOConfig,
    # so the single-agent baseline and this MARL version are directly comparable.
    run_name: str = "mappo_cfmmimo"

    # The networks are tiny (2x256) and the rollout is one env step at a time, so a single CPU
    # thread was measured faster than GPU or multi-threaded CPU. Use --cuda to override.
    cuda: bool = False
    cpu_threads: int = 1

    # MARL
    centralized_critic: bool = True  # True: MAPPO (critic sees all APs) | False: IPPO (critic sees own AP only)
    use_agent_id: bool = True        # append one-hot agent id to the actor (and IPPO critic) input
    obs_mode: str = "full_csi"       # "full_csi": H_l, h_sim_ue, phases, FA ports | "effective": h_user only
    obs_norm_eps: float = 1e-30      # epsilon of the observation normaliser (raw channel var is ~1e-10)
    eval_episodes: int = 100

    @property
    def steps_per_iteration(self):   # environment steps per PPO iteration
        return self.num_envs * self.num_steps

    @property
    def batch_size(self):            # (env step, agent) samples per PPO iteration
        return self.steps_per_iteration * self.ap_nums


def mlp(in_dim, out_dim, out_std):
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, out_dim), std=out_std),
    )


class MAPPOAgent(nn.Module):
    # One actor shared by all agents (parameter sharing) + a critic.
    # The environment requires actions in [-1, 1] -> tanh-squashed Gaussian.
    def __init__(self, obs_dim, act_dim, state_dim, n_agents, centralized_critic, use_agent_id):
        super().__init__()
        self.n_agents = n_agents
        self.centralized_critic = centralized_critic
        self.use_agent_id = use_agent_id
        self.actor_in_dim = obs_dim + (n_agents if use_agent_id else 0)
        self.critic_in_dim = state_dim if centralized_critic else self.actor_in_dim

        self.actor_mean = mlp(self.actor_in_dim, act_dim, out_std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = mlp(self.critic_in_dim, 1, out_std=1.0)
        self.register_buffer("agent_ids", torch.eye(n_agents))

    def make_inputs(self, local_obs, state):
        # local_obs: (L, obs_dim), state: (state_dim,) -> actor input (L, .), critic input (L, .)
        actor_in = torch.cat([local_obs, self.agent_ids], dim=-1) if self.use_agent_id else local_obs
        critic_in = state.unsqueeze(0).expand(self.n_agents, -1) if self.centralized_critic else actor_in
        return actor_in, critic_in

    def get_value(self, critic_in):
        return self.critic(critic_in).squeeze(-1)

    def get_action_and_value(self, actor_in, critic_in, raw_action=None, deterministic=False):
        mean = self.actor_mean(actor_in)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        if raw_action is None:
            raw_action = mean if deterministic else mean + std * torch.randn_like(mean)
        action = torch.tanh(raw_action)
        # Closed-form Gaussian log-prob / entropy (same values as torch.distributions.Normal,
        # without its per-call argument validation, which dominated the rollout time).
        # The tanh change-of-variables term gives the log-prob of the squashed action; the raw
        # action is stored in the rollout buffer, so no atanh round-trip is needed in the update.
        logprob = (
            -0.5 * ((raw_action - mean) / std).pow(2) - logstd - _HALF_LOG_2PI
            - torch.log(1.0 - action.pow(2) + 1e-6)
        ).sum(dim=-1)
        entropy = (0.5 + _HALF_LOG_2PI + logstd).sum(dim=-1)
        value = self.critic(critic_in).squeeze(-1)
        return raw_action, action, logprob, entropy, value


def save_checkpoint(path, agent, optimizer, env, iteration, global_step, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_normalizer": env.normalizer_state(),  # needed to reuse the policy later
            "iteration": iteration,
            "global_step": global_step,
            "config": asdict(config),
        },
        path,
    )


@torch.no_grad()
def evaluate(env, agent, device, episodes, mode):
    # mode: "random" (uniform actions) | "stochastic" (sample policy) | "deterministic" (mean action)
    env.training = False  # freeze the observation normaliser
    ep_returns, step_rates, feasible = [], [], []
    for _ in range(episodes):
        local, state, _ = env.reset()
        ep_return, done = 0.0, False
        while not done:
            if mode == "random":
                action = np.random.uniform(-1.0, 1.0, size=(env.n_agents, env.act_dim)).astype(np.float32)
            else:
                actor_in, critic_in = agent.make_inputs(
                    torch.as_tensor(local, device=device), torch.as_tensor(state, device=device))
                _, act, _, _, _ = agent.get_action_and_value(
                    actor_in, critic_in, deterministic=(mode == "deterministic"))
                action = act.cpu().numpy()
            local, state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            step_rates.append(float(info["sum_rate"]))
            feasible.append(float(info["fa_feasible"]))
        ep_returns.append(ep_return)
    env.training = True
    return {
        "episode_return_mean": float(np.mean(ep_returns)),
        "episode_return_std": float(np.std(ep_returns)),
        "sum_rate_per_step": float(np.mean(step_rates)),
        "fa_feasible_fraction": float(np.mean(feasible)),
    }


def train(config: MAPPOConfig):
    assert config.num_envs == 1, "this implementation runs a single environment"
    assert config.batch_size % config.num_minibatches == 0, (
        "batch_size must be divisible by num_minibatches"
    )
    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic
    torch.backends.cudnn.benchmark = not config.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(config.cpu_threads)

    # Logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.run_name}_{timestamp}"
    log_path = os.path.join(config.log_dir, run_name)
    checkpoint_path = os.path.join(config.checkpoint_dir, run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    if config.use_wandb:
        if wandb is None:
            raise ImportError("Weights & Biases is not installed. Run: pip install wandb")
        wandb.init(project=config.wandb_project, name=run_name, config=asdict(config), sync_tensorboard=True)

    # Environment (seeded before construction, like the single-agent baseline)
    env = CFMultiAgentEnv(config.env_kwargs(), obs_mode=config.obs_mode, norm_eps=config.obs_norm_eps)
    L, obs_dim, state_dim, act_dim = env.n_agents, env.obs_dim, env.state_dim, env.act_dim
    print("=" * 70)
    print("Device              :", device)
    print("Algorithm           :", "MAPPO (centralised critic)" if config.centralized_critic else "IPPO (local critic)")
    print("Agents (APs)        :", L)
    print("Observation mode    :", config.obs_mode)
    print("Local obs / agent   :", obs_dim)
    print("Global state        :", state_dim)
    print("Action / agent      :", act_dim)
    print("Batch (env steps)   :", config.steps_per_iteration)
    print("Batch (samples)     :", config.batch_size)
    print("Minibatch size      :", config.minibatch_size)
    print("=" * 70)

    agent = MAPPOAgent(
        obs_dim=obs_dim, act_dim=act_dim, state_dim=state_dim, n_agents=L,
        centralized_critic=config.centralized_critic, use_agent_id=config.use_agent_id,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Rollout storage: (step, agent, ...)
    T = config.num_steps
    actor_obs = torch.zeros((T, L, agent.actor_in_dim), device=device)
    critic_obs = torch.zeros((T, L, agent.critic_in_dim), device=device)
    raw_actions = torch.zeros((T, L, act_dim), device=device)
    logprobs = torch.zeros((T, L), device=device)
    values = torch.zeros((T, L), device=device)
    rewards = torch.zeros(T, device=device)  # shared team reward
    dones = torch.zeros(T, device=device)

    def to_inputs(local, state):
        return agent.make_inputs(
            torch.as_tensor(local, dtype=torch.float32, device=device),
            torch.as_tensor(state, dtype=torch.float32, device=device),
        )

    global_step = 0
    start_time = time.time()
    local, state, _ = env.reset(seed=config.seed)
    next_actor_obs, next_critic_obs = to_inputs(local, state)
    next_done = torch.zeros((), device=device)

    num_iterations = config.total_timesteps // config.steps_per_iteration
    episode_returns = deque(maxlen=100)
    best_mean_return = -np.inf
    completed_episodes = 0
    ep_return = 0.0

    for iteration in range(1, num_iterations + 1):
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * config.learning_rate

        # Collect rollout
        iter_rewards, iter_sum_rates, iter_feasible, iter_ep_returns = [], [], [], []
        for step in range(T):
            global_step += config.num_envs
            actor_obs[step] = next_actor_obs
            critic_obs[step] = next_critic_obs
            dones[step] = next_done
            with torch.no_grad():
                raw, action, logprob, _, value = agent.get_action_and_value(next_actor_obs, next_critic_obs)
            raw_actions[step] = raw
            logprobs[step] = logprob
            values[step] = value

            local, state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated or truncated
            rewards[step] = reward
            ep_return += reward
            iter_rewards.append(reward)
            iter_feasible.append(float(info["fa_feasible"]))
            if info["sum_rate"] is not None and np.isfinite(info["sum_rate"]):
                iter_sum_rates.append(float(info["sum_rate"]))

            if done:
                episode_returns.append(ep_return)
                iter_ep_returns.append(ep_return)
                completed_episodes += 1
                ep_return = 0.0
                local, state, _ = env.reset()
            next_actor_obs, next_critic_obs = to_inputs(local, state)
            next_done = torch.tensor(float(done), device=device)

        # GAE (per agent, shared team reward)
        with torch.no_grad():
            next_value = agent.get_value(next_critic_obs)  # (L,)
            advantages = torch.zeros_like(values)
            lastgaelam = torch.zeros(L, device=device)
            for t in reversed(range(T)):
                if t == T - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + config.gamma * next_values * next_nonterminal - values[t]
                lastgaelam = delta + config.gamma * config.gae_lambda * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # Flatten (step, agent) -> samples
        b_actor_obs = actor_obs.reshape(-1, agent.actor_in_dim)
        b_critic_obs = critic_obs.reshape(-1, agent.critic_in_dim)
        b_raw_actions = raw_actions.reshape(-1, act_dim)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_inds = np.arange(config.batch_size)

        clipfracs = []
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        approx_kl = torch.tensor(0.0, device=device)

        # PPO update
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                mb_inds = b_inds[start:start + config.minibatch_size]

                _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_actor_obs[mb_inds], b_critic_obs[mb_inds], raw_action=b_raw_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss_1 = -mb_advantages * ratio
                pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                # Value loss
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]).pow(2)
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -config.clip_coef, config.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * (newvalue - b_returns[mb_inds]).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()
            if config.target_kl is not None and approx_kl.item() > config.target_kl:
                break

        # Diagnostics
        with torch.no_grad():
            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y
            policy_std = torch.exp(agent.actor_logstd).mean().item()

        sps = int(global_step / max(time.time() - start_time, 1e-8))
        mean_episode_return = float(np.mean(episode_returns)) if episode_returns else np.nan
        mean_step_reward = float(np.mean(iter_rewards))

        # Same tag names as ppo_classical_agent so the TensorBoard curves overlay.
        # Step-level diagnostics are averaged per iteration to keep the event file small.
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/mean_step_reward", mean_step_reward, global_step)
        writer.add_scalar("charts/mean_raw_reward_1000", mean_step_reward, global_step)
        writer.add_scalar("env/fa_feasible_fraction", float(np.mean(iter_feasible)), global_step)
        if iter_sum_rates:
            writer.add_scalar("env/mean_sum_rate", float(np.mean(iter_sum_rates)), global_step)
        if iter_ep_returns:
            writer.add_scalar("charts/episode_return_vs_step", float(np.mean(iter_ep_returns)), global_step)
        if np.isfinite(mean_episode_return):
            writer.add_scalar("charts/mean_episode_return_100_vs_step", mean_episode_return, global_step)
            writer.add_scalar("charts/mean_episode_return_100_vs_episode", mean_episode_return, completed_episodes)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
        writer.add_scalar("losses/policy_std", policy_std, global_step)
        writer.add_scalar("charts/explained_variance", explained_var, global_step)

        if config.use_wandb and wandb is not None:
            wandb.log(
                {
                    "global_step": global_step,
                    "mean_step_reward": mean_step_reward,
                    "mean_episode_return_100": mean_episode_return,
                    "policy_loss": pg_loss.item(),
                    "value_loss": v_loss.item(),
                    "approx_kl": approx_kl.item(),
                    "SPS": sps,
                },
                step=global_step,
            )

        # Checkpoints
        if iteration % config.save_interval == 0 or iteration == num_iterations:
            save_checkpoint(os.path.join(checkpoint_path, f"checkpoint_iter_{iteration}.pt"),
                            agent, optimizer, env, iteration, global_step, config)
        if np.isfinite(mean_episode_return) and mean_episode_return > best_mean_return:
            best_mean_return = mean_episode_return
            save_checkpoint(os.path.join(checkpoint_path, "best_model.pt"),
                            agent, optimizer, env, iteration, global_step, config)

        # Console
        if iteration == 1 or iteration % config.print_interval == 0:
            print(
                f"[Iter {iteration:5d}/{num_iterations}] "
                f"step={global_step:8d} | "
                f"reward={mean_step_reward:8.4f} | "
                f"ep_return={mean_episode_return:8.4f} | "
                f"pg={pg_loss.item():8.4f} | "
                f"v={v_loss.item():8.4f} | "
                f"KL={approx_kl.item():8.6f} | "
                f"std={policy_std:5.3f} | "
                f"SPS={sps}"
            )

    save_checkpoint(os.path.join(checkpoint_path, "final_model.pt"),
                    agent, optimizer, env, num_iterations, global_step, config)

    # Final evaluation (normaliser frozen). "random" is the reference to judge learning against.
    results = {}
    for mode in ("random", "stochastic", "deterministic"):
        results[mode] = evaluate(env, agent, device, config.eval_episodes, mode)
        for key, val in results[mode].items():
            writer.add_scalar(f"eval/{mode}/{key}", val, global_step)
    with open(os.path.join(checkpoint_path, "results.json"), "w") as f:
        json.dump({"config": asdict(config), "eval": results}, f, indent=2)

    print("-" * 70)
    print(f"Final evaluation over {config.eval_episodes} episodes ({config.max_episode_steps} steps each)")
    for mode, res in results.items():
        print(f"  {mode:13s} | sum-rate/step = {res['sum_rate_per_step']:7.4f} | "
              f"episode return = {res['episode_return_mean']:8.4f} +- {res['episode_return_std']:.4f} | "
              f"FA feasible = {res['fa_feasible_fraction']:.3f}")
    print("-" * 70)

    writer.close()
    if config.use_wandb and wandb is not None:
        wandb.finish()
    print("Training completed.")
    print("Logs       :", log_path)
    print("Checkpoints:", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["mappo", "ippo"], default="mappo",
                        help="mappo: centralised critic | ippo: independent (local) critics")
    parser.add_argument("--obs-mode", choices=["full_csi", "effective"], default=None,
                        help="full_csi: underlying channels + phases + FA ports | effective: h_user only")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--cuda", action="store_true", help="use the GPU (slower for this network size)")
    args = parser.parse_args()

    config = MAPPOConfig()
    if args.variant == "ippo":
        config.centralized_critic = False
        config.run_name = "ippo_cfmmimo"
    if args.obs_mode is not None:
        config.obs_mode = args.obs_mode
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.seed is not None:
        config.seed = args.seed
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.cuda:
        config.cuda = True
    train(config)
