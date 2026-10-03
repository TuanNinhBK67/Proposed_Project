import os
import time
import random
from collections import deque
from dataclasses import dataclass, asdict
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from FA_SIM.Single_agent_case.Initial_Code.cf_mmimo_env import overallEnv

try:
    import wandb
except ImportError:
    wandb = None

@dataclass
class PPOConfig:
    # Reproducibility / device
    seed: int = 42
    cuda: bool = True
    torch_deterministic: bool = True

    # Training
    total_timesteps: int = 2_000_000 #2_000_000 - code goc --> total_timesteps / max_episode_steps = episode
    num_envs: int = 1 #1
    num_steps: int = 2048 #128 - code goc
    learning_rate: float = 1e-3 #1e-4
    anneal_lr: bool = True

    # PPO
    gamma: float = 0.9 #0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32 # 8
    update_epochs: int = 10 
    clip_coef: float = 0.4 # 0.2
    norm_adv: bool = True
    clip_vloss: bool = True
    ent_coef: float = 1e-4 #1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None #0.015

    # Logging / saving
    run_name: str = "ppo_cfmmimo"
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 50
    print_interval: int = 10

    # Optional W&B
    use_wandb: bool = False
    wandb_project: str = "FA-SIM-CF-mMIMO"

    # Environment
    ap_nums: int = 2 #3
    antenna_nums: int = 3 #4
    user_equipment_nums: int = 3 #4
    layer_nums: int = 2 #3
    num_elements_side: int = 3 #5
    FA_region_size_factor: float = 4.0
    frequency_hz: float = 28e9
    bandwidth_hz: float = 10e6
    area_size: float = 200.0
    noise_psd_dbm_hz: float = -174.0
    noise_figure_db: float = 0.0
    ap_transmit_power_dBm: float = 10.0
    ap_height_m: float = 15.0
    ue_height_m: float = 1.65
    path_loss_exp: float = 3.5
    ref_distance_m: float = 1.0
    transmit_path_num: int = 4
    receive_path_num: int = 4
    rician_factor_db: float = 10.0
    max_episode_steps: int = 10 # 1 episode = 10 env timesteps

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches

    def env_kwargs(self):
        return {
            "ap_nums": self.ap_nums,
            "antenna_nums": self.antenna_nums,
            "user_equipment_nums": self.user_equipment_nums,
            "layer_nums": self.layer_nums,
            "num_elements_side": self.num_elements_side,
            "FA_region_size_factor": self.FA_region_size_factor,
            "frequency_hz": self.frequency_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "area_size": self.area_size,
            "noise_psd_dbm_hz": self.noise_psd_dbm_hz,
            "noise_figure_db": self.noise_figure_db,
            "ap_transmit_power_dBm": self.ap_transmit_power_dBm,
            "ap_height_m": self.ap_height_m,
            "ue_height_m": self.ue_height_m,
            "path_loss_exp": self.path_loss_exp,
            "ref_distance_m": self.ref_distance_m,
            "transmit_path_num": self.transmit_path_num,
            "receive_path_num": self.receive_path_num,
            "rician_factor_db": self.rician_factor_db,
            "max_episode_steps": self.max_episode_steps,
        }

# Environment
def make_env(config: PPOConfig, env_idx: int):
    def thunk():
        env = overallEnv(**config.env_kwargs())
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        clipped_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10.0, 10.0).astype(np.float32),
            observation_space=clipped_space,
        )
        env.action_space.seed(config.seed + env_idx)
        env.observation_space.seed(config.seed + env_idx)
        return env
    return thunk

# PPO network
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    #The environment requires actions in [-1, 1].
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_size), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))

    def get_value(self, x):
        return self.critic(x)

    @staticmethod
    def _atanh(x):
        x = torch.clamp(x, -0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        if action is None:
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
        else:
            # Stored rollout actions are already in [-1, 1].
            action = torch.clamp(action, -0.999999, 0.999999)
            raw_action = self._atanh(action)

        logprob = (dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(x)
        return action, logprob, entropy, value

def save_checkpoint(path, agent, optimizer, iteration, global_step, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
            "global_step": global_step,
            "config": asdict(config),
        },
        path,
    )

def log_episode_statistics(infos, writer, global_step, episode_returns):
    if "episode" not in infos:
        return 0, []
    episode_info = infos["episode"]

    if not isinstance(episode_info, dict) or "r" not in episode_info:
        return 0, []
    returns = np.asarray(episode_info["r"])
    lengths = np.asarray(episode_info["l"])
    mask = infos.get("_episode", np.ones_like(returns, dtype=bool))
    mask = np.asarray(mask, dtype=bool)
    completed_returns = []

    for ep_return, ep_length in zip(returns[mask], lengths[mask]):
        ep_return = float(ep_return)
        ep_length = int(ep_length)
        
        episode_returns.append(ep_return)
        completed_returns.append(ep_return)
        writer.add_scalar(
            "charts/episode_return_vs_step",
            ep_return,
            global_step,
        )
        writer.add_scalar(
            "charts/episode_length_vs_step",
            ep_length,
            global_step,
        )
    return len(completed_returns), completed_returns

# Training
def train(config: PPOConfig):
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
            raise ImportError(
                "Weights & Biases is not installed. Run: pip install wandb"
            )
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=asdict(config),
            sync_tensorboard=True,
        )

    # Vector environments
    envs = gym.vector.SyncVectorEnv([make_env(config, i) for i in range(config.num_envs)])
    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    observation_size = int(np.prod(obs_shape))
    action_size = int(np.prod(action_shape))
    print("=" * 70)
    print("Device              :", device)
    print("Observation shape   :", obs_shape)
    print("Observation size    :", observation_size)
    print("Action shape        :", action_shape)
    print("Action size         :", action_size)
    print("Batch size          :", config.batch_size)
    print("Minibatch size      :", config.minibatch_size)
    print("=" * 70)

    # The current environment should be (96,) observation and (261,) action
    # for L=3, K=U=4, M=3, N=25.
    assert np.all(envs.single_action_space.low == -1.0)
    assert np.all(envs.single_action_space.high == 1.0)
    
    # Agent / optimizer
    agent = PPOAgent(observation_size=observation_size, action_size=action_size,).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5,)

    # -----------------------------
    # Rollout storage
    # -----------------------------
    obs = torch.zeros(
        (config.num_steps, config.num_envs) + obs_shape,
        dtype=torch.float32,
        device=device,
    )

    actions = torch.zeros(
        (config.num_steps, config.num_envs) + action_shape,
        dtype=torch.float32,
        device=device,
    )

    logprobs = torch.zeros(
        (config.num_steps, config.num_envs),
        dtype=torch.float32,
        device=device,
    )

    rewards = torch.zeros(
        (config.num_steps, config.num_envs),
        dtype=torch.float32,
        device=device,
    )

    dones = torch.zeros(
        (config.num_steps, config.num_envs),
        dtype=torch.float32,
        device=device,
    )

    values = torch.zeros(
        (config.num_steps, config.num_envs),
        dtype=torch.float32,
        device=device,
    )

    # Start rollout
    global_step = 0
    start_time = time.time()

    next_obs_np, _ = envs.reset(seed=config.seed)
    next_obs = torch.as_tensor(
        next_obs_np,
        dtype=torch.float32,
        device=device,
    )

    next_done = torch.zeros(
        config.num_envs,
        dtype=torch.float32,
        device=device,
    )

    num_iterations = config.total_timesteps // config.batch_size
    episode_returns = deque(maxlen=100)
    raw_reward_window = deque(maxlen=1000)
    best_mean_return = -np.inf
    completed_episodes = 0

    # PPO iterations
    for iteration in range(1, num_iterations + 1):
        # Learning-rate annealing
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = (frac * config.learning_rate)
        # Collect rollout
        for step in range(config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = (agent.get_action_and_value(next_obs))
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()
            next_obs_np, reward_np, terminated_np, truncated_np, infos = (envs.step(action.cpu().numpy().astype(np.float32)))
            
            done_np = np.logical_or(
                terminated_np,
                truncated_np,
            )

            rewards[step] = torch.as_tensor(
                reward_np,
                dtype=torch.float32,
                device=device,
            )

            next_obs = torch.as_tensor(
                next_obs_np,
                dtype=torch.float32,
                device=device,
            )

            next_done = torch.as_tensor(
                done_np,
                dtype=torch.float32,
                device=device,
            )

            for r in reward_np:
                raw_reward_window.append(float(r))

            # Environment diagnostics
            writer.add_scalar(
                "charts/mean_step_reward",
                float(np.mean(reward_np)),
                global_step,
            )

            if "fa_feasible" in infos:
                writer.add_scalar(
                    "env/fa_feasible_fraction",
                    float(np.mean(np.asarray(infos["fa_feasible"], dtype=float))),
                    global_step,
                )

            if "sum_rate" in infos:
                sum_rates = np.asarray(infos["sum_rate"], dtype=float)
                if np.any(np.isfinite(sum_rates)):
                    writer.add_scalar(
                        "env/mean_sum_rate",
                        float(np.nanmean(sum_rates)),
                        global_step,
                    )
            num_new_episodes, completed_returns = log_episode_statistics(
                infos,
                writer,
                global_step,
                episode_returns,
            )
            completed_episodes += num_new_episodes
        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)

            advantages = torch.zeros_like(
                rewards,
                device=device,
            )

            lastgaelam = torch.zeros(
                config.num_envs,
                dtype=torch.float32,
                device=device,
            )

            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]

                delta = (rewards[t] + config.gamma * next_values * next_nonterminal - values[t])
                lastgaelam = (delta + config.gamma * config.gae_lambda * next_nonterminal * lastgaelam)
                advantages[t] = lastgaelam

            returns = advantages + values
            
        # Flatten rollout batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_actions = actions.reshape((-1,) + action_shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(config.batch_size)

        clipfracs = []

        # Defaults so logging works even if target_kl exits early.
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        approx_kl = torch.tensor(0.0, device=device)

        # PPO update
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(
                0,
                config.batch_size,
                config.minibatch_size,
            ):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = (agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds],))
                logratio = (newlogprob - b_logprobs[mb_inds])
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]

                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std()+ 1e-8)

                # Policy loss
                pg_loss_1 = (-mb_advantages * ratio)
                pg_loss_2 = (-mb_advantages * torch.clamp(ratio, 1.0 - config.clip_coef,1.0 + config.clip_coef,))
                pg_loss = torch.max(pg_loss_1, pg_loss_2,).mean()

                # Value loss
                newvalue = newvalue.view(-1)

                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]).pow(2)
                    v_clipped = (b_values[mb_inds] + torch.clamp( newvalue - b_values[mb_inds], -config.clip_coef, config.clip_coef,))
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]).pow(2)
                    v_loss = (0.5 * torch.max(v_loss_unclipped, v_loss_clipped,).mean())

                else:
                    v_loss = (0.5 * ( newvalue - b_returns[mb_inds]).pow(2).mean())
                # Entropy / total loss
                entropy_loss = entropy.mean()

                loss = ( pg_loss - config.ent_coef * entropy_loss + config.vf_coef  * v_loss)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(),
                    config.max_grad_norm,
                )
                optimizer.step()
            if (config.target_kl is not None and approx_kl.item() > config.target_kl):
                break
        # Diagnostics
        with torch.no_grad():
            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()

            var_y = np.var(y_true)
            explained_var = (np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y )

        sps = int( global_step / max(time.time() - start_time, 1e-8))

        mean_raw_reward = (
            float(np.mean(raw_reward_window))
            if raw_reward_window
            else np.nan
        )

        mean_episode_return = (
            float(np.mean(episode_returns))
            if episode_returns
            else np.nan
        )

        writer.add_scalar(
            "charts/learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar(
            "charts/SPS",
            sps,
            global_step,
        )
        writer.add_scalar(
            "charts/mean_raw_reward_1000",
            mean_raw_reward,
            global_step,
        )

        if np.isfinite(mean_episode_return):
            writer.add_scalar(
                "charts/mean_episode_return_100_vs_step",
                mean_episode_return,
                global_step,
            )
            writer.add_scalar(
                "charts/mean_episode_return_100_vs_episode",
                mean_episode_return,
                completed_episodes,
            )

        writer.add_scalar(
            "losses/policy_loss",
            pg_loss.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/value_loss",
            v_loss.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/entropy",
            entropy_loss.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/approx_kl",
            approx_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/clipfrac",
            float(np.mean(clipfracs))
            if clipfracs
            else 0.0,
            global_step,
        )
        writer.add_scalar(
            "charts/explained_variance",
            explained_var,
            global_step,
        )

        # Optional W&B direct logs
        if config.use_wandb and wandb is not None:
            wandb.log(
                {
                    "global_step": global_step,
                    "mean_raw_reward_1000": mean_raw_reward,
                    "mean_episode_return_100": mean_episode_return,
                    "policy_loss": pg_loss.item(),
                    "value_loss": v_loss.item(),
                    "approx_kl": approx_kl.item(),
                    "SPS": sps,
                },
                step=global_step,
            )
        # Checkpoints

        if (iteration % config.save_interval == 0 or iteration == num_iterations):
            save_checkpoint(
                os.path.join(
                    checkpoint_path,
                    f"checkpoint_iter_{iteration}.pt",
                ),
                agent,
                optimizer,
                iteration,
                global_step,
                config,
            )

        if (np.isfinite(mean_episode_return) and mean_episode_return > best_mean_return):
            best_mean_return = mean_episode_return
            save_checkpoint(
                os.path.join(
                    checkpoint_path,
                    "best_model.pt",
                ),
                agent,
                optimizer,
                iteration,
                global_step,
                config,
            )

        # Console
        if (iteration == 1 or iteration % config.print_interval == 0):
            print(
                f"[Iter {iteration:5d}/{num_iterations}] "
                f"step={global_step:8d} | "
                f"reward={mean_raw_reward:8.4f} | "
                f"ep_return={mean_episode_return:8.4f} | "
                f"pg={pg_loss.item():8.4f} | "
                f"v={v_loss.item():8.4f} | "
                f"KL={approx_kl.item():8.6f} | "
                f"SPS={sps}"
            )

    # Final save / cleanup
    save_checkpoint(
        os.path.join(
            checkpoint_path,
            "final_model.pt",
        ),
        agent,
        optimizer,
        num_iterations,
        global_step,
        config,
    )

    envs.close()
    writer.close()

    if config.use_wandb and wandb is not None:
        wandb.finish()

    print("Training completed.")
    print("Logs       :", log_path)
    print("Checkpoints:", checkpoint_path)

if __name__ == "__main__":
    config = PPOConfig()
    train(config)