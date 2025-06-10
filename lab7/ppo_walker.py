#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip (Walker2d‑v4)
# Contributors: Wei Hung and Alison Wen  ─  Filled & refined by ChatGPT
# -----------------------------------------------------------------------------

import random
import os
from collections import deque
from typing import Deque, List, Tuple, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
import argparse
import wandb
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# -----------------------------------------------------------------------------
# 简化的观察标准化
# -----------------------------------------------------------------------------

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        # 重新开始时先进行形状检查
        if len(x.shape) == 1 and len(self.mean.shape) > 0:
            x = x.reshape(1, -1)
            
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        # 更新均值和方差
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class NormEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = 10.0
        self.clip_reward = 10.0
        self.ret = 0
        self.gamma = 0.999
        self.epsilon = 1e-8
        self.training = True
        
        # 添加观察历史以改善状态表示
        self.history_len = 3
        self.obs_history = deque(maxlen=self.history_len)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 保存原始奖励到info字典中
        original_reward = reward
        if 'original_reward' not in info:
            info['original_reward'] = original_reward
        
        if self.training:
            # 更新返回和奖励标准化
            self.ret = self.ret * self.gamma + reward
            self.ret_rms.update(np.array([self.ret]))
            
            # 更新观察标准化
            self.obs_rms.update(np.array([obs]))
        
        # 标准化观察
        obs_normalized = np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
            -self.clip_obs, 
            self.clip_obs
        )
        
        # 改进的奖励标准化
        reward_normalized = np.clip(
            reward / (np.sqrt(self.ret_rms.var) + self.epsilon),
            -self.clip_reward,
            self.clip_reward
        )
        
        # 更新观察历史
        self.obs_history.append(obs_normalized)
        
        if terminated or truncated:
            self.ret = 0
            # 重置历史
            self.obs_history.clear()
            
        return obs_normalized, reward_normalized, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.ret = 0
        self.obs_history.clear()
        
        if self.training:
            self.obs_rms.update(np.array([obs]))
            
        # 标准化观察
        obs_normalized = np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
            -self.clip_obs, 
            self.clip_obs
        )
        
        # 初始化历史
        for _ in range(self.history_len):
            self.obs_history.append(obs_normalized)
        
        return obs_normalized, info

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        log_std_init: float = -2.0,
    ):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(in_dim, hidden_dim))
        self.fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.mu_layer = layer_init(nn.Linear(hidden_dim, out_dim), std=0.01)
        self.log_std = nn.Parameter(torch.ones(out_dim) * log_std_init)

    def forward(self, state: torch.Tensor):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        mu = self.mu_layer(x)
        
        log_std = torch.clamp(self.log_std, -10, 2.0)
        std = log_std.exp()
        
        dist = Normal(mu, std)
        
        if self.training:
            action = dist.sample()
        else:
            action = mu
            
        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        larger_dim = int(hidden_dim * 1.5)
        self.fc1 = layer_init(nn.Linear(in_dim, larger_dim))
        self.fc2 = layer_init(nn.Linear(larger_dim, larger_dim))
        self.fc3 = layer_init(nn.Linear(larger_dim, larger_dim))
        self.v_head = layer_init(nn.Linear(larger_dim, 1), std=0.5)

    def forward(self, state: torch.Tensor):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        value = self.v_head(x)
        return value.squeeze(-1)


# -----------------------------------------------------------------------------
# Advantage estimation
# -----------------------------------------------------------------------------

def compute_gae(
    next_value: torch.Tensor,
    rewards: List[torch.Tensor],
    masks: List[torch.Tensor],
    values: List[torch.Tensor],
    gamma: float,
    tau: float,
) -> List[torch.Tensor]:
    values = values + [next_value]
    gae = 0
    gae_returns: List[torch.Tensor] = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae + values[step])
    return gae_returns


# -----------------------------------------------------------------------------
# PPO minibatch iterator
# -----------------------------------------------------------------------------

def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    old_dists_mean: torch.Tensor = None,
    old_dists_std: torch.Tensor = None,
):
    batch_size = states.size(0)
    for _ in range(update_epoch):
        indices = torch.randperm(batch_size)
        for start in range(0, batch_size, mini_batch_size):
            idx = indices[start : start + mini_batch_size]
            if old_dists_mean is not None and old_dists_std is not None:
                yield (
                    states[idx, :],
                    actions[idx],
                    values[idx],
                    log_probs[idx],
                    returns[idx],
                    advantages[idx],
                    old_dists_mean[idx],
                    old_dists_std[idx],
                )
            else:
                yield (
                    states[idx, :],
                    actions[idx],
                    values[idx],
                    log_probs[idx],
                    returns[idx],
                    advantages[idx],
                )


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------

class PPOAgent:

    def __init__(self, env: gym.Env, args):
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.initial_batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.use_kl_penalty = args.use_kl_penalty
        self.kl_target = args.kl_target
        self.kl_coef = args.kl_coef
        self.use_value_clipping = args.use_value_clipping
        self.dynamic_batch_size = args.dynamic_batch_size
        self.early_stopping = args.early_stopping
        
        self.env = NormEnv(env)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(
            self.obs_dim, 
            self.action_dim,
            hidden_dim=args.hidden_dim,
            log_std_init=args.log_std_init
        ).to(self.device)
        
        self.critic = Critic(
            self.obs_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.states, self.actions = [], []
        self.rewards, self.values = [], []
        self.masks, self.log_probs = [], []
        self.dist_means, self.dist_stds = [], []

        self.total_step = 1
        self.is_test = False
        self.best_reward = -float('inf')
        
        self.model_ensemble = []
        self.ensemble_weights = []

        self.current_epsilon = self.epsilon * 0.5
        
        self.avg_kl = 0

    # ------------------------------------------------------------------ actions
    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).to(self.device)

        self.actor.train(not self.is_test)
        
        with torch.no_grad():
            action, dist = self.actor(state_tensor)
            log_prob = dist.log_prob(action).sum(-1)
            value = self.critic(state_tensor)

        if not self.is_test:
            self.states.append(state_tensor.detach())
            self.actions.append(action.detach())
            self.values.append(value.detach())
            self.log_probs.append(log_prob.detach())
            self.dist_means.append(dist.loc.detach())
            self.dist_stds.append(dist.scale.detach())
            
        return action.detach().cpu().numpy()

    def step(self, action: np.ndarray):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        original_reward = info.get('original_reward', reward)

        next_state = np.reshape(next_state, (1, -1)).astype(np.float32)
        reward = np.reshape(reward, (1, -1)).astype(np.float32)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, original_reward, done

    # ------------------------------------------------------------- optimisation
    def update_model(self, next_state: np.ndarray):
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        
        with torch.no_grad():
            next_value = self.critic(next_state_tensor)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.stack(self.states).to(self.device)
        actions = torch.stack(self.actions).to(self.device)
        old_logp = torch.stack(self.log_probs).to(self.device)
        returns = torch.cat(returns).to(self.device).detach()
        values_old = torch.cat(self.values).to(self.device).detach()
        advantages = returns - values_old
        
        old_dist_means = torch.stack(self.dist_means).to(self.device)
        old_dist_stds = torch.stack(self.dist_stds).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.dynamic_batch_size:
            effective_batch_size = min(
                self.initial_batch_size + int(self.total_step / 500000) * 16, 
                256
            )
        else:
            effective_batch_size = self.initial_batch_size

        progress = min(1.0, self.total_step / (self.num_episodes * self.rollout_len * 0.1))
        self.current_epsilon = self.epsilon * 0.4 + 0.4 * self.epsilon * progress
        
        if self.total_step < 1000000:
            actor_lr = self.actor_lr * (1.0 + 0.5 * (self.total_step / 1000000))
            critic_lr = self.critic_lr * (1.0 + 0.5 * (self.total_step / 1000000))
        else:
            decay = max(0.3, 1.0 - ((self.total_step - 1000000) / 700000))
            actor_lr = self.actor_lr * decay
            critic_lr = self.critic_lr * decay
        
        actor_lr = self.actor_lr
        critic_lr = self.critic_lr

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr

        actor_losses, critic_losses, ent_losses, kl_divs = [], [], [], []
        
        iterator = ppo_iter(
            self.update_epoch,
            effective_batch_size,
            states,
            actions,
            values_old,
            old_logp,
            returns,
            advantages,
            old_dist_means,
            old_dist_stds,
        )
        
        update_counter = 0
        max_updates = self.update_epoch * (len(states) // effective_batch_size + 1)
        avg_kl = 0
        
        for (
            state_b,
            action_b,
            old_value_b,
            old_logprob_b,
            return_b,
            adv_b,
            old_mean_b,
            old_std_b,
        ) in iterator:
            update_counter += 1
            
            value_b = self.critic(state_b)
            _, dist = self.actor(state_b)
            logp_b = dist.log_prob(action_b).sum(-1)
            
            old_dist = Normal(old_mean_b, old_std_b)
            
            kl_div = kl_divergence(old_dist, dist).sum(-1).mean()
            kl_divs.append(kl_div.item())
            
            ratio = torch.exp(logp_b - old_logprob_b)
            
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - self.current_epsilon, 1 + self.current_epsilon) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            
            if self.use_kl_penalty:
                policy_loss += self.kl_coef * kl_div
            
            entropy_loss = -dist.entropy().mean()
            
            if self.use_value_clipping:
                value_pred_clipped = old_value_b + torch.clamp(
                    value_b - old_value_b, -self.current_epsilon, self.current_epsilon
                )
                value_loss1 = F.mse_loss(value_b, return_b)
                value_loss2 = F.mse_loss(value_pred_clipped, return_b)
                value_loss = torch.max(value_loss1, value_loss2)
            else:
                value_loss = F.mse_loss(value_b, return_b)
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            self.actor_optimizer.zero_grad()
            (policy_loss + self.entropy_weight * entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            actor_losses.append(policy_loss.item())
            critic_losses.append(value_loss.item())
            ent_losses.append(entropy_loss.item())
            
            if self.early_stopping and len(kl_divs) > 1:
                avg_kl = np.mean(kl_divs)
                if avg_kl > self.kl_target * 1.5:
                    print(f"Early stopping at update {update_counter}/{max_updates} due to high KL: {avg_kl:.5f}")
                    break
        
        self.avg_kl = avg_kl

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []
        self.dist_means, self.dist_stds = [], []

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy_loss": np.mean(ent_losses),
            "kl_div": np.mean(kl_divs) if kl_divs else 0,
            "epsilon": self.current_epsilon,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
        }

    # ---------------------------------------------------------------- training
    def train(self):
        self.is_test = False
        state, _ = self.env.reset()
        state = np.expand_dims(state, axis=0)

        episode_count = 0
        episode_reward = 0
        episode_length = 0
        episode_original_reward = 0
        
        for step in tqdm(range(1, self.num_episodes * self.rollout_len + 1)):
            self.total_step += 1
            
            action = self.select_action(state).reshape(self.action_dim,)
            next_state, reward, original_reward, done = self.step(action)
            
            episode_reward += reward[0][0]
            episode_original_reward += original_reward
            episode_length += 1

            if self.total_step % 500000 == 0:
                    checkpoint_path = f'./walker_models/{self.total_step}.pt'
                    torch.save({
                        'actor':  self.actor.state_dict(),
                        'critic': self.critic.state_dict(),
                        'steps':  step,
                        'obs_rms_mean':  self.env.obs_rms.mean,
                        'obs_rms_var':   self.env.obs_rms.var,
                        'obs_rms_count': self.env.obs_rms.count,
                        'ret_rms_mean':  self.env.ret_rms.mean,
                        'ret_rms_var':   self.env.ret_rms.var,
                        'ret_rms_count': self.env.ret_rms.count,
                    }, checkpoint_path)
            
            if done[0][0]:
                episode_count += 1
                
                wandb.log({
                    "episode_reward": episode_original_reward,
                    "episode_shaped_reward": episode_reward,
                    "episode_length": episode_length,
                    "episode": episode_count,
                    "total_steps": self.total_step,
                })
                
                print(f"Episode {episode_count}: Original = {episode_original_reward:.1f}, Shaped = {episode_reward:.1f}, Length = {episode_length}")
        
                if episode_original_reward > 2000:
                    checkpoint_path = f'./walker_models/{self.total_step}_{episode_original_reward:.0f}.pt'
                    torch.save({
                        'actor':  self.actor.state_dict(),
                        'critic': self.critic.state_dict(),
                        'steps':  step,
                        'obs_rms_mean':  self.env.obs_rms.mean,
                        'obs_rms_var':   self.env.obs_rms.var,
                        'obs_rms_count': self.env.obs_rms.count,
                        'ret_rms_mean':  self.env.ret_rms.mean,
                        'ret_rms_var':   self.env.ret_rms.var,
                        'ret_rms_count': self.env.ret_rms.count,
                    }, checkpoint_path)
        
                state, _ = self.env.reset()
                state = np.expand_dims(state, axis=0)
                
                if episode_original_reward > self.best_reward and self.total_step < 1000000:
                    self.best_reward = episode_original_reward
                    torch.save({
                        'actor':  self.actor.state_dict(),
                        'critic': self.critic.state_dict(),
                        'steps':  step,
                        'obs_rms_mean':  self.env.obs_rms.mean,
                        'obs_rms_var':   self.env.obs_rms.var,
                        'obs_rms_count': self.env.obs_rms.count,
                        'ret_rms_mean':  self.env.ret_rms.mean,
                        'ret_rms_var':   self.env.ret_rms.var,
                        'ret_rms_count': self.env.ret_rms.count,
                    }, './walker_models/best.pt')

                    print(f"New best model saved with original reward {episode_original_reward:.1f}")
                
                episode_reward = 0
                episode_original_reward = 0
                episode_length = 0
            else:
                state = next_state
            
            if step % self.rollout_len == 0:
                stats = self.update_model(next_state)
                
                wandb.log({
                    "reward": episode_original_reward,
                    "actor_loss": stats["actor_loss"],
                    "critic_loss": stats["critic_loss"],
                    "entropy_loss": stats["entropy_loss"],
                    "kl_div": stats["kl_div"],
                    "epsilon": stats["epsilon"],
                    "actor_lr": stats["actor_lr"],
                    "critic_lr": stats["critic_lr"],
                    "total_steps": self.total_step,
                    "avg_kl": self.avg_kl,
                })

        torch.save({
            'actor':  self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'steps':  step,
            'obs_rms_mean':  self.env.obs_rms.mean,
            'obs_rms_var':   self.env.obs_rms.var,
            'obs_rms_count': self.env.obs_rms.count,
            'ret_rms_mean':  self.env.ret_rms.mean,
            'ret_rms_var':   self.env.ret_rms.var,
            'ret_rms_count': self.env.ret_rms.count,
        }, './walker_models/final.pt')
        
        self.env.close()

    # ---------------------------------------------------------------- testing
    def test(self, video_folder: str, record_video: bool = True):
        self.is_test = True
        self.env.training = False

        if record_video:
            try:
                base_test_env = gym.make("Walker2d-v4", render_mode="rgb_array")
                test_env = NormEnv(base_test_env)
                test_env.training = False
                test_env.obs_rms = self.env.obs_rms
                test_env.ret_rms = self.env.ret_rms

                test_env = gym.wrappers.RecordVideo(test_env, video_folder=video_folder)
            except Exception as e:
                base_test_env = gym.make("Walker2d-v4")
                test_env = NormEnv(base_test_env)
                test_env.training = False
                test_env.obs_rms = self.env.obs_rms
                test_env.ret_rms = self.env.ret_rms
                record_video = False
        else:
            base_test_env = gym.make("Walker2d-v4")
            test_env = NormEnv(base_test_env)
            test_env.training = False
            test_env.obs_rms = self.env.obs_rms
            test_env.ret_rms = self.env.ret_rms

        state, _ = test_env.reset(seed=self.seed)
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < 1000:
            state_normalized = np.expand_dims(state, axis=0)

            action = self.select_action(state_normalized).reshape(self.action_dim,)

            next_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            state = next_state
            original_reward = info.get('original_reward', reward)
            total_reward += original_reward
            step_count += 1

        print(f"Test score: {total_reward:.2f}, Steps: {step_count}")
        test_env.close()

        return total_reward


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-optimized")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)  
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--entropy-weight", type=float, default=0.025)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--rollout-len", type=int, default=768)
    parser.add_argument("--update-epoch", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--use-kl-penalty", action="store_true", default=False)
    parser.add_argument("--kl-target", type=float, default=0.05)
    parser.add_argument("--kl-coef", type=float, default=0.0001)
    parser.add_argument("--use-value-clipping", action="store_true", default=False)
    parser.add_argument("--dynamic-batch-size", action="store_true", default=False)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--video-folder", type=str, default="videos")
    parser.add_argument("--model-path", type=str, default="./walker_models/ppo_walker_best.pt")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    seed_torch(args.seed)
    
    if args.test and not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)

    env = gym.make("Walker2d-v4")
    
    

    agent = PPOAgent(env, args)
    
    if args.test:
        checkpoint = torch.load(args.model_path, weights_only=False)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.env.obs_rms.mean   = checkpoint['obs_rms_mean']
        agent.env.obs_rms.var    = checkpoint['obs_rms_var']
        agent.env.obs_rms.count  = checkpoint['obs_rms_count']
        agent.env.ret_rms.mean   = checkpoint['ret_rms_mean']
        agent.env.ret_rms.var    = checkpoint['ret_rms_var']
        agent.env.ret_rms.count  = checkpoint['ret_rms_count']
        print(f"Loaded best model with reward: {checkpoint.get('reward', 'unknown')}")
        agent.test(args.video_folder)
    else:
        wandb.init(
            project="DLP‑Lab7‑PPO‑Walker",
            name=f"walker‑ppo‑optimized‑{args.seed}",
            config=vars(args),
            save_code=True
            )
        agent.train()

    env.close()
    wandb.finish()