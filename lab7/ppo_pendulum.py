#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple
import os

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
import time

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

class RunningMeanStd:
    def __init__(self, shape=(1,), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape,  dtype=np.float64)
        self.count = epsilon
    def update(self, x):
        x = np.asarray(x)
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta**2 * self.count * batch_count / tot) / tot
        self.mean, self.var, self.count = new_mean, new_var, tot
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        self.mean_net = nn.Sequential(
            init_layer_uniform(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            init_layer_uniform(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            init_layer_uniform(nn.Linear(hidden_dim, out_dim))
        )
        
        self.log_std = nn.Parameter(torch.zeros(out_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.distributions.Normal]:
        """Forward method implementation."""
        
        mu = self.mean_net(state)
        
        std = torch.exp(self.log_std).expand_as(mu)
        
        dist = Normal(mu, std)
        
        action = dist.rsample()

        return action, dist

class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        """Initialize."""
        super(Critic, self).__init__()

        self.value_net = nn.Sequential(
            init_layer_uniform(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            init_layer_uniform(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            init_layer_uniform(nn.Linear(hidden_dim, 1))
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        value = self.value_net(state)
        return value
    
def compute_gae(
    next_value: torch.Tensor,
    rewards: List[torch.Tensor],
    masks: List[torch.Tensor],
    values: List[torch.Tensor],
    gamma: float,
    tau: float
) -> List[torch.Tensor]:
    """Compute generalized advantage estimation."""
    
    values = values + [next_value]
    gae = 0
    gae_returns = []
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae + values[step])
    
    return gae_returns

def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    
    for _ in range(update_epoch):
        indices = torch.randperm(batch_size)
        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, batch_size)
            idx = indices[start_idx:end_idx]
            yield states[idx], actions[idx], values[idx], log_probs[idx], returns[idx], advantages[idx]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.max_steps = args.max_steps
        self.hidden_dim = args.hidden_dim
        self.action_std = args.action_std
        self.reward_rms = RunningMeanStd(shape=(1,))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(self.obs_dim, self.hidden_dim).to(self.device)
        
        with torch.no_grad():
            self.actor.log_std.fill_(np.log(self.action_std))

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        self.best_avg_reward = -float('inf')
        self.early_stop_threshold = -130
        self.model_saved = False

        self.total_step = 0
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, dist = self.actor(state)
            selected_action = dist.mean if self.is_test else action
            
            if not self.is_test:
                value = self.critic(state)
                self.states.append(state)
                self.actions.append(selected_action)
                self.values.append(value)
                self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.reward_rms.update(reward)
            norm_r = self.reward_rms.normalize(reward)
            self.rewards.append(torch.FloatTensor(norm_r).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        with torch.no_grad():
            next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            
            entropy = dist.entropy().mean()
            actor_loss = actor_loss - self.entropy_weight * entropy
            
            value = self.critic(state)
            critic_loss = F.mse_loss(return_, value)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
        critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0
        
        wandb.log({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "step": self.total_step
        })

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        self.is_test = False
        start_time = time.time()

        state, _ = self.env.reset()
        state = np.expand_dims(state, axis=0)

        scores = []
        score = 0
        episode_count = 0
        recent_scores = deque(maxlen=10)

        for ep in range(1, self.num_episodes + 1):
            score = 0
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    recent_scores.append(score)
                    avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else score
                    
                    print(f"Episode {episode_count}: Total Reward = {score:.2f}, Avg(10) = {avg_score:.2f}, Steps: {self.total_step}")
                    wandb.log({
                        "episode": episode_count,
                        "score": score,
                        "avg_score": avg_score,
                        "step": self.total_step
                    })
                    
                    score = 0
                
                if self.total_step >= self.max_steps:
                    print(f"Reached max steps: {self.max_steps}")
                    break

            if self.states:
                actor_loss, critic_loss = self.update_model(next_state)
            
            if ep % 10 == 0 or self.total_step >= self.max_steps:
                eval_score = self.evaluate(num_episodes=10)

                self.save_model(f"./ppo_models/ppo_pendulum_epoch{ep}.pt")
                if eval_score > self.best_avg_reward:
                    self.best_avg_reward = eval_score
                    self.save_model("./ppo_models/ppo_pendulum_best.pt")
                    print(f"New best model with avg reward: {eval_score:.2f}")
                
                if eval_score > self.early_stop_threshold:
                    print(f"Early stopping at {self.total_step} steps with avg reward: {eval_score:.2f}")
                    break
            
            if self.total_step >= self.max_steps:
                break

        self.save_model("./ppo_models/ppo_pendulum_final.pt")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds, {self.total_step} steps")
        
        if os.path.exists("ppo_pendulum_best.pt"):
            self.load_model("ppo_pendulum_best.pt")
        
        print("Performing final evaluation...")
        final_eval = self.evaluate(num_episodes=20, verbose=True)
        print(f"Final evaluation average: {final_eval:.2f}")
        
        self.env.close()
    
    def evaluate(self, num_episodes=10, verbose=False):
        """Evaluate current policy"""
        was_test = self.is_test
        self.is_test = True
        
        total_rewards = []
        for i in range(num_episodes):
            state, _ = self.env.reset(seed=self.seed + i)
            state = np.expand_dims(state, axis=0)
            done = False
            episode_reward = 0
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                episode_reward += reward[0][0]
                
                if done[0][0]:
                    break
                    
            total_rewards.append(episode_reward)
            if verbose:
                print(f"Test episode {i+1}: {episode_reward:.2f}")
        
        avg_reward = sum(total_rewards) / num_episodes
        print(f"Evaluation over {num_episodes} episodes: {avg_reward:.2f}")
        wandb.log({
            "eval_score": avg_reward,
            "step": self.total_step
        })
        
        self.is_test = was_test
        return avg_reward
    
    def save_model(self, path):
        """Save model to path"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
        self.model_saved = True
    
    def load_model(self, path):
        """Load model from path"""
        if not os.path.exists(path):
            print(f"Warning: Model file {path} does not exist!")
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"Model loaded from {path}")
        return True

    def test(self, num_episodes=20, render=False):
        """Test the agent."""
        self.is_test = True
        
        if render:
            try:
                tmp_env = self.env
                self.env = gym.wrappers.RecordVideo(
                    self.env, 
                    video_folder="./ppo_eval",
                    episode_trigger=lambda x: True
                )
            except Exception as e:
                print(f"Warning: Failed to setup video recording: {e}")
                tmp_env = None
        else:
            tmp_env = None

        all_scores = []
        for i in range(num_episodes):
            state, _ = self.env.reset(seed=self.seed + i)
            state = np.expand_dims(state, axis=0)
            done = False
            score = 0
            steps = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward[0][0]
                steps += 1
                
                if done[0][0]:
                    break
            
            all_scores.append(score)
            print(f"Test {i+1}: Score = {score:.2f}, Steps = {steps}")

        avg_score = sum(all_scores) / num_episodes
        print(f"Average score over {num_episodes} episodes: {avg_score:.2f}")
        
        if render and tmp_env is not None:
            self.env.close()
            self.env = tmp_env
        
        return all_scores, avg_score
 
def seed_torch(seed):
    """設置隨機種子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-ppo-fixed")
    parser.add_argument("--actor-lr", type=float, default=7e-4)
    parser.add_argument("--critic-lr", type=float, default=7e-3)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=4096)  
    parser.add_argument("--update-epoch", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--action-std", type=float, default=0.75)
    parser.add_argument("--test", action="store_true", help="Test mode (no training)")
    parser.add_argument("--model-path", type=str, default="ppo_pendulum_best.pt")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
 
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = args.seed
    seed_torch(seed)
    
    if not args.test:
        wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, save_code=True)
    
    agent = PPOAgent(env, args)
    
    if not args.test:
        agent.train()
        agent.test(num_episodes=1, render=True)
    else:
        success = agent.load_model(args.model_path)
        if success:
            scores, avg_score = agent.test(num_episodes=20, render=args.render)
            print(f"All scores: {scores}")
            print(f"Average score: {avg_score:.2f}")
        else:
            print("Failed to load model. Please train a model first or specify a valid model path.")