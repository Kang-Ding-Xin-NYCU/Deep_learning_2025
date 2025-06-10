#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
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
from typing import Tuple
import os

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

def initialize_orthogonal(layer: nn.Linear, gain: float = 1.0):
    """Initialize the weights with orthogonal initialization and bias with zeros."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, out_dim)
        self.fc_std = nn.Linear(64, out_dim)
        
        initialize_orthogonal(self.fc1, gain=np.sqrt(2))
        initialize_orthogonal(self.fc2, gain=np.sqrt(2))
        initialize_orthogonal(self.fc3, gain=np.sqrt(2))
        initialize_orthogonal(self.fc_mu, gain=0.01)
        initialize_orthogonal(self.fc_std, gain=0.01)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mu = torch.tanh(self.fc_mu(x)) * 2.0
        
        std = F.softplus(self.fc_std(x)) + 0.35
        
        dist = Normal(mu, std)
        action = dist.rsample()
        
        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        
        initialize_orthogonal(self.fc1, gain=np.sqrt(2))
        initialize_orthogonal(self.fc2, gain=np.sqrt(2))
        initialize_orthogonal(self.fc3, gain=np.sqrt(2))
        initialize_orthogonal(self.fc_out, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc_out(x)
        
        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.normalize_reward = args.normalize_reward
        self.clip_grad = args.clip_grad
        self.reward_scale = args.reward_scale
        
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = []
        
        self.best_score = -np.inf
        self.best_episode = 0

        self.total_step = 0

        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state_tensor)

        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1).detach()
            entropy = dist.entropy().sum(dim=-1).detach()
            value = self.critic(state_tensor).squeeze().detach()

            self.buffer.append((state, selected_action.detach().cpu().numpy(), log_prob, entropy, value))

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            scaled_reward = reward / self.reward_scale if self.normalize_reward else reward
            self.buffer[-1] += (scaled_reward, done)

        return next_state, reward, done

    def compute_advantages(self):
        """Compute discounted returns and advantages (no GAE)."""
        states, actions, log_probs, entropies, values, rewards, dones = zip(*self.buffer)

        returns = []
        R = 0 if dones[-1] else self.critic(torch.FloatTensor(self.buffer[-1][0]).to(self.device)).item()

        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1.0 - dones[i])
            returns.insert(0, R)

        advantages = [ret - val.item() for ret, val in zip(returns, values)]
        return returns, advantages


    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        if len(self.buffer) == 0:
            return 0, 0
            
        states, actions, log_probs, entropies, values, rewards, dones = zip(*self.buffer)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        log_probs = torch.stack(log_probs).detach()
        entropies = torch.stack(entropies).detach()
        values = torch.stack(values).detach()
        
        next_state = torch.FloatTensor(self.buffer[-1][0]).to(self.device) if not dones[-1] else None
        next_value = 0 if dones[-1] else self.critic(next_state).detach().item()
        
        returns, advantages = self.compute_advantages()
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        _, dist = self.actor(states)
        current_log_probs = dist.log_prob(actions).sum(-1)
        current_entropy = dist.entropy().sum(-1)
        
        policy_loss = -(current_log_probs * advantages).mean()
        entropy_loss = -self.entropy_weight * current_entropy.mean()
        actor_loss = policy_loss + entropy_loss
        
        current_values = self.critic(states).squeeze()
        value_loss = F.mse_loss(current_values, returns)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimizer.step()
        
        self.buffer = []
        
        wandb.log({
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": actor_loss.item() + value_loss.item(),
            "advantage_mean": advantages.mean().item() if len(advantages) > 0 else 0,
            "returns_mean": returns.mean().item() if len(returns) > 0 else 0,
        })
        
        return actor_loss.item(), value_loss.item()

    def save_model(self, path, score=None, episode=None):
        """Save model to path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        actor_state = self.actor.state_dict()
        critic_state = self.critic.state_dict()

        checkpoint = {
            'actor': actor_state,
            'critic': critic_state,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'score': score,
            'episode': episode,
        }

        try:
            torch.save(checkpoint, path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            try:
                torch.save({
                    'actor': actor_state,
                    'critic': critic_state,
                }, path)
                print(f"Model weights saved to {path} (without optimizer states)")
            except Exception as e2:
                print(f"All saving attempts failed: {e2}")
        
    def load_model(self, path):
        """Load model from path."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            score = checkpoint.get('score', float('-inf'))
            episode = checkpoint.get('episode', 0)
            print(f"Model loaded from {path}")
            return score, episode
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting alternative loading method...")
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                self.actor.load_state_dict(checkpoint['actor'])
                self.critic.load_state_dict(checkpoint['critic'])
                print(f"Model weights loaded from {path} (optimizer states omitted)")
                return float('-inf'), 0
            except Exception as e2:
                print(f"All loading attempts failed: {e2}")
                return float('-inf'), 0

    def train(self):
        """Train the agent."""
        self.is_test = False
        best_score = float('-inf')
        avg_reward = 0
        
        scores_window = []
        window_size = 20
        
        for ep in tqdm(range(1, self.num_episodes + 1)): 
            state, _ = self.env.reset()
            score = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                
                raw_reward = reward
                if self.normalize_reward:
                    raw_reward = reward * self.reward_scale
                
                score += raw_reward
                self.total_step += 1
                state = next_state
                
                if len(self.buffer) >= 32 or done:
                    actor_loss, critic_loss = self.update_model()
                    wandb.log({
                        "step": self.total_step,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                    })
            
            scores_window.append(score)
            if len(scores_window) > window_size:
                scores_window.pop(0)
            
            avg_score = np.mean(scores_window)
            
            print(f"Episode {ep}: Score = {score:.2f}, Avg Score = {avg_score:.2f}")
            
            wandb.log({
                "episode": ep,
                "score": score,
                "avg_score": avg_score,
                "total_steps": self.total_step,
            })
            
            if avg_score > best_score:
                best_score = avg_score
                self.save_model(f"{self.model_dir}/a2c_best.pt", score=avg_score, episode=ep)
                
            if ep % 10 == 0:
                self.save_model(f"{self.model_dir}/a2c_checkpoint_{ep}.pt", score=avg_score, episode=ep)
        
        self.save_model(f"{self.model_dir}/a2c_final.pt", score=avg_score, episode=self.num_episodes)
        print(f"Training finished. Best average score: {best_score:.2f}")

    def test(self, video_folder: str, num_tests: int = 20):
        """Test the agent."""
        self.is_test = True
        scores = []
        for i in range(num_tests):
            tmp_env = self.env
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder + f'{i}')
            state, _ = self.env.reset(seed=self.seed + i)
            done = False
            score = 0

            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    _, dist = self.actor(state_tensor)
                    action = dist.mean.cpu().numpy()
                    action = np.clip(action, -2.0, 2.0)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                score += reward
                state = next_state

            scores.append(score)
            print(f"Test #{i+1:02d} Seed #{self.seed + i} - Score: {score:.2f}")

        self.env.close()
        self.env = tmp_env

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        median_score = np.median(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)

        filtered_scores = sorted(scores)[2:]
        filtered_avg = np.mean(filtered_scores)
        filtered_std = np.std(filtered_scores)

        print(f"\nTest Results ({num_tests} episodes):")
        print(f"Average: {avg_score:.2f} ± {std_score:.2f}")
        print(f"Median: {median_score:.2f}")
        print(f"Min/Max: {min_score:.2f}/{max_score:.2f}")
        print(f"Filtered Average (removing worst 2): {filtered_avg:.2f} ± {filtered_std:.2f}")

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-3)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--normalize-reward", type=bool, default=True)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--clip-grad", type=float, default=0.7)
    parser.add_argument("--test", action="store_true", help="Test mode (no training)")
    parser.add_argument("--model-path", type=str, default="models/a2c_best.pt", help="Path to load model for testing")
    args = parser.parse_args()
    
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    if not args.test:
        wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True, config=vars(args))
        agent = A2CAgent(env, args)
        agent.train()
        agent.load_model(f"{agent.model_dir}/a2c_best.pt")
        agent.test("./a2c_eval/")
    else:
        agent = A2CAgent(env, args)
        agent.load_model(args.model_path)
        agent.test("./a2c_eval/", num_tests=20)