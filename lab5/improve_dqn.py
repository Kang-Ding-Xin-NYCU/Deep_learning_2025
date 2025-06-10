# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
from torch.nn import functional as F

gym.register_envs(ale_py)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)



def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, input_shape, num_actions, use_cnn=False):
        super(DQN, self).__init__()
        self.use_cnn = use_cnn
        ########## YOUR CODE HERE (5~10 lines) ##########
        if self.use_cnn:
            c, h, w = input_shape
            self.network = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
                NoisyLinear(7 * 7 * 64, 512), nn.ReLU(),
                NoisyLinear(512, num_actions)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_shape, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, num_actions)
            )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        if self.use_cnn:
            x = x / 255.0
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        priority = (abs(error) + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
    
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(probs), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
    
        # Importance sampling weights
        total = len(probs)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        ########## END OF YOUR CODE (for Task 3) ########## 
        return samples, indices, weights
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for i, error in zip(indices, errors):
            self.priorities[i] = (abs(error) + 1e-6) ** self.alpha
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def __len__(self):
        return len(self.buffer)

        

class DQNAgent:
    def __init__(self, env_name, args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.env_name = env_name
        if env_name == "CartPole-v1":
            self.preprocessor = lambda obs: obs
            input_shape = self.env.observation_space.shape[0]
            use_cnn = False
        else :
            self.preprocessor = AtariPreprocessor()
            dummy_input = np.zeros(self.env.observation_space.shape, dtype=np.uint8)
            input_shape = self.preprocessor.reset(dummy_input).shape
            use_cnn = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(input_shape, self.num_actions, use_cnn=use_cnn).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(input_shape, self.num_actions, use_cnn=use_cnn).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.n_step = 10
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.memory = PrioritizedReplayBuffer(capacity=args.memory_size)


    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        if isinstance(state, torch.Tensor):
            state_tensor = state.float().unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=2000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            if self.env_name == "CartPole-v1":
                state = self.preprocessor(obs)
            else :
                state = self.preprocessor.reset(obs)
            
            done = False
            total_reward = 0
            step_count = 0
            n_step_buffer = deque(maxlen=self.n_step)

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if self.env_name == "CartPole-v1":
                    next_state = self.preprocessor(next_obs)
                else:
                    next_state = self.preprocessor.step(next_obs)

                n_step_buffer.append((state, action, reward))

                if len(n_step_buffer) == self.n_step:
                    R = sum([n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])
                    s0, a0, _ = n_step_buffer[0]
                    self.memory.add((s0, a0, R, next_state, done), error=1.0)


                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "Total Steps": self.env_count
                    })
                    ########## YOUR CODE HERE  ##########
                    wandb.log({
                        "Q Net Weight Norm": sum(p.norm().detach().item() for p in self.q_net.parameters()),
                    })

                    wandb.log({
                        "Replay Buffer Size": len(self.memory.buffer)
                    })
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            wandb.log({
                "Replay Buffer Size": len(self.memory.buffer),
                "Current Epsilon": self.epsilon
            })
            ########## END OF YOUR CODE ##########  
            if self.env_count in [200_000, 400_000, 600_000, 800_000, 1_000_000]:
                model_path = os.path.join(self.save_dir, f"model_env{self.env_count}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model at env step {self.env_count} to {model_path}")
                
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()

        if self.env_name == "CartPole-v1":
            state = self.preprocessor(obs)
        else :
            state = self.preprocessor.reset(obs)
        
        done = False
        total_reward = 0

        while not done:
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).float().to(self.device)
            else:
                state_tensor = state.float().to(self.device)

            if len(state_tensor.shape) == 5:
                state_tensor = state_tensor.view(-1, state_tensor.shape[2], state_tensor.shape[3], state_tensor.shape[4])

            state_tensor = state_tensor.unsqueeze(0)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if self.env_name == "CartPole-v1":
                state = self.preprocessor(next_obs)
            else :
                state = self.preprocessor.step(next_obs)
            

        return total_reward

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        samples, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states      = torch.tensor(np.array(states),      dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions,              dtype=torch.int64,   device=self.device)
        rewards     = torch.tensor(rewards,              dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones,                dtype=torch.float32, device=self.device)
        weights     = torch.tensor(weights,              dtype=torch.float32, device=self.device)

        next_actions = self.q_net(next_states).argmax(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target_q = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        errors = torch.abs(q_values - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=120000)
    parser.add_argument("--lr", type=float, default= 1e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    if args.wandb_run_name == "cartpole-run":
        wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    else :
        wandb.init(project="DLP-Lab5-DQN-Pong", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run()