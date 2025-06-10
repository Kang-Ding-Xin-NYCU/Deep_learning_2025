import torch
import gymnasium as gym
import numpy as np
import imageio
import os

from dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CartPole 環境（改成 rgb_array 可錄影）
env = gym.make("CartPole-v1", render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# 載入訓練好的模型
model = DQN(input_shape=obs_dim, num_actions=n_actions).to(device)
model.load_state_dict(torch.load("./results_carpole/best_model.pt"))
model.eval()

# 錄影設定
frames = []
episode_reward = 0
obs, _ = env.reset()
done = False

# ---------- 取代原本「錄影 / 單回合」區段 ----------

N_EPISODES = 20
all_rewards = []

for ep in range(1, N_EPISODES + 1):
    frames = []                        # 若要保留錄影就留著；不想錄影可註解
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        frame = env.render()
        frames.append(frame)

        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state).argmax(dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

    all_rewards.append(episode_reward)
    print(f"[Ep {ep:02d}] reward = {episode_reward}")

    output_path = f"./eval_videos_carpole/cartpole_demo_ep{ep:02d}.mp4"
    imageio.mimsave(output_path, frames, fps=30)

env.close()
print(f"Avg Reward: {np.mean(all_rewards):.2f}  (±{np.std(all_rewards):.2f})")
print(f"All Rewards: {all_rewards}")