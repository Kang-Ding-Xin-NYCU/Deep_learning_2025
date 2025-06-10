#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Walker2d速度获取测试脚本

import gymnasium as gym
import numpy as np
import time

def test_velocity_methods(env, obs):
    """测试所有获取速度的方法"""
    results = {}
    
    # 测试方法1: data.qvel
    try:
        vel1 = env.unwrapped.data.qvel[0]
        results["data.qvel[0]"] = vel1
    except Exception as e:
        results["data.qvel[0]"] = f"错误: {str(e)}"
    
    # 测试方法2: sim.data.qvel (旧版)
    try:
        if hasattr(env.unwrapped, 'sim'):
            vel2 = env.unwrapped.sim.data.qvel[0]
            results["sim.data.qvel[0]"] = vel2
        else:
            results["sim.data.qvel[0]"] = "属性不存在"
    except Exception as e:
        results["sim.data.qvel[0]"] = f"错误: {str(e)}"
    
    # 测试方法3: get_body_com
    try:
        if hasattr(env.unwrapped, 'get_body_com'):
            vel3 = env.unwrapped.get_body_com("torso")[0]
            results["get_body_com('torso')[0]"] = vel3
        else:
            results["get_body_com('torso')[0]"] = "属性不存在"
    except Exception as e:
        results["get_body_com('torso')[0]"] = f"错误: {str(e)}"
    
    # 测试方法4: 观察向量中的不同位置
    try:
        # 测试观察向量的各个可能位置
        for i in range(min(17, len(obs))):
            results[f"obs[{i}]"] = obs[i]
    except Exception as e:
        results["obs"] = f"错误: {str(e)}"
    
    return results

def run_velocity_test():
    """运行速度获取测试"""
    print("创建Walker2d-v4环境...")
    env = gym.make("Walker2d-v4")
    
    print("重置环境...")
    obs, _ = env.reset(seed=42)
    
    # 打印观察空间
    print(f"观察空间: {env.observation_space}")
    print(f"观察向量形状: {obs.shape}")
    
    # 先执行一些随机动作让智能体移动
    print("\n执行随机动作...")
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"步骤 {i}: 测试各种速度获取方法:")
            results = test_velocity_methods(env, obs)
            
            print("\n速度获取测试结果:")
            for method, value in results.items():
                print(f"  {method}: {value}")
            print("-" * 50)
        
        if terminated or truncated:
            print("环境重置")
            obs, _ = env.reset()
    
    # 最终测试
    print("\n最终测试 - 带有细节分析:")
    # 检查env.unwrapped的属性
    print("\nenv.unwrapped的可用属性:")
    for attr in dir(env.unwrapped):
        if not attr.startswith('_'):  # 跳过私有属性
            print(f"  {attr}")
    
    # 最后一次尝试获取速度
    results = test_velocity_methods(env, obs)
    print("\n最终速度获取结果:")
    for method, value in results.items():
        print(f"  {method}: {value}")
    
    # 推荐最佳方法
    print("\n分析结果:")
    valid_methods = {k: v for k, v in results.items() if isinstance(v, (int, float)) and not k.startswith("obs[")}
    if valid_methods:
        best_method = max(valid_methods.items(), key=lambda x: abs(x[1]) if not isinstance(x[1], str) else 0)
        print(f"推荐使用方法: {best_method[0]} (值: {best_method[1]})")
    else:
        print("没有找到有效的速度获取方法!")
    
    env.close()
    print("\n测试完成!")

if __name__ == "__main__":
    run_velocity_test()