import gymnasium as gym
from stable_baselines3 import PPO, SAC
import os
import time
import argparse
import numpy as np
from dogfight_env import DogfightEnv

def test():
    # --- 옵션 받기 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="SAC", help="Algorithm (PPO/SAC)")
    parser.add_argument("--difficulty", type=str, default="HARD", help="Difficulty (EASY/HARD/EVAL_LV.1/EVAL_LV.2)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (.zip)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to test")
    args = parser.parse_args()

    # --- 환경 생성 ---
    # 테스트하려는 난이도에 맞춰 환경 설정
    env = DogfightEnv(difficulty=args.difficulty)
    
    # --- 모델 로드 ---
    if not os.path.exists(args.model_path):
        print(f"Error: 모델 파일을 찾을 수 없습니다 -> {args.model_path}")
        return

    print(f"모델 로드 중입니다: {args.model_path}")
    if args.algo == "PPO":
        model = PPO.load(args.model_path, env=env)
    else:
        model = SAC.load(args.model_path, env=env)



    # --- 시뮬레이션 시작 ---
    for episode in range(args.episodes):
        obs, _ = env.reset() # (원하면 여기서 seed 고정 가능)
        env.max_steps = 500
        done = False
        total_reward = 0
        step_count = 0
        


        print(f"\n🎬 Episode {episode+1} Start ({args.difficulty} Mode)")
        
        while not done:
            # deterministic=True: 학습된 대로 가장 확률 높은 행동 선택
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # 시각화
            env.render()
            
            
            done = terminated or truncated
            
            if terminated:
                if info.get('enemy_health', 0) <= 0:
                    print("   >>>  적 격추! (Win)")
                elif info.get('agent_health', 0) <= 0:
                    print("   >>>  피격 당함! (Lose)")
                else:
                    print("   >>>  충돌 또는 이탈")
        
        print(f"   [종료] Steps: {step_count}, Reward: {total_reward:.2f}")

if __name__ == "__main__":
    test()