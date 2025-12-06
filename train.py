import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import os
import glob
import time
import datetime
import numpy as np
import random
import torch
import argparse
from dogfight_env import DogfightEnv

# 멀티 프로세싱을 위한 설정 (멀티 프로세싱 과부하 방지 설정)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Curricullum 학습을 위한 클래스 선언
class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, ramp_ratio=0.5, verbose=0):
        """
        :param total_timesteps: 전체 학습 예정 스텝 (예: 2000000)
        :param ramp_ratio: 0.5이면 전체의 절반(100만) 시점에 난이도 100% 도달 
        (예: 0 step: EASY로 시작 -> 1000000 step: HARD 도달 -- 이후부터는 계속 HARD)
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        # 난이도가 오르는 구간의 길이 (예: 200만 * 0.5 = 100만 스텝)
        self.ramp_steps = total_timesteps * ramp_ratio 

    def _on_step(self) -> bool:
        # 1. 진행률 계산 (0.0 ~ 1.0)
        # 현재 스텝이 ramp_steps보다 작으면 0.x, 크면 1.0으로 고정(min 함수)
        current_progress = self.num_timesteps / self.ramp_steps
        progress = min(current_progress, 1.0)
        
        # 2. 환경에 진행률 전달
        # SubprocVecEnv(멀티 프로세싱)를 쓰고 있으므로 env_method로 함수 호출
        self.training_env.env_method("set_curriculum_progress", progress)
        
        # # (선택) 로그 출력: 10만 스텝마다 확인
        if self.num_timesteps % 100000 == 0:
            status = "Ramping Up" if progress < 1.0 else "Max Difficulty"
            print(f"[{self.num_timesteps}/{self.total_timesteps}] {status} | Difficulty Progress: {progress*100:.1f}%")
            
        return True


# 1. 메인 학습 함수
def train():
    # 예시: python train.py --difficulty HARD --steps 2000000 --seed 1 --algo SAC --n_envs 10 && python train.py --difficulty HARD --steps 2000000 --seed 2 --algo SAC --n_envs 10
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="SAC", help="Algorithm (PPO/SAC)")
    parser.add_argument("--difficulty", type=str, default="HARD", help="Difficulty")
    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    parser.add_argument("--steps", type=int, default=2000000, help="Total Timesteps")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load model")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment Folder Name")
    parser.add_argument("--units", type=int, default=128, choices=[64, 128, 256], help="Number of neurons (64/128/256)")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    
    args = parser.parse_args()

    # --- 폴더 이름 결정 로직 ---
    if args.exp_name:
        VERSION = args.exp_name
    else:
        # 폴더 이름에 유닛 수(64, 128, ...)가 자동으로 반영되도록 함
        VERSION = f"dogfight_{args.algo.lower()}_{args.units}_{args.difficulty.lower()}_seed{args.seed}"
    
    PATH = os.path.join("logs", VERSION)
    monitor_path = os.path.join(PATH, "monitor")
    checkpoint_path = os.path.join(PATH, "checkpoints")
    final_model_name = f"dogfight_pilot_{args.difficulty.lower()}.zip"
    final_model_path = os.path.join(checkpoint_path, final_model_name)

    # 전역 시드 고정 --> 공정한 학습
    print(f"Set Global Seed: {args.seed}")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    start_time = time.time()

    # ---------------------------------------------------------
    # 멀티 프로세싱 환경 생성 (make_vec_env)
    # ---------------------------------------------------------
    print(f"Creating {args.n_envs} parallel environments...")

    # n_envs가 1보다 크면 SubprocVecEnv(진짜 병렬) 사용
    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else None

    # make_vec_env가 알아서 여러 개의 DogfightEnv를 만들고 Monitor 로그도 저장함
    env = make_vec_env(
        DogfightEnv, 
        n_envs=args.n_envs, 
        seed=args.seed,
        vec_env_cls=vec_env_cls,
        env_kwargs={"difficulty": args.difficulty}, # 환경에 난이도 전달
        monitor_dir=monitor_path # 로그 저장 위치 (monitor_0.csv, monitor_1.csv 자동 생성)
    )

    print(f"로그 파일 저장 위치: {monitor_path}")
    print(f"체크포인트 저장 위치: {checkpoint_path}")

    # 콜백 설정
    real_save_freq = max(20000 // args.n_envs, 1)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=real_save_freq, 
        save_path=checkpoint_path,
        name_prefix="dogfight_model" 
    )

    curriculum_cb = CurriculumCallback(
        total_timesteps=args.steps, 
        ramp_ratio=0.5  # 절반까지만 난이도 올리고, 나머지는 Hard 유지, 200만 step 기준 Curriculum_V1의 ramp_ratio=0.5, Curriculum_V1의 ramp_ratio=0.5
    )

    # --- 모델 로드 및 생성 ---
    model = None
    reset_timesteps = True 

    # Case 1) 사용자가 특정 파일을 지정한 경우
    if args.load_model:
        if os.path.exists(args.load_model):
            raise FileNotFoundError("모델 파일을 찾을 수 없습니다.")
        
        print(f" 지정된 모델 로드: '{args.load_model}'")
        reset_timesteps = False 
        if args.algo == "PPO":
            model = PPO.load(args.load_model, env=env)
        else:
            model = SAC.load(args.load_model, env=env)

    # Case 2) 지정하진 않았지만, 같은 설정으로 학습하던 파일이 있는 경우 (자동 이어하기)
    elif os.path.exists(final_model_path):
        print(f"기존 학습 기록이 존재합니다.'{final_model_path}' (기존 모델에서 이어 학습을 진행하겠습니다.)")
        reset_timesteps = False
        
        if args.algo == "PPO":
            model = PPO.load(final_model_path, env=env)
        else:
            model = SAC.load(final_model_path, env=env)
    
    # Case 3) 새로 시작하기
    else:
        print(f" 새로운 {args.algo} 모델 생성 (Seed: {args.seed}, Units: {args.units})")
        
        # 입력받은 units 값 적용 ([64, 64] or [128, 128] ...)
        policy_kwargs = dict(net_arch=[args.units, args.units])
        
        if args.algo == "PPO":
            model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, seed=args.seed, policy_kwargs=policy_kwargs)
        else:
            model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.0003, seed=args.seed, policy_kwargs=policy_kwargs)

    # --- 학습 시작 ---
    print(f"학습을 시작합니다. 목표 스텝: {args.steps}")
    model.learn(
        total_timesteps=args.steps, 
        callback=[checkpoint_callback, curriculum_cb],
        reset_num_timesteps=reset_timesteps 
    )
    print("학습 완료")

    # --- 모델 저장 ---
    model.save(final_model_path)
    print(f"최종 모델 저장 완료: {final_model_path}")

    end_time = time.time()
    total_seconds = int(end_time - start_time)
    print(f"총 학습 시간: {str(datetime.timedelta(seconds=total_seconds))}")

if __name__ == "__main__":
    train()