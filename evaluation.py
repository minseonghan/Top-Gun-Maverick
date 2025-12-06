import glob
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, SAC
from dogfight_env import DogfightEnv


# --- [설정] 실험 하이퍼파라미터 --- 
EVAL_CONFIG = {
    # 기본 평가 set up
    "episodes": 50,     # 모델당 평가할 에피소드 수 (많을수록 정확하지만 오래 걸림)
    "base_seed": 2025,  # 평가용 고정 시드, 에피소드 진행될 때마다 1씩 증가 (Ep 1: 2025, Ep 2: 2026 ,...)
    "max_steps": 500,    # max_step
    "difficulty": "EVAL_LV.2",  # EVAL_LV.1, EVAL_LV.2 (난이도 LV.1 < LV.2)

    """
    difficulty만 설정하면 나머지는 default로 결정되나, 
    밑의 setup으로 agenet와 적의 설정 값을 바꿔 난이도를 조정할 수 있음
    만약 값을 바꿔 난이도를 조정할 시, 하단의 evaluate_checkpoints 함수에서 변경 부분 주석 해제
    아래 세팅은 EVAL_LV.2와 동일 
    """
    "agent_turn_rate": 0.1,
    "enemy_turn_rate": 0.05,
    "agent_speed": 2.0,
    "enemy_speed": 1.5,
    "agent_angle": 30, 
    "enemy_angle": 25,
    "agent_hp": 10.0,
    "enemy_hp": 10.0,
    "prob_start": 0.5
}

# --- 1. 학습 로그(Monitor.csv) 불러오기 함수 ---
#    ->  Reward 그래프용
def load_training_data(exp_name, log_dirs):
    """
    여러 시드의 monitor.csv 파일을 읽어서 '시간 순서(t)'대로 정렬하여 합칩니다.
    """
    all_data = []
    
    for i, log_dir in enumerate(log_dirs):
        # (1) 해당 시드 폴더 내의 모든 monitor 파일 가져오기 (0~9)
        monitor_files = glob.glob(os.path.join(log_dir, "monitor", "*.csv"))
        
        seed_data = []
        for file in monitor_files:
            try:
                # monitor.csv 읽기 (첫 2줄 헤더 제외)
                df = pd.read_csv(file, skiprows=1)
                if not df.empty:
                    seed_data.append(df)
            except Exception as e:
                print(f"{file} 읽기 실패하였습니다. 에러는 다음과 같습니다. {e}")
        
        if not seed_data: continue

        """ 
        멀티 프로레싱 --n_envs를 가정하고 주석 설명
        --n_envs를 따로 설정하지 않았어도 문제없이 코드 실행됨
        """
        # (2) N개 파일의 데이터를 하나로 합침 
        # (N: --n_envs 설정과 몇 번 이어서 모델을 학습했는지에 따라 결정됨)
        merged_df = pd.concat(seed_data, ignore_index=True)

        # (3) 't' (경과 시간) 기준으로 오름차순 정렬
        # 이렇게 하면 CPU 0번과 9번이 동시에 수행한 일들이 시간 순서대로 섞인다.
        merged_df = merged_df.sort_values(by='t')

        # (4) Total Steps 재계산 (누적 합)
        # 시간 순서대로 정렬된 상태에서 에피소드 길이를 더하면 정확한 Total Steps가 됨
        merged_df['total_steps'] = merged_df['l'].cumsum()
        merged_df['total_steps'] = (merged_df['total_steps'] // 1000) * 1000

        # (5) 데이터 정리
        merged_df = merged_df[['r', 'total_steps']]
        merged_df['seed'] = f"seed_{i}"
        merged_df['experiment'] = exp_name
        
        # (6) 스무딩 (그래프 예쁘게)
        # 데이터가 많아졌으므로 window를 넉넉하게 잡는다.
        merged_df['smooth_reward'] = merged_df['r'].rolling(window=1000).mean()
        
        all_data.append(merged_df)

    if not all_data: return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


# 2. 체크포인트 승률 평가 함수
#    -> Win Rate 그래프용 (시간 오래 걸림)
def evaluate_checkpoints(exp_name, log_dirs):
    """
    여러 시드의 체크포인트(= step 별로 학습된 모델)를 모두 평가하여 승률(Win Rate) 데이터 만든다.
    """
    all_results = []
    env = DogfightEnv(difficulty=EVAL_CONFIG["difficulty"]) # 평가용 환경 생성
    

    for i, log_dir in enumerate(log_dirs):
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        model_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
        
        # Checkpoints 파일명에서 스텝 수 추출 및 정렬
        sorted_files = []
        for f in model_files:
            match = re.search(r"_(\d+)_steps", os.path.basename(f))
            if match:
                sorted_files.append((int(match.group(1)), f))
            
        sorted_files.sort(key=lambda x: x[0])
        
        print(f"[{exp_name} | Seed {i+1}] 모델 {len(sorted_files)}개 평가 시작...")

        for step, model_path in sorted_files:
            # 알고리즘 자동 감지, PPO와 SAC만 있다고 가정
            try:
                if "ppo" in log_dir.lower():
                    model = PPO.load(model_path, env=env)
                else:
                    model = SAC.load(model_path, env=env)
            except:
                print(f"모델 로드 실패: {model_path}")
                continue

            wins = 0
            
            # --- 평가 루프 (고정 시드) ---
            for ep in range(EVAL_CONFIG["episodes"]):
                """
                각 체크포인트의 모델을 평가할 때, 
                같은 episode에서는 같은 seed를 가지고 평가 --> 공정한 평가를 위한 작업
                ex) 즉, 20000step 모델의 10번째 episode와 1000000step 모델의 10번째 episode의 seed가 같음
                """
                current_seed = EVAL_CONFIG["base_seed"] + ep        
                np.random.seed(current_seed)
                obs, _ = env.reset(seed=current_seed)
                env.max_steps = 500

                # [난이도 변경시 해당 부분 주석 해제]
                #  기본 설정 값 --> EVAL_LV.2
                # env.agent.turn_rate = EVAL_CONFIG["agent_turn_rate"]
                # env.enemy.turn_rate = EVAL_CONFIG["enemy_turn_rate"]
                # env.enemy.speed = EVAL_CONFIG["speed"]
                # env.enemy.speed = EVAL_CONFIG["speed"]
                # env.agent.LOCK_ANGLE = np.deg2rad(EVAL_CONFIG["angle"])
                # env.enemy. LOCK_ANGLE = np.deg2rad(EVAL_CONFIG["angle"])
                # env.agent.health = EVAL_CONFIG["agent_hp"]
                # env.enemy.health = EVAL_CONFIG["enemy_hp"]
                # env.prob_start = EVAL_CONFIG["prob_start"]
                
                obs = env._get_observation()
                
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    if terminated and env.enemy.health <= 0:
                        wins += 1
            
            win_rate = wins / EVAL_CONFIG["episodes"]
            print(f"  Step {step}: Win Rate {win_rate:.2f}")
            
            all_results.append({
                "total_steps": step,
                "win_rate": win_rate,
                "seed": f"seed_{i+1}",
                "experiment": exp_name
            })

    return pd.DataFrame(all_results)


# 3. 그래프 그리기 함수 (Seaborn)
def plot_comparison(df, x_col, y_col, title, filename):
    plt.figure(figsize=(10, 6))
    
    # seaborn의 lineplo을 통해서 seed별 평균(선)과 신뢰구간(그림자)을 그린다.
    sns.lineplot(data=df, x=x_col, y=y_col, hue="experiment", errorbar='sd') 
    
    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel(y_col)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Experiment")
    plt.savefig(filename)
    print(f"그래프 저장 완료: {filename}")
    plt.close()


# 메인 함수 실행 (experiments 내부를 수정해서 Evaluation 하시면 됩니다.)
if __name__ == "__main__":
    # 비교할 실험 그룹 정의
    # 예시: 폴더 경로를 리스트로 넣어주세요 ("SAC_Hard_128": ["logs/dogfight_sac_128_hard_seed1", ..., ])
    experiments = {
        "SAC_Easy_128": ["logs/dogfight_sac_128_easy_seed1", "logs/dogfight_sac_128_easy_seed2", "logs/dogfight_sac_128_easy_seed3"],
        "SAC_Hard_128": ["logs/dogfight_sac_128_hard_seed1", "logs/dogfight_sac_128_hard_seed2", "logs/dogfight_sac_128_hard_seed3"],
        "SAC_Curriculum_V1_128": ["logs/dogfight_sac_128_curriculum_v1_seed1", "logs/dogfight_sac_128_curriculum_v1_seed2", "logs/dogfight_sac_128_curriculum_v1_seed3"],
        "SAC_Curriculum_V2_128": ["logs/dogfight_sac_128_curriculum_v2_seed1", "logs/dogfight_sac_128_curriculum_v2_seed2", "logs/dogfight_sac_128_curriculum_v2_seed3"],
        "SAC_Random_128": ["logs/dogfight_sac_128_random_seed1", "logs/dogfight_sac_128_random_seed2", "logs/dogfight_sac_128_random_seed3"],
    }

    #  데이터 수집 모드 선택 
    DO_REWARD_ANALYSIS = True   # Monitor.csv 분석, 기존에 학습시키면서 저장해둔 log 불러옴 (커리큘럼 학습 평가에서는 애매함, 리워드 기준 같으면 가능)
    DO_WINRATE_ANALYSIS = True  # Checkpoint 승률 평가 (episode 개수 설정 가능)


    # (1) 학습 곡선 (Reward) 분석 및 그리기
    if DO_REWARD_ANALYSIS:
        print("\n=== Reward 데이터 로드 중... ===")
        reward_dfs = []
        for exp_name, log_dirs in experiments.items():
            df = load_training_data(exp_name, log_dirs)
            if not df.empty:
                reward_dfs.append(df)
        
        if reward_dfs:
            final_reward_df = pd.concat(reward_dfs, ignore_index=True)
            plot_comparison(final_reward_df, "total_steps", "smooth_reward", 
                          "Learning Curve (Reward)", "graph_reward_comparison.png")

    # (2) 승률 (Win Rate) 평가 및 그리기 
    # - Win은 적을 격추 시킨 경우만 해당되게끔 하였습니다. (적이 맵 밖으로 나간 건 승리로 count하지 않았습니다.)
    if DO_WINRATE_ANALYSIS:
        print("\n=== Checkpoint 승률 평가 시작 (episode 설정에 따라 시간 차이 존재) ===")
        win_dfs = []
        for exp_name, log_dirs in experiments.items():
            # 이미 평가된 csv 파일이 있으면 로드, 없으면 계산
            csv_path = f"results_{exp_name}.csv"
            
            if os.path.exists(csv_path):
                print(f"{csv_path} 로드")
                df = pd.read_csv(csv_path)
            else:
                df = evaluate_checkpoints(exp_name, log_dirs)
                df.to_csv(csv_path, index=False) # 저장해두기 (나중에 재사용)
            
            win_dfs.append(df)
            
        if win_dfs:
            final_win_df = pd.concat(win_dfs, ignore_index=True)
            plot_comparison(final_win_df, "total_steps", "win_rate", 
                          "Evaluation Win Rate (Mode)", "graph_winrate_comparison.png")
