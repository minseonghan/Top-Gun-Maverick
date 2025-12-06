import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import matplotlib.pyplot as plt

# ***************************
# 1. 전투기 물리 엔진 클래스
# ***************************
class Aircraft:
    def __init__(self, x, y, heading, speed=2.0, turn_rate=0.1, health=10.0):
        self.x = x                  # x 좌표
        self.y = y                  # y 좌표
        self.heading = heading      # 전투기의 방향 (~pi, pi)
        self.speed = speed          # 전투기의 속력
        self.turn_rate = turn_rate  # 전투기의 선회율(전투기가 방향을 얼마나 빠르게 바꾸는가)
        self.health = health        # Agent의 체력 
        self.max_health = health    # 적의 체력
        self.trajectory = []        # 전투기 이동경로 (화면에 시각화하기 위함, 전투기 방향성 및 경로를 알 수 있음)

    def move(self, action):
        """action: -1.0(좌) ~ 1.0(우)"""
        clipped_action = np.clip(action, -1.0, 1.0)
        self.heading += clipped_action * self.turn_rate
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi # -pi ~ pi 정규화

        self.x += self.speed * np.cos(self.heading)     # x 이동거리
        self.y += self.speed * np.sin(self.heading)     # y 이동거리
        self.trajectory.append((self.x, self.y))        # 이동경로를 시각화하기 위해 저장

    @property
    def position(self):
        return np.array([self.x, self.y])

# *********************************************
# 2. 강화학습 Dogfight(전투기 공중전) 환경 클래스
# *********************************************
class DogfightEnv(gym.Env):
    def __init__(self, difficulty="HARD"):
        super(DogfightEnv, self).__init__()
        
        self.difficulty = difficulty    #학습 난이도 EASY, HARD, Curriculum_V1
        
        # 커리큘럼 진행도 (Curriculum_V1으로 학습하기 위한 변수, 0.0: EASY ~ 1.0: HARD)
        self.curr_progress = 0.0 
        
        # --- 환경 설정 ---
        self.map_size = 200.0
        self.max_steps =500  
        
        # --- 전투 파라미터 ---
        self.LOCK_DIST = 100.0  # Lock-on 거리 (공통)

        # --- Action Space (회전만 제어) ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # --- Observation Space ---
        # [내x, 내y, 내heading, 적x, 적y, 적heading, 적health, 내health]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    # 외부(Callback)에서 진행률을 업데이트해주는 함수
    def set_curriculum_progress(self, progress):
        self.curr_progress = np.clip(progress, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0           # 한 에피소드에서 진행된 step
        self.lock_on_step_count = 0     # Lock_on한 step
        self.bitten_step_count = 0      # 적에게 Lock_on 당한 step



        # 난이도에 따른 Agent, 적의 전투기 물리 엔진 설정값 및 움직임 설정
        # --- 학습 난이도, 평가 난이도 --- 
        # --- 학습 난이도  ---
        # EASY << Curriculum_V1 << Curriculum_V2 << HARD <<<<<<<<<<<<< RANDOM
        if self.difficulty == "HARD":
            my_hp = 10.0        # Agent의 체력
            enemy_hp = 10.0     # 적의 체력
            t_rate = 0.05       # 선회율 (전투기가 방향을 얼마나 빠르게 바꾸는가)
            self.AGENT_LOCK_ANGLE = np.deg2rad(30) # Agent의 Lock-on 각도
            self.ENEMY_LOCK_ANGLE = np.deg2rad(30) # 적의 Lock-on 각도
            
             # 적의 움직임 설정
            rand_val = np.random.rand()
            prob_start = 0.7    # 70% 확률로 적의 뒤에서 시작, 30% 확률로 Agent와 적 모두 랜덤한 위치에서 시작
            if rand_val < 0.4: self.enemy_type = "HYBRID"  # 추격 + 회피 기동
            elif rand_val < 0.7: self.enemy_type = "PURSUIT"  # 추격
            elif rand_val < 0.85: self.enemy_type = "RIGHT"  # 우회전
            else: self.enemy_type = "LEFT"  # 좌회전

        elif self.difficulty == "EASY":
            my_hp = 10.0
            enemy_hp = 10.0
            t_rate = 0.05
            self.AGENT_LOCK_ANGLE = np.deg2rad(35) # Agent의 Lock-on 각도
            self.ENEMY_LOCK_ANGLE = np.deg2rad(30) # 적의 Lock-on 각도

             # 적의 움직임 설정
            rand_val = np.random.rand()
            prob_start = 0.9
            if rand_val < 0.2: self.enemy_type = "HYBRID"  # 추격 + 회피 기동
            if rand_val < 0.5: self.enemy_type = "PURSUIT"  # 추격 
            elif rand_val < 0.75: self.enemy_type = "RIGHT"  # 우회전
            else: self.enemy_type = "LEFT"  # 좌회전

        # Curriculum_V2 - ratio: 0.5 설정 (200만 step 기준)
        elif self.difficulty == "Curriculum_V2":
            # [점진적 커리큘럼 로직 구현]
            r = self.curr_progress # 0.0(Start) ~ 1.0(End)

            my_hp = 10.0
            enemy_hp = 10.0
            t_rate = 0.05
            agent_angle_deg = 30
            self.AGENT_LOCK_ANGLE = np.deg2rad(agent_angle_deg)
            self.ENEMY_LOCK_ANGLE = np.deg2rad(30)
            
             # 적의 움직임 설정
            rand_val = np.random.rand()
            
            # 선형보간 방식으로 EASY보다 어려운 조건(Medium)에서 HARD로 점진적 증가
            prob_start = 0.8 + (0.7 - 0.8) * r
            th_hybrid = (0.3 + (0.4 - 0.3) * r)
            th_pursuit = (0.6 + (0.7 -0.6)* r)
            th_right = (0.8 + (0.85 - 0.8) * r)
            if rand_val < th_hybrid: self.enemy_type = "HYBRID"  # 추격 + 회피 기동
            if rand_val < th_pursuit: self.enemy_type = "PURSUIT"  # 추격 
            elif rand_val < th_right: self.enemy_type = "RIGHT"  # 우회전
            else: self.enemy_type = "LEFT"  # 좌회전   

        # Curriculum_V1 - ratio: 0.5 설정 (200만 step 기준)
        elif self.difficulty == "Curriculum_V1":
            # [점진적 커리큘럼 로직 구현]
            r = self.curr_progress # 0.0(Start) ~ 1.0(End)

            my_hp = 10.0
            enemy_hp = 10.0
            t_rate = 0.05
            agent_angle_deg = 35.0 + (30.0 - 35.0) * r
            self.AGENT_LOCK_ANGLE = np.deg2rad(agent_angle_deg)
            self.ENEMY_LOCK_ANGLE = np.deg2rad(30)

             # 적의 움직임 설정
            rand_val = np.random.rand()
            
            # 선형보간 방식으로 EASY에서 HARD로 점진적 증가
            prob_start = 0.9 + (0.7 - 0.9) * r
            th_hybrid = (0.2 + (0.4 - 0.2) * r)
            th_pursuit = (0.5 + (0.7 -0.5)* r)
            th_right = (0.75 + (0.85 - 0.75) * r)
            if rand_val < th_hybrid: self.enemy_type = "HYBRID"  # 추격 + 회피 기동
            if rand_val < th_pursuit: self.enemy_type = "PURSUIT"  # 추격 
            elif rand_val < th_right: self.enemy_type = "RIGHT"  # 우회전
            else: self.enemy_type = "LEFT"  # 좌회전      

        elif self.difficulty == "RANDOM":    
            """ 
            이 모드에서 prob_start를 1로 바꾸더라도 학습되지 않음. 
            't_rate' = 0.1와 오직 HYBRID(추격+회피) 전략만 갖고 있는 것이 매우 강력함
            """ 
            my_hp = 10.0                   
            enemy_hp = 10.0                 
            t_rate = 0.1                           
            self.AGENT_LOCK_ANGLE = np.deg2rad(30) # Agent의 Lock-on 각도
            self.ENEMY_LOCK_ANGLE = np.deg2rad(30) # 적의 Lock-on 각도
            
             # 적의 움직임 설정
            rand_val = np.random.rand()
            prob_start = 0              # Agent와 적의 위치가 랜덤하게 시작 (0에 가까울수록 랜덤, 1에 가까울수록 적의 뒤에서 시작)
            self.enemy_type = "HYBRID"  # 추격 + 회피 기동     

       
        # --- 평가 기준 ----
        #  EVAL_LV.1 << EVAL_LV.2
        elif self.difficulty == "EVAL_LV.2":
            my_hp = 10.0
            enemy_hp = 10.0
            t_rate = 0.05
            self.AGENT_LOCK_ANGLE = np.deg2rad(30) # Agent의 Lock-on 각도
            self.ENEMY_LOCK_ANGLE = np.deg2rad(25) # Lock-on 각도

             # 적의 움직임 설정
            rand_val = np.random.rand()
            prob_start = 0.5
            if rand_val < 0.4: self.enemy_type = "HYBRID"  # 추격 + 회피        
            elif rand_val < 0.6: self.enemy_type = "PURSUIT"  # 추격
            elif rand_val < 0.8: self.enemy_type = "RIGHT"  # 우회전
            else: self.enemy_type = "LEFT"  # 좌회전

        elif self.difficulty == "EVAL_LV.1":
            my_hp = 10.0
            enemy_hp = 8.0
            t_rate = 0.04
            self.AGENT_LOCK_ANGLE = np.deg2rad(35) # Agent의 Lock-on 각도
            self.ENEMY_LOCK_ANGLE = np.deg2rad(25) #  Lock-on 각도

             # 적의 움직임 설정
            rand_val = np.random.rand()
            prob_start = 0.7
            if rand_val < 0.3: self.enemy_type = "HYBRID"  # 추격 + 회피        
            elif rand_val < 0.6: self.enemy_type = "PURSUIT"  # 추격
            elif rand_val < 0.8: self.enemy_type = "RIGHT"  # 우회전
            else: self.enemy_type = "LEFT"  # 좌회전        


        
        # [학습 전략] prob_start 확률로 공격 유리 위치(꼬리)에서 시작
        #  HARD: 0.7, EASY = 0.9, EVAL_LV.2 = 0.7, EVAL_LV.2 = 0.5
        if np.random.rand() < prob_start:
            # --- control-zone start (replace your existing chained-rand block with this) ---
            # 적을 맵 중앙에 생성, heading을 랜덤으로 줘서 다양성 확보
            self.enemy = Aircraft(x=0.0, y=0.0, heading=np.random.uniform(-np.pi, np.pi),
                                speed=1.5, turn_rate=t_rate, health=enemy_hp)

            # 나를 적의 뒤쪽(꼬리) 근처에 생성 (Control Zone)
            start_dist = np.random.uniform(50.0, 100.0)     # 꼬리에서의 거리
            tail_angle = (self.enemy.heading + np.pi)       # 적의 꼬리 방향
            # 약간의 편차를 주기 위한 각도 노이즈
            heading_noise = np.random.uniform(-0.1, 0.1)

            # 위치: 적 위치(0,0) 기준, 꼬리 방향으로 start_dist만큼 떨어진 지점 + 작은 횡방향 오프셋
            offset_along_tail_x = start_dist * math.cos(tail_angle)
            offset_along_tail_y = start_dist * math.sin(tail_angle)
            lateral_offset = np.random.uniform(-10.0, 10.0)

            # lateral_offset을 꼬리 방향과 직교하는 벡터로 처리
            # 꼬리 방향의 직교(왼쪽) 벡터 = tail_angle + pi/2
            orth_x = math.cos(tail_angle + math.pi/2)
            orth_y = math.sin(tail_angle + math.pi/2)

            agent_x = self.enemy.x + offset_along_tail_x + lateral_offset * orth_x
            agent_y = self.enemy.y + offset_along_tail_y + lateral_offset * orth_y
            agent_heading = (self.enemy.heading + heading_noise)

            self.agent = Aircraft(
                x=agent_x,
                y=agent_y,
                heading=agent_heading,
                health=my_hp
            )


        else:
            # 3. 일반 모드: 랜덤 위치 
            self.agent = Aircraft(
                x=np.random.uniform(-75, 75), y=np.random.uniform(-75, 75),
                heading=np.random.uniform(-np.pi, np.pi), health=my_hp
            )
            self.enemy = Aircraft(
                x=np.random.uniform(-75, 75), y=np.random.uniform(-75, 75),
                heading=np.random.uniform(-np.pi, np.pi), speed=1.5, turn_rate=t_rate, health=my_hp
            )
        
        return self._get_observation(), {}


    # --- 매 step마다 해당 함수로 보상, 종료조건 판단 ---
    def step(self, action):
        self.current_step += 1
        
        # 1. 움직임
        self.agent.move(action[0])
        enemy_action = self._get_rule_based_enemy_action()
        self.enemy.move(enemy_action)
        
        # 2. 상태 계산 (거리 및 각도)
        dist = np.linalg.norm(self.agent.position - self.enemy.position)
        
        # ATA: Antenna Train Angle (내 기축선과 적 사이 각도, 보고서 7p그림 참고)
        agent_ata = self._get_aim_angle(self.agent, self.enemy)
        enemy_ata = self._get_aim_angle(self.enemy, self.agent)
        
        # AA: Aspect Angle (적 꼬리와 나 사이의 각도, 보고서 7p그림 참고)) - Control Zone 체크용
        agent_aa = self._get_aspect_angle(self.enemy, self.agent)
        
        # 3. Lock-on 및 피격 판정
        # 내가 공격: 사거리 내 + 조준각(ATA) 내
        is_agent_locking = (dist < self.LOCK_DIST) and (agent_ata < self.AGENT_LOCK_ANGLE)
        
        # 내가 피격: 사거리 내 + 적이 나를 조준(ATA)
        is_being_bitten = (dist < self.LOCK_DIST) and (enemy_ata < self.ENEMY_LOCK_ANGLE)

        # 4. 체력(Health) 처리 [cite: 164, 357]
        damage = 0.1 # 프레임당 데미지

        # Agent가 적을 Lock-on 한 경우
        if is_agent_locking:
            self.enemy.health -= damage
            self.lock_on_step_count += 1
        else:
            self.lock_on_step_count = 0
        
        # 적이 Agent를 Lock-on 한 경우
        if is_being_bitten:
            self.agent.health -= damage
            self.bitten_step_count += 1
        else:
            self.bitten_step_count = 0

        # 5. 보상 계산
        reward = self._calculate_reward(dist, agent_ata, agent_aa, is_agent_locking, is_being_bitten)
        
        # 6. 종료 조건
        terminated = self._check_termination(dist)
        truncated = self.current_step >= self.max_steps
        
        info = {
            "agent_health": self.agent.health,
            "enemy_health": self.enemy.health,
            "lock_steps": self.lock_on_step_count
        }
        
        return self._get_observation(), reward, terminated, truncated, info


    # --- 보상 함수 ---
    def _calculate_reward(self, dist, ata, aa, is_agent_locking, is_being_bitten):
        """
        R = R_pos + R_close + R_gun + R_outcome + R_safety
        """
        reward = 0.0
        
        # --- (1) R_position (Control Zone 유지) ---
        # 적의 꼬리(AA=0)를 물고, 적을 바라보고(ATA=0) 있을수록 점수
        k_pos = 1.0
        norm_ata = 1.0 - (ata / np.pi) # 0~1
        norm_aa = 1.0 - (aa / np.pi)   # 0~1
        # 두 조건이 다 맞아야 큰 점수 (곱하기 연산)
        reward += k_pos * (norm_ata * norm_aa)     # k_pos가 적으면, 뒤에서 꼬리 잡는 것의 중요성 인식 X, 또한 그러다 보니 적을 격추 시키는 경우가 없어서 agent가 적의 격추 성공 맛을 못 봄

        # --- (2) R_closure (접근 보상) ---
        # 공격 각이 나오고(ATA < 60도) 거리가 멀면 접근 보상
        if ata < np.deg2rad(60):
            if dist > 50.0:
                reward += 0.05 # 접근 유도
            elif dist < 30.0:
                reward -= 0.1   # 너무 가까우면(충돌 위험) 패널티
        if dist < 10.0:         # 거리 가까울수록 더 큰 패널티
            reward -= (10 - dist) * 0.02

        # --- (3) R_gunsnap (공격 보상) ---
        # 락온 중이면 지속 보상
        if is_agent_locking:
            reward += 1.0 #  보상       # 공격 보상이 1이고 충돌 패널티가 -50 일때는, 충돌하면서 공격하는 것에 계속 수렴했음
            
        # [승리] 적 체력 0 (격추)
        if self.enemy.health <= 0:
            reward += 50.0 # 결정적 보상

        # --- (4) R_safety (페널티) ---
        # 맵 이탈
        if abs(self.agent.x) > self.map_size or abs(self.agent.y) > self.map_size:
            reward -= 110.0     # 공격 당하여 격추당하는 패널티(-100)보다 더 크기가 커야함. 그러지 않으면 쫓기는 순간 바로 맵을 이탈하려함
        
        # 충돌
        if dist < 5.0:
            reward -= 80.0     # 공격 보상이 1이고 충돌 패널티가 -50 일때는, 충돌하면서 공격하는 것에 계속 수렴했음
            
        if is_being_bitten:
            reward -= 0.5
            
        # [패배] 내 체력 0 (격추 당함) 
        if self.agent.health <= 0:
            reward -= 50.0          # 격추와 맵 이탈과의 패널티가 같을 때는 싸우지 않고 맵 이탈하는 경향이 매우 컸음. 왜냐하면 싸우다가 격추 당하면 being_bitten + 격추 패털티의 크기가 맵이탈 크기보다 컸기 때문

        return reward*0.1

   

    def _check_termination(self, dist):
        # 1. 체력 0 (승패)
        if self.agent.health <= 0 or self.enemy.health <= 0:
            return True
            
        # 2. 맵 이탈
        if abs(self.agent.x) > self.map_size or abs(self.agent.y) > self.map_size \
            or abs(self.enemy.x) > self.map_size or abs(self.enemy.y) > self.map_size:
            return True

        # 3. 충돌
        if dist < 5.0:
            return True
            
        return False

    def _get_observation(self):
        # [내x, 내y, 내h, 적x, 적y, 적h, 적HP, 내HP]
        # HP 정보를 주어야 에이전트가 "내가 죽어가니 도망가야겠다"는 판단 가능
        return np.array([
            self.agent.x, self.agent.y, self.agent.heading,
            self.enemy.x, self.enemy.y, self.enemy.heading,
            self.enemy.health, self.agent.health
        ], dtype=np.float32)

    # --- 적의 전략 ---
    def _get_rule_based_enemy_action(self):
        # 1. RIGHT 행동 (무조건 오른쪽으로 돌되, 회전 반경이 계속 변함)
        if self.enemy_type == "RIGHT":
            # 기본 0.4에 -0.3 ~ +0.1의 변화를 줌
            # 결과: 0.1 ~ 0.5 사이의 값이 나옴 -> "항상 우회전" 보장됨
            noise = np.random.uniform(-0.3, 0.1)
            action = 0.4 + noise
            return float(np.clip(action, -1.0, 1.0))
        
        # 2. LEFT 행동 (무조건 왼쪽으로 돌되, 회전 반경이 계속 변함)
        elif self.enemy_type == "LEFT":
            # 기본 -0.4에 -0.1 ~ +0.3의 변화를 줌
            # 결과: -0.5 ~ -0.1 사이의 값이 나옴 -> "항상 좌회전" 보장됨
            noise = np.random.uniform(-0.1, 0.3)
            action = -0.4 + noise
            return float(np.clip(action, -1.0, 1.0))
        
        # 3. PURSUIT 행동 
        elif self.enemy_type == "PURSUIT":        
        # Pure Pursuit (단순 추격)
            target_angle = math.atan2(self.agent.y - self.enemy.y, self.agent.x - self.enemy.x)
            heading_diff = target_angle - self.enemy.heading
            heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi     # -pi ~ pi 정규화
            action = heading_diff * 1.0 / self.enemy.turn_rate      # gain = 1.0 고정
            return float(np.clip(action, -1.0, 1.0))
        
        # 평소엔 쫓아가다가, 꼬리를 잡히면 회피 기동
        elif self.enemy_type == "HYBRID":
            dist = np.linalg.norm(self.agent.position - self.enemy.position)
            enemy_aa = self._get_aspect_angle(self.enemy, self.agent) # 적 기준 AA
            
            # # 위험 조건: 거리가 가깝고(70이내), 적이 내 뒤쪽(AA < 20도)에 있을 때
            # is_threatened = (dist < dist_treatened) and (enemy_aa < np.deg2rad(aa_treatened))
            is_threatened = (dist < 50) and (enemy_aa < np.deg2rad(15))

            if is_threatened:
                return self._calculate_evade_action() # 회피 기동 함수 호출
            else:
                # 안전하면 추격 (PURSUIT 로직 복사)
                target_angle = math.atan2(self.agent.y - self.enemy.y, self.agent.x - self.enemy.x)
                heading_diff = target_angle - self.enemy.heading
                heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
                action = heading_diff * 1.0 / self.enemy.turn_rate  # gain
                return float(np.clip(action, -1.0, 1.0))
        
        return 0.0

    # --- 회피 기동 계산 함수 분리 --_
    def _calculate_evade_action(self):
        """ 회피 기동을 실행하는 주체를 '나'라고 가정하고 주석 처리했습니다."""
        # 1. 적이 있는 방향 계산
        angle_to_agent = math.atan2(self.agent.y - self.enemy.y, self.agent.x - self.enemy.x)
        
        # 2. 내 진행 방향과의 차이
        heading_diff = angle_to_agent - self.enemy.heading
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
        
        if abs(heading_diff) < 1e-4:    # 차이가 0인 경우 직진하는 오류가 있어서 노이즈 추가
            heading_diff = 1e-4

        # 회피 기동
        # 적 방향으로 회전하여 위험 조건에서 벗어남
        noise = np.random.uniform(-0.1, 0.1)
        magnitude = 0.2 + noise
        evade_action = magnitude * np.sign(heading_diff)
        return float(evade_action)

    # --- AA, ATA를 구하는 함수 (보고서 7p 참고) ---
    def _get_aim_angle(self, entity1, entity2):
        """ATA: entity1이 entity2를 바라보는 각도 차이 (0 ~ pi)"""
        dx = entity2.x - entity1.x
        dy = entity2.y - entity1.y
        target_angle = math.atan2(dy, dx)
        diff = abs((target_angle - entity1.heading + np.pi) % (2*np.pi) - np.pi)
        return diff
    
    def _get_aspect_angle(self, entity_target, entity_chaser):
        """AA: Target의 꼬리 기준 Chaser의 각도 (0 ~ pi)"""
        dx = entity_chaser.x - entity_target.x
        dy = entity_chaser.y - entity_target.y
        angle_to_chaser = math.atan2(dy, dx)
        target_tail = entity_target.heading + np.pi
        diff = abs((angle_to_chaser - target_tail + np.pi) % (2*np.pi) - np.pi)
        return diff

    # 시각화 함수
    def render(self):
        plt.clf()
        plt.xlim(-self.map_size, self.map_size)
        plt.ylim(-self.map_size, self.map_size)
        
        # Agent (Blue)
        ax, ay = zip(*self.agent.trajectory) if self.agent.trajectory else ([],[])
        plt.plot(ax, ay, 'b-', alpha=0.3)
        plt.plot(self.agent.x, self.agent.y, 'b^', markersize=10, label=f'Agent HP:{self.agent.health:.1f}')
        
        # 적 (Red)
        ex, ey = zip(*self.enemy.trajectory) if self.enemy.trajectory else ([],[])
        plt.plot(ex, ey, 'r--', alpha=0.3)
        plt.plot(self.enemy.x, self.enemy.y, 'rv', markersize=10, label=f'Enemy HP:{self.enemy.health:.1f}')
        
        plt.legend()
        plt.title(f"Step: {self.current_step}")
        plt.pause(0.001)
