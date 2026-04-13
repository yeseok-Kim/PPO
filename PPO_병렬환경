'''
유의사항
1. 이 코드가 포함되어 있는 파일을 직접 실행시켜야 함.
   (이 파일을 import해서 사용하면 실행 안됨.)
2. 모든 코드는 if __name__ == "__main__": 조건문 아래에서 작성해야 함.
'''
# 아래 조건문은 병렬환경 실행 시 오류를 방지하기 위함. 이 코드가 없으면 병렬환경 갯수만큼 순차 실행 (!= 병렬환경 학습)
if __name__ == "__main__": # 해당 코드가 구현된 파일이 직접 실행됐을 때만 실행
    
    import warnings
    # FutureWarning을 무시
    warnings.simplefilter(action='ignore', category=FutureWarning)

    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True

    # -------------------------------
    # 1. 데이터 정의
    # -------------------------------
    import torch
    import numpy as np

    fixed_net_demand = torch.tensor([
        [ 0,  0,  0],    # depot (0)
        [-3, -2,  2],    # node 1
        [-2,  4,  0],    # node 2
        [ 2, -3,  2],    # node 3
        [ 4, -2, -2],    # node 4
        [ 0,  1, -1],    # node 5
        [-1,  2, -1]     # node 6 (추가)
    ], dtype=torch.int32)

    dist_matrix = np.array([
        [0.0, 1.2, 2.5, 1.8, 2.0, 1.9, 2.2],
        [1.2, 0.0, 1.3, 1.5, 1.7, 2.1, 1.9],
        [2.5, 1.3, 0.0, 1.1, 0.9, 1.8, 2.3],
        [1.8, 1.5, 1.1, 0.0, 1.4, 2.0, 2.2],
        [2.0, 1.7, 0.9, 1.4, 0.0, 1.3, 1.5],
        [1.9, 2.1, 1.8, 2.0, 1.3, 0.0, 1.2],
        [2.2, 1.9, 2.3, 2.2, 1.5, 1.2, 0.0]
    ])
    item_to_group = [0, 0, 1]
    group_cap = {0: 4, 1: 2}

    # -------------------------------
    # 2. 환경 클래스 정의
    # -------------------------------
    import gymnasium as gym
    from gymnasium.spaces import MultiDiscrete, Box
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from sklearn.manifold import MDS
    import matplotlib.pyplot as plt

    class VRPEnv(gym.Env):
        def __init__(self, fixed_net_demand, dist_matrix, item_to_group, group_cap):
            super().__init__()

            # 입력 데이터를 PyTorch 텐서로 변환
            self.fixed_net_demand = fixed_net_demand
            self.dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32)
            self.item_to_group = torch.tensor(item_to_group, dtype=torch.int64)

            # 그룹 용량을 텐서로 변환
            self.num_groups = len(set(item_to_group))
            self.group_cap_tensor = torch.zeros(self.num_groups, dtype=torch.int32)

            for g, cap in group_cap.items():
                self.group_cap_tensor[g] = cap
            self.group_cap = group_cap
            self.num_nodes = fixed_net_demand.shape[0]
            self.num_items = fixed_net_demand.shape[1]
            self.max_steps = 3000
            max_amount = max(group_cap.values())
            self.action_space = MultiDiscrete([self.num_nodes, self.num_items, 2, max_amount + 1])

            # 그룹별 아이템 마스크 생성
            self.group_masks = []

            for g in range(self.num_groups):
                mask = (self.item_to_group == g)
                self.group_masks.append(mask)

            self.reset()
            obs_low = []
            obs_high = []
            min_demand = torch.min(fixed_net_demand).item()
            max_demand = torch.max(fixed_net_demand).item()
            min_dist = float(np.min(dist_matrix))
            max_dist = float(np.max(dist_matrix))

            for _ in range(self.num_nodes):
                # 수요
                obs_low.extend([min_demand] * self.num_items)
                obs_high.extend([max_demand] * self.num_items)

                # 현재 위치 (one-hot)
                obs_low.append(0.0)
                obs_high.append(1.0)

                # 거리
                obs_low.append(min_dist)
                obs_high.append(max_dist)

                # 도달 가능 여부
                obs_low.append(0.0)
                obs_high.append(1.0)

            # 차량 적재량
            max_group_cap = max(group_cap.values())
            obs_low.extend([0] * self.num_items)
            obs_high.extend([max_group_cap] * self.num_items)

            # 그룹 잔여 용량
            for g in sorted(group_cap.keys()):
                obs_low.append(0.0)
                obs_high.append(float(group_cap[g]))

            # 관측공간 정의
            self.observation_space = Box(
                low=np.array(obs_low, dtype=np.float32),
                high=np.array(obs_high, dtype=np.float32),
                dtype=np.float32
            )

        def _get_total_unbalance(self):
            # 벡터화: 모든 노드(depot 제외)의 절대값 수요 합 계산
            return torch.sum(torch.abs(self.net_demand[1:])).item()
        
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.net_demand = self.fixed_net_demand.clone()
            self.vehicle_capacity = torch.zeros(self.num_items, dtype=torch.int32)
            self.vehicle_pos = 0
            self.step_count = 0
            return self._get_obs(), {}
        
        def _get_obs(self):
            # 차량 위치 원-핫 인코딩
            vehicle_pos_onehot = torch.zeros(self.num_nodes, dtype=torch.float32)
            vehicle_pos_onehot[self.vehicle_pos] = 1.0

            # 모든 노드까지의 거리
            distances = self.dist_matrix[self.vehicle_pos].clone()

            # 그룹별 사용량 계산 (벡터화)
            group_used = torch.zeros(self.num_groups, dtype=torch.int32)
            for g in range(self.num_groups):
                group_mask = self.group_masks[g]
                group_used[g] = torch.sum(self.vehicle_capacity[group_mask])

            # 그룹별 잔여 용량
            group_remaining = self.group_cap_tensor - group_used

            # feasible 계산 (노드별)
            feasible = torch.zeros(self.num_nodes, dtype=torch.float32)
            feasible[0] = 1.0  # depot는 항상 도달 가능

            for n in range(1, self.num_nodes):
                for i in range(self.num_items):
                    net_value = self.net_demand[n, i].item()
                    group_id = self.item_to_group[i].item()
                    # 픽업 가능 조건: 양수 수요 + 그룹에 여유 공간 있음
                    can_pickup = (net_value > 0) and (group_remaining[group_id] >= 1)
                    # 배송 가능 조건: 음수 수요 + 차량에 해당 아이템 있음
                    can_deliver = (net_value < 0) and (self.vehicle_capacity[i] >= 1)

                    if can_pickup or can_deliver:
                        feasible[n] = 1.0
                        break

            # 관찰 벡터 구성
            obs = []

            # 각 노드에 대한 관찰 값 추가
            for i in range(self.num_nodes):
                # 노드의 아이템별 수요
                node_demand = self.net_demand[i].float().tolist()

                # 노드에 차량이 있는지 여부
                at_node = vehicle_pos_onehot[i].item()

                # 현재 위치에서 노드까지의 거리
                node_dist = distances[i].item()

                # 노드가 도달 가능한지 여부
                node_feasible = feasible[i].item()

                # 노드 관찰 벡터 구성
                node_obs = node_demand + [at_node, node_dist, node_feasible]
                obs.extend(node_obs)

            # 차량 적재 상태
            obs.extend(self.vehicle_capacity.float().tolist())

            # 그룹별 잔여 적재 가능량
            for g in sorted(self.group_cap.keys()):
                obs.append(float(group_remaining[g].item()))
            return np.array(obs, dtype=np.float32)
        
        def step(self, action):
            prev_unbalance = self._get_total_unbalance()
            node, item, task_type, amount = map(int, action.tolist())
            dist = self.dist_matrix[self.vehicle_pos, node].item()
            self.vehicle_pos = node
            self.step_count += 1
            net = self.net_demand[node, item].item()
            cap = self.vehicle_capacity[item].item()
            amt = int(amount)

            # 그룹 사용량 계산 (벡터화)
            group_id = self.item_to_group[item].item()
            group_mask = self.group_masks[group_id]
            group_used = torch.sum(self.vehicle_capacity[group_mask]).item()
            group_limit = self.group_cap[group_id]
            pickup = 0
            delivery = 0
            reward = -dist - 0.5

            if node != 0:
                if task_type == 0:  # 픽업
                    if net >= amt and (group_used + amt <= group_limit):
                        pickup = amt
                        self.net_demand[node, item] -= amt
                        self.vehicle_capacity[item] += amt
                    else:
                        reward -= 3.0  # 잘못된 픽업

                elif task_type == 1:  # 배송
                    if -net >= amt and cap >= amt:
                        delivery = amt
                        self.net_demand[node, item] += amt
                        self.vehicle_capacity[item] -= amt
                    else:
                        reward -= 3.0  # 잘못된 배송

            new_unbalance = self._get_total_unbalance()
            reward += (prev_unbalance - new_unbalance) * 3

            # 종료 조건 체크 (모든 노드의 불균형이 해소되고 차량이 depot에 있는 경우)
            terminated = bool(torch.all(self.net_demand[1:] == 0)) and self.vehicle_pos == 0
            truncated = self.step_count >= self.max_steps

            if terminated:
                reward += 8000.0

            return self._get_obs(), reward, terminated, truncated, {
                "pickup": pickup,
                "delivery": delivery,
                "cap": self.vehicle_capacity.tolist(),
                "action": [node, item, task_type, amt],
                "dist": dist
            }

    # ---------------
    # 2. 병렬환경 구성
    # ---------------
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor

    # 환경을 생성하는 함수 선언
    def make_env():
        def _init():
            train_env = VRPEnv(fixed_net_demand, dist_matrix, item_to_group, group_cap)
            # Monitor을 이용하여 학습 때마다 ep_len_mean, ep_rew_mean을 출력
            return Monitor(train_env)
        return _init

    # range 내 숫자 = 사용할 cpu 코어 갯수
    # 사용하고자 하는 코어 갯수만큼 환경을 생성하여 병렬 학습
    train_env = SubprocVecEnv([make_env() for _ in range(4)])

    # -------------------------------
    # 3. 병렬환경에서 학습
    # -------------------------------
    from stable_baselines3.common.callbacks import EvalCallback

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs={"net_arch": [512, 512, 512]},
        verbose=1,
        n_epochs=20,
        device = 'cpu'
    )
    
    eval_callback = EvalCallback(
    train_env,
    best_model_save_path="./ppo_vrp_model",  # 모델 저장 폴더
    log_path="./logs",                   # 로그 저장 경로
    eval_freq=10000,                     # 평가 주기 (스텝 수 기준)
    deterministic=True,
    render=False
    )

    model.learn(total_timesteps=100000, 
            callback=eval_callback
            )
    
    # 병렬환경에서 학습된 모델을 저장
    model.save("ppo_vrp_model/vrp_info")
    
    # -------------------------------
    # 4. 평가 및 시각화
    # -------------------------------

    # 모델 평가는 단일환경에서 진행
    env = VRPEnv(fixed_net_demand, dist_matrix, item_to_group, group_cap)
    env = Monitor(env)

    obs, _ = env.reset()
    done = False
    route = []
    total_reward = 0.0
    total_dist = 0.0

    # 위에서 저장한 모델(병렬환경에서 학습된 모델)을 불러옴
    model = PPO.load("ppo_vrp_model/vrp_info.zip", device="cpu")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward = reward
        total_reward += reward
        total_dist += info["dist"]
        route.append((info["action"], info["delivery"], info["pickup"], info["cap"], info["dist"], reward))

    print("\n상위 분류 기반 공유 용량 복합 행동 경로")

    for i, (a, d, p, cap, dist, reward) in enumerate(route):
        print(f"Step {i+1}: 노드 {a[0]}, 품목 {a[1]}, 작업 {'배송' if a[2]==1 else '픽업'}, 수량 {a[3]} → 배송 {d}, 픽업 {p}, 적재 {cap}, 이동거리 {dist:.2f}, 보상 {reward:.2f}")
    print(f"\n총 이동 거리: {total_dist:.2f}")
    print(f"누적 보상: {total_reward:.2f}")

    # 시각화
    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(dist_matrix)
    node_locs = {i: tuple(coords[i]) for i in range(len(coords))}
    visited_nodes = [step[0][0] for step in route]

    plt.figure(figsize=(8, 6))
    plt.title("방문 경로 및 순서 방향 시각화")

    for i, (x, y) in node_locs.items():
        plt.scatter(x, y, c="red" if i == 0 else "blue", s=250, zorder=2)
        plt.text(x + 0.05, y + 0.05, f"Node {i}", fontsize=10, zorder=3)

    for i in range(len(visited_nodes) - 1):
        src = node_locs[visited_nodes[i]]
        dst = node_locs[visited_nodes[i + 1]]
        dx, dy = dst[0] - src[0], dst[1] - src[1]
        plt.arrow(src[0], src[1], dx * 0.9, dy * 0.9,
                  head_width=0.05, head_length=0.1, fc='green', ec='green', alpha=0.8, length_includes_head=True, zorder=1)
        mid_x = (src[0] + dst[0]) / 2
        mid_y = (src[1] + dst[1]) / 2
        plt.text(mid_x, mid_y, f"{i+1}", fontsize=9, color="darkgreen", zorder=4)

    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
