import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box

# -------------------------------
# 2. 신경망 파라미터 정의(문제 규모에 따라 수정해야 할 수 있음)
# -------------------------------
net_arch = [512] * 16  # 신경망 아키텍처: 2개의 은닉층, 각 층의 유닛 수는 512개 (경험상, 성능에 결정적인지 불분명)
learning_rate = 1e-4  # 학습률: PPO 학습 시 사용되는 학습률, 0.0001로 설정 (너무 크면 수렴이 확실히 안됨)

epoch = 20  # 에포크 수: PPO 학습 시 각 업데이트마다 20번의 에포크를 수행(에포크를 조정하면 적절한 total_timesteps이 달라짐)





# -------------------------------
# 3. 환경 및 학습 파라미터 정의 (실험 시 조정을 자주해야 할 수 있음)
# -------------------------------

maxstep = 500 # 최대 스텝 수(한 에피소드마다 최대 행동 수, 로그에서 평균 스텝 길이(mean_ep_length)가 이 값에 거의 근접하거나 이 값과 같아지면 이 값을 늘려야함, 에이전트가 답을 얻기까지 탐색이 부족한 것임)
              # 문제 규모가 크고 복잡하면 늘려야함
              # 지금의 데이터(7노드 3품목)에서는 500 스텝이면 충분함, 참고로 수렴시 17~18 까지 수렴함

total_timesteps = 300000  # 총 학습 스텝 수: PPO 학습 시 전체 학습 스텝 수
                          # 너무 작으면 수렴이 충분히 이루어지지 않을 수 있음
                          # 문제 규모가 크면 늘려야함
                          # 너무 크면 발산을 유발할 수 있음(하지만 베스트 모델을 주기적으로 저장하기 때문에 문제는 되지 않음, 시간만 늘어남)
                          # 이 값은 환경에 따라 조정이 필요함(너무 작은 거 보단 차라리 큰게 낫고, 실험하면서 조정하는게 좋음)


# -------------------------------
# 4. 평가 파라미터 정의
# -------------------------------
eval_freq =1000 # 평가 주기: PPO 학습 시 모델을 평가하는 주기, 1000 스텝마다 평가 수행
                # 모델이 학습 도중에 평가 주기마다 평가되며, 이 주기마다 모델의 성능을 평가하고 베스트 모델을 저장함 (모델이 너무 학습이 진행되면 발산하기 때문에 필요함)
                # 값을 너무 늘리면 중간에 좋은 파라미터인 구간을 놓칠 수 있음


model_save_path = './ppo_vrp_model_fixh'  # 학습된 모델을 저장할 경로
model_load_path = "ppo_vrp_model_fixh/best_model.zip"  # 학습된 모델을 저장할 경로


# -------------------------------
# 5. 보상 파라미터 정의
# -------------------------------
dist_panalty = 10.0 # 거리 패널티 계수: 이동 거리에 대한 보상 감소 계수, 정규화 된 총거리가 멀수록 보상이 감소함
step_penalty = 5.0 # 스텝 패널티 계수: 매 스텝마다 일정한 감소 보상, 에이전트가 최대한 간소화된 행동을 하도록 유도하기 위한 보상, 스텝 수가 많을수록 보상이 감소함

progress_reward = 200.0 # 진행 보상 계수: 수요공급 해소율에 따라 보상을 조정하는 계수
complited_reward = 500.0 # 작업 완료 보상 상수: 모든 노드의 순수 수요가 0이 되고 차량이 차고지로 복귀한 경우 주어지는 보상, 작업을 모두 완료했을 때 주어지는 보상

class VRPEnv(gym.Env):
    def __init__(self, fixed_net_demand, dist_matrix, item_to_group, group_cap):
        super().__init__()

        self.fixed_net_demand = fixed_net_demand  
        self.dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32) # 거리 행렬을 텐서로 변환
        self.item_to_group = torch.tensor(item_to_group, dtype=torch.int64) # 품목을 그룹에 매핑하는 텐서로 변환

        self.num_groups = len(set(item_to_group)) # 그룹의 개수: item_to_group에서 고유한 그룹의 개수를 계산
        self.group_cap_tensor = torch.zeros(self.num_groups, dtype=torch.int32) # 그룹 용량을 저장할 텐서 생성
        for g, cap in group_cap.items():    # 그룹 용량을 텐서에 저장
            self.group_cap_tensor[g] = cap

        self.group_cap = group_cap # 그룹 용량을 딕셔너리 형태로 저장
        self.num_nodes = fixed_net_demand.shape[0] # 노드의 개수: 고정된 순수 수요 텐서의 행 개수로 설정
        self.num_items = fixed_net_demand.shape[1] # 품목의 개수: 고정된 순수 수요 텐서의 열 개수로 설정
        self.max_steps = maxstep # 최대 스텝 수: 환경이 종료되기 전까지의 최대 스텝 수로 설정

        self.action_space = MultiDiscrete([ 
            self.num_nodes,
            self.num_items,
            2
        ]) # 행동 공간 설정: 노드 선택, 품목 선택, 작업 유형(픽업/배송)으로 구성 (에이전트는 이 행동공간 내에서 행동을 선택함)

        self.group_masks = [(self.item_to_group == g) for g in range(self.num_groups)] # 그룹별 마스크 생성: 각 그룹에 속하는 품목을 True로 설정하는 마스크 리스트


        # 관측 공간 범위 설정
        obs_low = []    # 관측 공간의 하한값 초기화
        obs_high = []   # 관측 공간의 상한값 초기화
        for _ in range(self.num_nodes):  # 고정된 순수 수요 텐서의 행 개수만큼 반복
            obs_low.extend([-1.0] * self.num_items + [0.0, 0.0, 0.0]) # 각 노드에 대한 관측값: 품목 수요(최소 -1.0), 차량 위치(one-hot 인코딩), 거리(0.0), 노드가 작업 가능 여부(0.0)
            obs_high.extend([1.0] * self.num_items + [1.0, 1.0, 1.0]) # 각 노드에 대한 관측값: 품목 수요(최대 1.0), 차량 위치(one-hot 인코딩), 거리(1.0), 노드가 작업 가능 여부(1.0)
        obs_low.extend([0.0] * self.num_items + [0.0] * self.num_groups) # 차량의 품목 용량(최소 0.0)과 그룹별 잔여 용량(최소 0.0)
        obs_high.extend([1.0] * self.num_items + [1.0] * self.num_groups) # 차량의 품목 용량(최대 1.0)과 그룹별 잔여 용량(최대 1.0)

        self.observation_space = Box(low=np.array(obs_low, dtype=np.float32), # 관측 공간 설정
                                     high=np.array(obs_high, dtype=np.float32),
                                     dtype=np.float32)

        self.reset()

    # 고정된 순수 수요 텐서를 생성하는 함수
    def _get_total_unbalance(self):
        return torch.sum(torch.abs(self.net_demand[1:])).item()

    # 환경을 초기화하는 함수(고정된 환경을 생성하는 함수)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.net_demand = self.fixed_net_demand.clone()  # 고정된 순수 수요 텐서를 복사하여 초기화
        self.vehicle_capacity = torch.zeros(self.num_items, dtype=torch.int32) # 차량의 품목 용량을 0으로 초기화
        self.vehicle_pos = 0 # 차량의 초기 위치를 0(차고)으로 설정
        self.step_count = 0 # 현재 스텝 수를 0으로 초기화
        self.initial_unbalance = max(self._get_total_unbalance(), 1e-6) # 초기 불균형을 계산(0이 되지 않도록 최소값을 1e-6으로 설정)
        return self._get_obs(), {}


    # 현재 상태의 관측값을 반환하는 함수
    def _get_obs(self):
        vehicle_pos_onehot = torch.zeros(self.num_nodes, dtype=torch.float32) # 차량 위치를 one-hot 인코딩으로 표현
        vehicle_pos_onehot[self.vehicle_pos] = 1.0  # 현재 차량 위치를 1.0으로 설정
        distances = self.dist_matrix[self.vehicle_pos].clone() # 현재 차량 위치에서 각 노드까지의 거리
        max_dist = self.dist_matrix.max().item() # 최대 거리: 현재 거리 행렬에서 최대값을 가져옴

        group_used = torch.tensor([torch.sum(self.vehicle_capacity[mask]) for mask in self.group_masks]) # 각 그룹별로 차량이 사용 중인 용량을 계산
        group_remaining = self.group_cap_tensor - group_used # 각 그룹별로 남은 용량을 계산

        feasible = torch.zeros(self.num_nodes, dtype=torch.float32) # 각 노드가 작업 가능한지 여부를 저장하는 텐서
        feasible[0] = 1.0 # 노드 0(창고)은 항상 작업 가능
        for n in range(1, self.num_nodes): # 노드 1부터 시작하여 각 노드에 대해 작업 가능 여부를 판단
            for i in range(self.num_items): # 각 품목에 대해
                net = self.net_demand[n, i].item() # 현재 노드의 순수 수요
                gid = self.item_to_group[i].item() # 현재 품목이 속한 그룹
                if (net > 0 and group_remaining[gid] >= 1) or (net < 0 and self.vehicle_capacity[i] >= 1): # 순수 수요가 양수일 때 그룹의 남은 용량이 1 이상이거나, 순수 수요가 음수일 때 차량의 품목 용량이 1 이상인 경우
                    feasible[n] = 1.0
                    break

        obs = []
        group_cap_vector = self.group_cap_tensor[self.item_to_group] # 각 품목이 속한 그룹의 용량을 벡터로 변환
        for i in range(self.num_nodes): # 각 노드에 대한 관측값을 생성
            norm_demand = (self.net_demand[i].float() / group_cap_vector).clamp(-1.0, 1.0).tolist() # 현재 노드의 순수 수요를 그룹 용량으로 정규화하고 -1.0에서 1.0 사이로 클램핑
            node_obs = norm_demand + [vehicle_pos_onehot[i].item(),     # 차량 위치(one-hot 인코딩)
                                      distances[i].item() / max_dist, # 현재 노드까지의 거리(최대 거리로 정규화)
                                      feasible[i].item()] # 현재 노드가 작업 가능한지 여부
            obs.extend(node_obs)

        norm_capacity = (self.vehicle_capacity.float() / group_cap_vector).clamp(0.0, 1.0) # 차량의 품목 용량을 그룹 용량으로 정규화하고 0.0에서 1.0 사이로 클램핑
        obs.extend(norm_capacity.tolist()) # 차량의 품목 용량을 관측값에 추가
        for g in sorted(self.group_cap.keys()): # 각 그룹의 남은 용량을 관측값에 추가 
            obs.append(min(group_remaining[g].item() / self.group_cap[g], 1.0)) # 그룹의 남은 용량을 그룹 용량으로 정규화하고 1.0으로 클램핑

        return np.array(obs, dtype=np.float32)

    # 환경의 상태를 업데이트하고 보상을 계산하는 함수
    def step(self, action):
        prev_unbalance = self._get_total_unbalance() # 이전 상태의 불균형을 저장
        node, item, task_type = map(int, action.tolist()) # 행동(action)을 정수형으로 변환하여 노드, 품목, 작업 유형을 추출

        dist = self.dist_matrix[self.vehicle_pos, node].item() # 현재 차량 위치에서 선택한 노드까지의 거리
        self.vehicle_pos = node # 차량 위치를 선택한 노드로 업데이트
        self.step_count += 1 # 스텝 수를 증가시킴

        net = self.net_demand[node, item].item() # 현재 노드의 순수 수요를 가져옴
        cap = self.vehicle_capacity[item].item() # 현재 차량의 품목 용량을 가져옴
        gid = self.item_to_group[item].item() # 현재 품목이 속한 그룹을 가져옴
        gmask = self.group_masks[gid] # 현재 품목이 속한 그룹의 마스크를 가져옴
        gused = torch.sum(self.vehicle_capacity[gmask]).item() # 현재 그룹에서 차량이 사용 중인 용량을 계산
        glimit = self.group_cap[gid] # 현재 그룹의 용량 제한을 가져옴

        amt = 0 # 작업량 초기화
        pickup = 0 # 픽업량 초기화
        delivery = 0  # 배송량 초기화
        max_dist = self.dist_matrix.max().item() # 최대 거리: 현재 거리 행렬에서 최대값을 가져옴
        reward = -(dist / max_dist) * dist_panalty # 이동 거리로 인한 보상 계산 (거리가 멀수록 보상이 감소) 정규화거리에 특정 보상계수를 곱하여 보상으로 사용

        if node != 0: # 노드가 0(차고)이 아닌 경우
            if task_type == 0 and net > 0: # 픽업 작업인 경우
                amt = min(net, glimit - gused) # 현재 노드의 순수 수요와 그룹 용량 제한을 비교하여 픽업할 수 있는 최대량을 계산
                if amt > 0 and gused + amt <= glimit: # 픽업할 수 있는 양이 0보다 크고 그룹 용량 제한을 초과하지 않는 경우
                    pickup = amt # 픽업량을 설정
                    self.net_demand[node, item] -= amt # 현재 노드의 순수 수요에서 픽업량을 차감
                    self.vehicle_capacity[item] += amt # 차량의 품목 용량에 픽업량을 추가
                else: 
                    reward -= 1.0 # 픽업이 불가능한 경우 보상을 감소
                    amt = 0 # 픽업량을 0으로 설정
            elif task_type == 1 and net < 0: # 배송 작업인 경우
                amt = min(-net, cap) # 현재 노드의 순수 수요의 절댓값과 차량의 품목 용량을 비교하여 배송할 수 있는 최대량을 계산
                if amt > 0 and cap >= amt: # 배송할 수 있는 양이 0보다 크고 차량의 품목 용량이 충분한 경우
                    delivery = amt # 배송량을 설정
                    self.net_demand[node, item] += amt # 현재 노드의 순수 수요에 배송량을 추가
                    self.vehicle_capacity[item] -= amt # 차량의 품목 용량에서 배송량을 차감
                else:
                    reward -= 1.0 # 배송이 불가능한 경우 보상을 감소
                    amt = 0 # 배송량을 0으로 설정
            else:
                reward -= 1.0 # 작업 유형이 잘못되었거나 순수 수요가 0인 경우 보상을 감소
                amt = 0 # 작업량을 0으로 설정

        new_unbalance = self._get_total_unbalance() # 새로운 상태의 불균형을 계산
        delta = prev_unbalance - new_unbalance # 새로운 상태의 불균형을 계산
        if self.initial_unbalance > 0: # 초기 불균형이 0보다 큰 경우
            # reward += (delta / self.initial_unbalance) * progress_reward # 초기 불균형 대비 변화량에 따라 보상을 조정
            reward += np.sign(delta) * progress_reward * (abs(delta) / self.initial_unbalance) ** 0.5
        reward -= step_penalty # 매 스텝마다 일정한 감소 보상 적용(스텝을 최대한 간소화 하기 위해)

        terminated = bool(torch.all(self.net_demand[1:] == 0)) and self.vehicle_pos == 0 # 모든 노드의 순수 수요가 0이고(작업을 모두 완료) 차량 위치가 0(차고)인 경우(차고지 복귀) 환경을 종료
        truncated = self.step_count >= self.max_steps # 최대 스텝 수에 도달한 경우 환경을 중단

        if terminated: # 환경terminated시 (작업을 모두 완료하고 차량이 차고지로 복귀한 경우)
            reward += complited_reward  # 작업 완료 보상 추가

        return self._get_obs(), reward, terminated, truncated, {
            "pickup": pickup,
            "delivery": delivery,
            "cap": self.vehicle_capacity.tolist(),
            "action": [node, item, task_type],
            "dist": dist,
            "amt": amt
        }

# 테스트 실행 예시
fixed_net_demand = np.array([
    [   0,    0,    0],
    [-220,  130,    0],
    [ -50,    0,   60],
    [  20,    0,    0],
    [  90,  170,    0],
    [   0,   40,    0],
    [   0, -100,    0],
    [   0, -240,  -60],
    [ 160,    0,    0]
])

fixed_net_demand = fixed_net_demand  # 단위 변환


dist_matrix = np.array([
    #   0     1      2      5    10    12    13    14    17
    [0.00, 0.84, 0.16, 0.11, 0.12, 0.78, 0.30, 0.44, 0.11], # 0
    [0.84, 0.00, 1.27, 1.81, 0.70, 1.56, 1.12, 1.22, 1.58], # 1
    [0.16, 1.27, 0.00, 0.65, 0.90, 2.43, 1.87, 1.28, 0.53], # 2
    [0.11, 1.81, 0.65, 0.00, 1.44, 2.21, 1.18, 0.67, 0.17], # 5
    [0.12, 0.70, 0.90, 1.44, 0.00, 2.11, 1.43, 1.50, 1.24], # 10
    [0.78, 1.56, 2.43, 2.21, 2.11, 0.00, 0.61, 1.19, 2.11], # 12
    [0.30, 1.12, 1.87, 1.18, 1.43, 0.61, 0.00, 0.44, 1.43], # 13
    [0.44, 1.22, 1.28, 0.67, 1.50, 1.19, 0.44, 0.00, 0.74], # 14
    [0.11, 1.58, 0.53, 0.17, 1.24, 2.11, 1.43, 0.74, 0.00] # 17
])
item_to_group = [0, 0, 1]
group_cap = {0: 600, 1: 600}

def greedy_balanced_route(dist_matrix, net_demand, item_to_group, group_cap, depot=0):
    N, K = net_demand.shape
    u = net_demand.astype(float).copy()
    initial_unbal = np.sum(np.abs(u))
    total_dist = 0.0
    route = []  # (from_node, to_node, item, signed_qty, ΔU, distance, score)
    current = depot

    load_by_item = {k: 0.0 for k in range(K)}
    load_by_group = {g: 0.0 for g in group_cap}

    first_step = True

    while np.sum(np.abs(u)) > 0:
        candidates = []

        for i in range(N):
            if i == depot:
                continue
            for k in range(K):
                g = item_to_group[k]
                val = u[i, k]

                # 첫 스텝이면 반드시 depot에서 출발
                if first_step and current != depot:
                    continue

                # 픽업
                if val > 0 and load_by_group[g] < group_cap[g]:
                    q = min(val, group_cap[g] - load_by_group[g])
                    if q > 0:
                        score = q / dist_matrix[current, i] if dist_matrix[current, i] > 0 else q * 1e6
                        candidates.append((i, k, -q, q, dist_matrix[current, i], score))

                # 배송
                elif val < 0 and load_by_item[k] > 0:
                    q = min(-val, load_by_item[k])
                    if q > 0:
                        score = q / dist_matrix[current, i] if dist_matrix[current, i] > 0 else q * 1e6
                        candidates.append((i, k, q, q, dist_matrix[current, i], score))

        if not candidates:
            break

        # 가장 좋은 후보 선택
        i_sel, k_sel, signed_q, delta_u, dist, score = max(candidates, key=lambda x: x[-1])
        g = item_to_group[k_sel]

        # 상태 갱신
        u[i_sel, k_sel] += signed_q
        if signed_q < 0:  # 픽업
            load_by_item[k_sel] += -signed_q
            load_by_group[g] += -signed_q
        else:  # 배송
            load_by_item[k_sel] -= signed_q
            load_by_group[g] -= signed_q

        # 거리 계산 & 경로 기록
        move_dist = dist_matrix[current, i_sel]
        total_dist += move_dist
        route.append((current, i_sel, k_sel, int(signed_q), delta_u, move_dist, score))

        # 이동 후 상태
        current = i_sel
        first_step = False

        # depot 복귀 시 다음 스텝도 depot 출발 강제
        if current == depot:
            first_step = True

    # 마지막 위치가 depot이 아니면 depot 복귀
    if current != depot:
        return_dist = dist_matrix[current, depot]
        total_dist += return_dist
        route.append((current, depot, None, 0, 0.0, return_dist, 0.0))

    # 결과 DataFrame
    final_unbal = np.sum(np.abs(u))
    total_reduction = initial_unbal - final_unbal
    df_route = pd.DataFrame(route, columns=["From", "To", "Item", "SignedQty", "ΔU", "Distance", "Score"])
    return df_route, total_dist, total_reduction


df_route, total_dist, total_reduction = greedy_balanced_route(
    dist_matrix, fixed_net_demand, item_to_group, group_cap
)

print(df_route.to_string(index=False))
print("총 이동 거리:", total_dist)
print("총 불균형 감소량:", total_reduction)




# 누적 적재량 추적
load_by_item = [0] * fixed_net_demand.shape[1]
cumulative_dist = 0.0

for step, row in df_route.iterrows():
    from_node = int(row["From"])
    to_node = int(row["To"])
    
    # NaN 방지 처리
    if pd.isna(row["Item"]):
        item = -1
    else:
        item = int(row["Item"])
    
    qty = int(row["SignedQty"])
    delta = row["ΔU"]
    dist = row["Distance"]
    score = row["Score"]
    cumulative_dist += dist

    task = "픽업" if qty < 0 else "배송"
    abs_qty = abs(qty)

    # 적재량 갱신
    if item >= 0:  # depot 복귀 시 스킵
        if qty < 0:
            load_by_item[item] += abs_qty
        else:
            load_by_item[item] -= abs_qty

    # 작업 결과 출력
    print(f"Step {step + 1}: {from_node} → {to_node}, 품목 {item}, 작업 {task}, 수량 {abs_qty} → "
          f"배송 {0 if qty < 0 else abs_qty}, 픽업 {abs_qty if qty < 0 else 0}, "
          f"적재 {load_by_item}, 이동거리 {dist:.2f}, 누적거리 {cumulative_dist:.2f}")

import networkx as nx
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# 1) 좌표를 한 번만 생성
coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(dist_matrix)

def visualize_route_mds_df(df_route, coords, title="Route Visualization 1"):
    node_locs = {i: tuple(coords[i]) for i in range(len(coords))}
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6)); plt.title(title)

    # 노드
    for i, (x, y) in node_locs.items():
        plt.scatter(x, y, c="red" if i == 0 else "blue", s=250, zorder=2)
        plt.text(x + 0.05, y + 0.05, f"Node {i}", fontsize=10, zorder=3)
        
    # 간선(직선 + 화살표)
    for step, row in df_route.iterrows():
        i0, i1 = int(row["From"]), int(row["To"])
        x0, y0 = node_locs[i0]; x1, y1 = node_locs[i1]
        plt.plot([x0, x1], [y0, y1], color="green", linewidth=1, zorder=1)
        dx, dy = x1 - x0, y1 - y0
        plt.arrow(x0, y0, dx*0.9, dy*0.9, head_width=0.05, head_length=0.1,
                  fc='green', ec='green', alpha=0.85, length_includes_head=True, zorder=1)
        mx, my = (x0 + x1)/2, (y0 + y1)/2
        plt.text(mx, my, f"{step+1}", fontsize=8, color="darkgreen", zorder=4)
    plt.grid(True); plt.axis("equal"); plt.tight_layout(); plt.show()

# 호출 예시
visualize_route_mds_df(df_route, coords)
