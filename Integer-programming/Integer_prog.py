import torch
import numpy as np
from pulp import *
import pandas as pd

def solve_cluster_balancing(fixed_net_demand, num_clusters=4):
    """
    각 클러스터의 공급-수요 합이 0이 되도록 노드를 분할하는 정수계획법 모델
    
    Parameters:
    - fixed_net_demand: torch.tensor of shape (n_nodes, n_commodities)
    - num_clusters: 클러스터 수
    
    Returns:
    - 최적화 결과 및 클러스터 할당
    """
    
    # 데이터 준비
    demand_np = fixed_net_demand.numpy()
    n_nodes, n_commodities = demand_np.shape
    
    print(f"노드 수: {n_nodes}, 품목 수: {n_commodities}, 클러스터 수: {num_clusters}")
    print(f"전체 공급-수요 합계: {demand_np.sum(axis=0)}")
    
    # 문제 생성
    prob = LpProblem("Cluster_Balancing", LpMinimize)
    
    # 결정변수: x[i,k,c] = 노드 i의 품목 k를 클러스터 c에 할당하는 양
    # 음수 수요는 양수로 변환하여 처리
    x = {}
    for i in range(n_nodes):
        for k in range(n_commodities):
            for c in range(num_clusters):
                # 공급(양수)과 수요(음수)를 별도로 처리
                if demand_np[i, k] >= 0:  # 공급
                    x[(i, k, c, 'supply')] = LpVariable(f"x_supply_{i}_{k}_{c}", 
                                                       lowBound=0, upBound=demand_np[i, k], 
                                                       cat='Integer')
                else:  # 수요
                    x[(i, k, c, 'demand')] = LpVariable(f"x_demand_{i}_{k}_{c}", 
                                                       lowBound=0, upBound=-demand_np[i, k], 
                                                       cat='Integer')
    
    # 추가 결정변수: 각 클러스터 사용 여부를 나타내는 이진 변수
    cluster_used = {}
    for c in range(num_clusters):
        cluster_used[c] = LpVariable(f"cluster_used_{c}", cat='Binary')
    
    # 목적함수: 사용되지 않는 클러스터 수 최소화 (모든 클러스터 사용 강제)
    prob += -lpSum([cluster_used[c] for c in range(num_clusters)])
    
    # 제약조건 1: 각 노드의 공급/수요량이 정확히 분할되어야 함
    for i in range(n_nodes):
        for k in range(n_commodities):
            if demand_np[i, k] >= 0:  # 공급
                prob += lpSum([x[(i, k, c, 'supply')] for c in range(num_clusters)]) == demand_np[i, k]
            else:  # 수요
                prob += lpSum([x[(i, k, c, 'demand')] for c in range(num_clusters)]) == -demand_np[i, k]
    
    # 제약조건 2: 각 클러스터의 각 품목에 대한 공급-수요 균형
    for c in range(num_clusters):
        for k in range(n_commodities):
            supply_sum = lpSum([x.get((i, k, c, 'supply'), 0) for i in range(n_nodes)])
            demand_sum = lpSum([x.get((i, k, c, 'demand'), 0) for i in range(n_nodes)])
            prob += supply_sum == demand_sum
    
    # 제약조건 3: 클러스터 사용 여부 연결 (BigM 제약)
    M = 1000  # 충분히 큰 수
    for c in range(num_clusters):
        total_allocation = lpSum([x.get((i, k, c, 'supply'), 0) + x.get((i, k, c, 'demand'), 0) 
                                 for i in range(n_nodes) for k in range(n_commodities)])
        prob += total_allocation >= cluster_used[c]  # 할당이 있으면 사용됨
        prob += total_allocation <= M * cluster_used[c]  # 사용되지 않으면 할당 없음
    
    # 제약조건 4: 모든 클러스터 사용 강제
    prob += lpSum([cluster_used[c] for c in range(num_clusters)]) == num_clusters
    
    # 문제 해결
    print("최적화 시작...")
    prob.solve(PULP_CBC_CMD(msg=1))
    
    # 결과 분석
    if prob.status == LpStatusOptimal:
        print("최적해를 찾았습니다!")
        
        # 결과 정리
        results = []
        cluster_assignments = {c: {'nodes': [], 'balance': np.zeros(n_commodities)} 
                              for c in range(num_clusters)}
        
        for i in range(n_nodes):
            node_allocation = {c: np.zeros(n_commodities) for c in range(num_clusters)}
            
            for k in range(n_commodities):
                for c in range(num_clusters):
                    if demand_np[i, k] >= 0:  # 공급
                        if (i, k, c, 'supply') in x:
                            val = x[(i, k, c, 'supply')].varValue or 0
                            if val > 0:
                                node_allocation[c][k] += val
                                cluster_assignments[c]['balance'][k] += val
                    else:  # 수요
                        if (i, k, c, 'demand') in x:
                            val = x[(i, k, c, 'demand')].varValue or 0
                            if val > 0:
                                node_allocation[c][k] -= val
                                cluster_assignments[c]['balance'][k] -= val
            
            # 각 노드가 할당된 클러스터 정보 저장
            for c in range(num_clusters):
                if np.any(node_allocation[c] != 0):
                    cluster_assignments[c]['nodes'].append((i, node_allocation[c]))
                    results.append({
                        'node': i,
                        'cluster': c,
                        'original_demand': demand_np[i],
                        'allocated_demand': node_allocation[c]
                    })
        
        return True, results, cluster_assignments
    
    else:
        print(f"최적화 실패: {LpStatus[prob.status]}")
        return False, None, None


def print_results(success, results, cluster_assignments):
    """결과 출력"""
    if not success:
        print("해결할 수 없는 문제입니다.")
        return
    
    print("\n=== 클러스터별 결과 ===")
    for c, info in cluster_assignments.items():
        print(f"\n클러스터 {c}:")
        print(f"  균형 확인: {info['balance']} (모두 0이어야 함)")
        print(f"  할당된 노드 수: {len(info['nodes'])}")
        
        for node_idx, allocation in info['nodes']:
            print(f"    노드 {node_idx}: {allocation}")
    
    print("\n=== 노드별 할당 상세 ===")
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        for node in sorted(df_results['node'].unique()):
            node_data = df_results[df_results['node'] == node]
            print(f"\n노드 {node} (원래: {node_data.iloc[0]['original_demand']}):")
            for _, row in node_data.iterrows():
                print(f"  클러스터 {row['cluster']}: {row['allocated_demand']}")


# 실행 예제
if __name__ == "__main__":
    # 주어진 데이터
    fixed_net_demand = torch.tensor([
        [-5, -4,  1],
        [-5, -5,  5],
        [-1,  5,  3],
        [-5,  5, -5],
        [ 5, -5, -5],
        [ 1, -1,  2],
        [ 1, -2, -3],
        [-5, -3, -5],
        [-1, -2,  2],
        [ 3,  5, -1],
        [-2, -1, -5],
        [ 1, -3,  0],
        [ 2,  5,  3],
        [ 3, -5,  3],
        [-1,  5,  2],
        [ 4,  5,  1],
        [ 0,  5,  2],
        [ 5, -4,  0]
    ], dtype=torch.float32)
    
    print("품목별 전체 공급-수요 현황:")
    total_balance = fixed_net_demand.sum(dim=0)
    print(f"품목1: {total_balance[0]}, 품목2: {total_balance[1]}, 품목3: {total_balance[2]}")
    
    # 해결 가능성 확인
    if torch.any(total_balance != 0):
        print("\n경고: 전체 공급-수요가 균형을 이루지 않습니다.")
        print("각 품목별로 공급과 수요의 총합이 같아야 클러스터 균형이 가능합니다.")
    else:
        print("\n✓ 전체 공급-수요가 균형을 이룹니다. 클러스터 분할이 가능합니다!")
    
    # 문제 해결
    success, results, cluster_assignments = solve_cluster_balancing(fixed_net_demand, num_clusters=4)
    print_results(success, results, cluster_assignments)
