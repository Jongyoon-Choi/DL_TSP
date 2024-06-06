"""
Part 2. Q-Learning
"""
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# 시드 고정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# 시드 설정
set_seed(42)

# 경로의 총 거리 계산
def total_distance(path, dist_matrix):
    if len(path) < 2:
        return 0 
    
    total_dist = 0
    for i in range(len(path) - 1):
        total_dist += dist_matrix[path[i], path[i+1]].item()
        
    if len(path) == dist_matrix.shape[0]:
        total_dist += dist_matrix[path[-1], path[0]].item()

    return total_dist

def generate_mutate_path_2opt(path):
    """ 2-opt 아이디어 기반으로 경로 변형
        경로에서 일부 구간을 뒤집어서 반환
    """
    new_path = path.copy()
    num_cities = len(new_path)

    # 두 개의 임의의 인덱스 선택
    i, k = sorted(random.sample(range(1, num_cities), 2))

    # i에서 k까지의 구간을 뒤집음
    new_path[i:k+1] = list(reversed(new_path[i:k+1]))

    return new_path

# 몬테카를로 기법을 사용하여 value table 업데이트 함수
def monte_carlo_value_iteration(dist_matrix, num_simulations=1000000, alpha=0.03):
    print('num_simulations =',num_simulations)
    num_cities = dist_matrix.shape[0]

    # 랜덤한 경로 생성 (출발점 고정)
    curr_path = np.concatenate(([0], np.random.permutation(np.arange(1, num_cities))))

    value_table = np.zeros((num_cities, num_cities))

    for _ in tqdm(range(num_simulations)):
        # 2-opt 아이디어 기반 mutated solution 생성
        mutate_path = generate_mutate_path_2opt(curr_path)
        mutate_distance = total_distance(mutate_path, dist_matrix)

        reward = np.exp((250 - mutate_distance)/25) # 향상된 정도의 수치화

        if (mutate_distance<total_distance(curr_path, dist_matrix)):
            curr_path = mutate_path
        
        # 각 상태-액션 쌍에 대한 value update
        for i in range(num_cities - 1):
            state = mutate_path[i]
            action = mutate_path[i+1]
    
            # V(s) ← V(s) + α [G_t - V(s)] (alpha는 learning rate)
            value_table[state, action] += alpha * (reward - value_table[state, action])
    
    return value_table

# value table 기반 greedy 탐색
def value_greedy_solution(value_table):
    num_cities = value_table.shape[0]
    visited = [False] * num_cities
    solution = []

    # 시작 도시는 0으로 설정
    current_node = 0
    visited[current_node] = True
    solution.append(current_node)
    
    for _ in range(num_cities - 1):
        # 가장 value가 높은 노드를 찾기
        highest_value = -np.inf
        highest_value_node = None

        for next_node in range(num_cities):
            if not visited[next_node] and value_table[current_node, next_node] > highest_value:
                highest_value = value_table[current_node, next_node]
                highest_value_node = next_node

        # 가장 value가 높은 노드를 방문
        visited[highest_value_node] = True
        solution.append(highest_value_node)
        current_node = highest_value_node
    
    return solution

# CSV 파일에서 TSP 좌표 데이터 로드
data = pd.read_csv('2024_AI_TSP.csv', header=None)
coordinates = data.iloc[:, :2].values

# 거리 행렬 생성
dist_matrix = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=2)

# 몬테카를로 기법을 사용하여 Value table 생성
value_table = monte_carlo_value_iteration(dist_matrix)

# Value table을 사용하여 greedy solution 생성
solution = value_greedy_solution(value_table)
# print("Solution:", solution)
print("distance:", total_distance(solution, dist_matrix))