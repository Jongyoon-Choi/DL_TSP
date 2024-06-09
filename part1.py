"""
Part 1. Value-based Policy Extraction
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

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

# Value Iteration 방식으로 Value table 생성
def generate_value_table(dist_matrix, gamma=0.9, max_iter=1000, tol=1e-6):
    
    # Value table 초기화: 모든 상태에 대해 value를 0으로 설정
    num_cities=len(dist_matrix[0])
    value_table = np.zeros((num_cities, num_cities))

    # 거리의 역수를 취하여 reward_table 생성 (자기 자신에 대한 거리는 0으로 유지)
    reward_table = np.zeros_like(dist_matrix)
    non_zero_indices = dist_matrix > 0
    reward_table[non_zero_indices] = 1 / dist_matrix[non_zero_indices]
    
    # 반복
    for iter in tqdm(range(max_iter)):
        prev_value_table = np.copy(value_table)
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    value_table[i, j] = reward_table[i, j] + gamma * np.max(prev_value_table[j])
        
        # 수렴 확인
        if np.max(np.abs(value_table - prev_value_table)) < tol:
            print('반복 횟수 : ',iter)
            break
    
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

# Value Iteration을 사용하여 Value table 생성
optimal_values = generate_value_table(dist_matrix)

# 생성된 Value table을 사용하여 greedy solution 생성
solution = value_greedy_solution(optimal_values)
print("Solution:", solution)
print("distance:", total_distance(solution, dist_matrix))