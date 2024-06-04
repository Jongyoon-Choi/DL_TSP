"""
Part 1. Value-based Policy Extraction
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

def total_distance(solution, W):
    total_dist = 0
    for i in range(len(solution) - 1):
        total_dist += W[solution[i], solution[i+1]].item()
        
    # if this solution is "complete", go back to initial point
    if len(solution) == W.shape[0]:
        total_dist += W[solution[-1], solution[0]].item()

    return total_dist

# CSV 파일에서 TSP 좌표 데이터 로드
data = pd.read_csv('2024_AI_TSP.csv', header=None)
coordinates = data.iloc[:, :2].values

# 거리 행렬 생성
dist_matrix = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=2)

# 역수를 취하여 reward_table 생성 (자기 자신에 대한 거리는 0으로 유지)
reward_table = np.zeros_like(dist_matrix)
non_zero_indices = dist_matrix > 0
reward_table[non_zero_indices] = 1 / dist_matrix[non_zero_indices]

"""
# 1-normalized_distance로 reward_table 생성
reward_table = np.zeros_like(dist_matrix)
non_zero_indices = dist_matrix > 0
normalized_distances = 1 - (dist_matrix - np.min(dist_matrix)) / (np.max(dist_matrix) - np.min(dist_matrix))
reward_table[non_zero_indices] = normalized_distances[non_zero_indices]
"""

# Value table 생성
def generate_value_table(rewards, gamma=0.9, max_iter=1000, tol=1e-6):
    # Value table 초기화: 모든 상태에 대해 가치를 0으로 설정
    num_cities=len(rewards[0])
    value_table = np.zeros((num_cities, num_cities))
    
    # 반복
    for iter in tqdm(range(max_iter)):
        prev_value_table = np.copy(value_table)
        for i in range(num_cities):
            for j in range(num_cities):
                # 보상을 할당
                if i != j:
                    value_table[i, j] = rewards[i, j] + gamma * np.max(prev_value_table[j])
        
        # 수렴 확인
        if np.max(np.abs(value_table - prev_value_table)) < tol:
            print('반복 횟수 : ',iter)
            break
    
    return value_table

# value table 기반 greedy 탐색
def value_greedy_solution(value_table):
    num_cities = len(value_table)
    # 시작 도시는 0으로 설정
    current_city = 0
    # 방문한 도시를 저장하는 리스트
    visited_cities = [current_city]
    
    # 모든 도시를 방문할 때까지 반복
    while len(visited_cities) < num_cities:
        # 방문하지 않은 도시의 가치 리스트 생성
        unvisited_values = [value_table[current_city, city] if city not in visited_cities else -np.inf for city in range(num_cities)]
        # 이미 방문한 도시를 제외하고 가치가 가장 높은 도시 선택
        next_city = np.argmax(unvisited_values)
        # 다음 도시를 방문한 도시 리스트에 추가
        visited_cities.append(next_city)
        # 다음 도시를 현재 도시로 설정
        current_city = next_city
    
    return visited_cities

# 테스트를 위한 예시
num_cities = 200
reward_table=reward_table[:num_cities, :num_cities]
dist_matrix=dist_matrix[:num_cities, :num_cities]

# Value Iteration 수행
optimal_values = generate_value_table(reward_table)
# print(optimal_values[0])

# Value table을 사용하여 greedy solution 생성
solution = value_greedy_solution(optimal_values)
print("Solution:", solution)
print("distance:", total_distance(solution, dist_matrix))