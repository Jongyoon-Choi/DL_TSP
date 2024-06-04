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

def greedy_tsp_solution(dist_matrix):
    num_cities = dist_matrix.shape[0]
    visited = [False] * num_cities
    solution = []

    # 시작 노드 (여기서는 0번 노드로 시작)
    current_node = 0
    visited[current_node] = True
    solution.append(current_node)

    for _ in range(num_cities - 1):
        # 가장 가까운 미방문 노드를 찾기
        nearest_distance = float('inf')
        nearest_node = None

        for next_node in range(num_cities):
            if not visited[next_node] and dist_matrix[current_node, next_node] < nearest_distance:
                nearest_distance = dist_matrix[current_node, next_node]
                nearest_node = next_node

        # 가장 가까운 노드를 방문
        visited[nearest_node] = True
        solution.append(nearest_node)
        current_node = nearest_node

    return solution

# mutated solution 생성
def generate_mutate_path(solution, num_mutate):
    result = solution.copy()

    num_cities = len(result)

    # 변이시킬 index 선택 (오름 or 내림차순 X)
    mutate_idx = random.sample(range(0, num_cities - 1), num_mutate)

    # mutate_idx에 해당하는 원소들을 한 칸씩 앞으로 이동
    temp = result[mutate_idx[0]]
    for i in range(num_mutate - 1):
        result[mutate_idx[i]]=result[mutate_idx[i+1]]
    result[mutate_idx[-1]] = temp

    return result

# 몬테카를로 기법을 사용하여 value table 업데이트 함수
def monte_carlo_value_iteration(dist_matrix, num_simulations=10000, gamma=0.9):
    # greedy solution 생성
    greedy_solution = greedy_tsp_solution(dist_matrix)
    greedy_distance = total_distance(greedy_solution, dist_matrix)

    num_cities = dist_matrix.shape[0]
    weight = num_simulations * 0.1

    value_table = np.zeros((num_cities, num_cities))
    counts = np.zeros((num_cities, num_cities))

    for _ in tqdm(range(num_simulations)):
        # greedy solution 기반 mutated solution 생성
        mutate_path = generate_mutate_path(greedy_solution, 2)
        mutate_distance = total_distance(mutate_path, dist_matrix)

        # if(mutate_distance < greedy_distance):
        #     print('향상')
        
        # 각 상태-액션 쌍에 대한 보상을 계산하고 누적
        for i in range(num_cities - 1):
            state = mutate_path[i]
            action = mutate_path[i+1]
            reward = 50 + greedy_distance - mutate_distance  # greedy 대비 거리의 증감을 보상으로 설정

            # 거리가 향상되면 보상에 가중치를 줌
            if reward > 50:
                value_table[state, action] += reward * weight
            else:
                value_table[state, action] += reward
            counts[state, action] += 1
    
    # 평균 보상으로 value table 업데이트
    for i in range(num_cities):
        for j in range(num_cities):
            if counts[i, j] > 0:
                value_table[i, j] /= counts[i, j]
    
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

# 몬테카를로 기법을 사용하여 Value table 업데이트
value_table = monte_carlo_value_iteration(dist_matrix)

# Value table을 사용하여 greedy solution 생성
solution = value_greedy_solution(value_table)
print("Solution:", solution)
print("distance:", total_distance(solution, dist_matrix))