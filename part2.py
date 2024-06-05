"""
Part 2. Q-Learning
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

def generate_mutate_path_2opt(path, dist_matrix):
    """ 2-opt 아이디어 기반으로 경로 변형
        경로에서 일부 구간을 뒤집어서 성능이 향상되면 반환(꼬인 경로 풀기)
    """
    origin_path = path.copy()
    origin_distance = total_distance(origin_path, dist_matrix)

    for i in range(1, len(origin_path) - 1):
        for k in range(i + 1, len(origin_path)):
            new_path = origin_path[:i] + origin_path[i:k+1][::-1] + origin_path[k+1:]
            new_distance = total_distance(new_path, dist_matrix)
            
            if new_distance < origin_distance:
                return new_path
            
    print('이 방법으로 더 이상 향상시킬 수 없습니다.')
    return None

"""
def two_opt(dist_matrix, initial_path):
    best_path = initial_path  # 초기 경로 설정
    best_distance = total_distance(best_path, dist_matrix)  # 초기 경로의 총 거리 계산
    
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_path) - 1):
            for k in range(i + 1, len(best_path)):
                # 경로의 일부를 뒤집어서 새로운 경로 생성
                new_path = best_path[:i] + best_path[i:k+1][::-1] + best_path[k+1:]
                new_distance = total_distance(new_path, dist_matrix)
                
                # 새로운 경로가 더 짧다면 갱신
                if new_distance < best_distance:
                    best_path = new_path
                    best_distance = new_distance
                    improved = True  # 경로가 개선되었음을 표시
    return best_path, best_distance
"""

# 몬테카를로 기법을 사용하여 value table 업데이트 함수
def monte_carlo_value_iteration(dist_matrix, num_simulations=50, alpha=0.1):

    # dist_matrix 기반 greedy solution 생성
    curr_path = value_greedy_solution(value_table = -dist_matrix)
    origin_distance = total_distance(curr_path, dist_matrix)

    num_cities = dist_matrix.shape[0]

    value_table = np.zeros((num_cities, num_cities))

    for _ in tqdm(range(num_simulations)):
        # 2-opt 아이디어 기반 mutated solution 생성
        mutate_path = generate_mutate_path_2opt(curr_path, dist_matrix)
        mutate_distance = total_distance(mutate_path, dist_matrix)

        reward = origin_distance - mutate_distance # 원래 경로(greedy) 대비 향상된 거리

        curr_path = mutate_path
        
        # 각 상태-액션 쌍에 대한 value update
        for i in range(num_cities - 1):
            state = mutate_path[i]
            action = mutate_path[i+1]
    
            # V(s) ← V(s) + α [G_t - V(s)] (alpha는 learning rate)
            value_table[state, action] += alpha * (reward - value_table[state, action])
    print(mutate_distance)
    
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
print("Solution:", solution)
print("distance:", total_distance(solution, dist_matrix))