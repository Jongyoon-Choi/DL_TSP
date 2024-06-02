"""
Part 2. Q-Learning
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

# 랜덤 경로 생성 함수 (샘플링 방법 수정 필요)
def generate_random_path(num_cities):
    path = np.arange(num_cities)
    np.random.shuffle(path)
    return path

# 몬테카를로 기법을 사용하여 value table 업데이트 함수
def monte_carlo_value_iteration(dist_matrix, num_simulations=10000, gamma=0.9):
    value_table = np.zeros((num_cities, num_cities))
    counts = np.zeros((num_cities, num_cities))
    
    for _ in tqdm(range(num_simulations)):
        path = generate_random_path(num_cities)
        distance = total_distance(path, dist_matrix)
        
        # 각 상태-액션 쌍에 대한 보상을 계산하고 누적
        for i in range(num_cities - 1):
            state = path[i]
            action = path[i+1]
            reward = 1 / distance  # 거리가 짧을수록 보상이 높아지도록 설정
            value_table[state, action] += reward
            counts[state, action] += 1
    
    # 평균 보상으로 value table 업데이트
    for i in range(num_cities):
        for j in range(num_cities):
            if counts[i, j] > 0:
                value_table[i, j] /= counts[i, j]
    
    return value_table

def greedy_solution(value_table):
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
num_cities = 998
dist_matrix=dist_matrix[:num_cities, :num_cities]

# 몬테카를로 기법을 사용하여 Value table 업데이트
value_table = monte_carlo_value_iteration(dist_matrix)
# print(value_table[0])

# Value table을 사용하여 greedy solution 생성
solution = greedy_solution(value_table)
print("Solution:", solution)
print("distance:", total_distance(solution, dist_matrix))