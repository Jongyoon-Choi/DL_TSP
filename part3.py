from collections import namedtuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class QNet(nn.Module):

    def __init__(self, emb_dim, T=4, node_dim = 5):
        super(QNet, self).__init__()
        self.emb_dim = emb_dim
        self.T = T
        self.node_dim = node_dim      
        
        self.theta1 = nn.Linear(self.node_dim, self.emb_dim, True)
        self.theta2 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta3 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta4 = nn.Linear(1, self.emb_dim, True)
        self.theta5 = nn.Linear(2*self.emb_dim, 1, True)
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True)
        
        self.layer = nn.Linear(self.emb_dim, self.emb_dim, True) 
        
    def forward(self, xv, Ws):

        num_nodes = xv.shape[1]   # 전체 도시의 수
        batch_size = xv.shape[0]  # batch size
        
        
        # distance matrix의 값이 0인 곳은 0으로, 0 이상인 곳은 1로 채운 conn_matries
        # --> 대각 원소 = 0, 그 외의 원소 = 1
        conn_matrices = torch.where(Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)).to(device)
        
        mu = torch.zeros(batch_size, num_nodes, self.emb_dim, device=device)

        # state 정보를 embedding 
        s1 = self.theta1(xv)                     # (batch_size, num_nodes, 5) --> (batch_size, num_nodes, emb_dim)
        s1 = self.layer(F.relu(s1))              # (batch_size, num_nodes, emb_dim) --> (batch_size, num_nodes, emb_dim)
        
        # distance matrix 정보를 embedding
        s3_0 = Ws.unsqueeze(3)                   # (batch_size, num_nodes, num_nodes) --> (batch_size, num_nodes, num_nodes, 1)
        s3_1 = F.relu(self.theta4(s3_0))         # (batch_size, num_nodes, num_nodes, 1) --> (batch_size, num_nodes, num_nodes, emb_dim)
        s3_2 = torch.sum(s3_1, dim=1)            # (batch_size, num_nodes, num_nodes, emb_dim) --> (batch_size, num_nodes, emb_dim)
        s3 = self.theta3(s3_2)                   # (batch_size, num_nodes, emb_dim) --> (batch_size, num_nodes, emb_dim)
        

        # state 정보(s1)와 각 state에 대한 나머지 node들의 distance 정보(s3)를 함께 embedding
        for _ in range(self.T):
            s2 = self.theta2(conn_matrices.matmul(mu))    # state와 action이 동일한 경우 (대각 원소)를 제외하고 정보 융합
            mu = F.relu(s1 + s2 + s3)
            
        # 전체적인 state와 distance에 대한 정보를 모든 노드에 동일하게 제공
        global_state = self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))
        
        # 각각의 state에 대한 정보
        local_action = self.theta7(mu)  
            
        # 전체적인 정보와, 각각의 state에서의 정보를 함께 융합하여 Q-value 예측
        out = F.relu(torch.cat([global_state, local_action], dim=2))
        return self.theta5(out).squeeze(dim=2)
    


class QTrainer():
    def __init__(self, model, optimizer, lr_scheduler):
        # QNetwork 인스턴스
        self.model = model                  

        # 학습에 활용할 QNetwork 학습 구성요소
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = nn.MSELoss()
    

    def predict(self, state_tsr, W):
        # batch가 1인 인풋을 가정. inference 시 호출
        with torch.no_grad():
            estimated_q_value = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))

        return estimated_q_value[0]
                

    def get_best_action(self, state_tsr, state):
        """ 
            주어진 state에 대해 최적의 greedy action을 선택하는 단계. 
            다음 노드(aciton)의 index와 추정된 q_value.
        """
        W = state.W
        estimated_q_value = self.predict(state_tsr, W)  
        sorted_q_value_idx = estimated_q_value.argsort(descending=True)
        
        solution = state.partial_solution
        
        already_in = set(solution)
        for idx in sorted_q_value_idx.tolist():
            if (len(solution) == 0 or W[solution[-1], idx] > 0) and idx not in already_in:
                return idx, estimated_q_value[idx].item()
        

    def batch_update(self, states_tsrs, Ws, actions, targets):
        """ 
            Batch단위의 (embedding of state, distance matrix, action, target_q_value)를 
            통해 Gradient를 통한 최적화를 수행하는 단계.
            states_tsrs: list of (single) state tensors
            Ws: list of W tensors
            actions: list of actions taken
            targets: list of targets (resulting estimated q_value after taking the actions)
        """        
        Ws_tsr = torch.stack(Ws).to(device)
        xv = torch.stack(states_tsrs).to(device)
        self.optimizer.zero_grad()
        
        estimated_q_value = self.model(xv, Ws_tsr)[range(len(actions)), actions]
        
        
        loss = self.loss_fn(estimated_q_value, torch.tensor(targets, device=device))
        loss_val = loss.item()
        
        loss.backward()
        self.optimizer.step()        
        self.lr_scheduler.step()
        
        return loss_val
    
class State:
    def __init__(self, W, partial_solution):
        self.W = W
        self.partial_solution = partial_solution

# 상태 벡터 생성
def state2tens(state, coordinates):
    xv = []
    for i in range(num_nodes):
        visit_status = 1 if i in set(state.partial_solution) else 0
        first_node = 1 if i == state.partial_solution[0] else 0
        last_node = 1 if i == state.partial_solution[-1] else 0
        x, y = coordinates[i]
        xv.append([visit_status, first_node, last_node, x, y])
    
    return torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=device)

def total_distance(solution, W):
    if len(solution) < 2:
        return 0  # there is no travel
    
    total_dist = 0
    for i in range(len(solution) - 1):
        total_dist += W[solution[i], solution[i+1]].item()
        
    # if this solution is "complete", go back to initial point
    if len(solution) == W.shape[0]:
        total_dist += W[solution[-1], solution[0]].item()

    return total_dist

# Note: we store state tensors in experience to compute these tensors only once later on
Experience = namedtuple('Experience', ('state', 'state_tsr', 'action', 'reward', 'next_state', 'next_state_tsr'))

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.nr_inserts = 0
        
    def remember(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        self.nr_inserts += 1
        
    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return min(self.nr_inserts, self.capacity)
    
def is_state_final(state):
    return len(set(state.partial_solution)) == state.W.shape[0]

def get_next_neighbor_random(state):
    solution, W = state.partial_solution, state.W
    
    if len(solution) == 0:
        return random.choice(range(W.shape[0]))
    already_in = set(solution)
    candidates = list(filter(lambda n: n.item() not in already_in, W[solution[-1]].nonzero()))
    if len(candidates) == 0:
        return None
    return random.choice(candidates).item()

# 하이퍼 파라미터
num_epochs = 2  # 학습을 진행할 에포크 수
emb_dim = 128
MEMORY_CAPACITY = 10000
BATCH_SIZE=16
MIN_EPSILON = 0.01
EPSILON_DECAY_RATE = 0.05

# 모델 초기화
model = QNet(emb_dim=emb_dim).to(device)

# 옵티마이저 및 학습률 스케줄러 초기화
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

# 트레이너 초기화
trainer = QTrainer(model, optimizer, lr_scheduler)

# CSV 파일에서 TSP 좌표 데이터 로드
data = pd.read_csv('2024_AI_TSP.csv', header=None)
coordinates = data.iloc[:, :2].values

# 거리 행렬 계산
num_nodes = len(coordinates)
dist_matrix = np.zeros((num_nodes, num_nodes))

for i in range(num_nodes):
    for j in range(num_nodes):
        dist_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])

W = torch.tensor(dist_matrix, dtype=torch.float32).to(device)  # (998, 998)


losses = []
path_lengths = []

# Create memory
memory = Memory(MEMORY_CAPACITY)

for epoch in range(num_epochs):
    solution = [0]

    current_state = State(W=W, partial_solution=solution)
    current_state_tsr = state2tens(current_state, coordinates)

    states = [current_state]
    states_tsrs = [current_state_tsr]
    rewards = []
    actions = []

    # current value of epsilon
    epsilon = max(MIN_EPSILON, (1-EPSILON_DECAY_RATE)**epoch)
    print(f'Epoch: {epoch + 1}, epsilon: {epsilon}')

    nr_explores = 0

    with tqdm(total=num_nodes - 1) as pbar:
        while not is_state_final(current_state):
            if epsilon >= random.random():
                # explore
                next_node = get_next_neighbor_random(current_state)
                nr_explores += 1
            else:
                # exploit
                next_node, est_reward = trainer.get_best_action(current_state_tsr, current_state)

            next_solution = solution + [next_node]

            reward = -W[next_solution[-2], next_solution[-1]].item()

            next_state = State(partial_solution=next_solution, W=W)
            next_state_tsr = state2tens(next_state, coordinates)

            states.append(next_state)
            states_tsrs.append(next_state_tsr)
            rewards.append(reward)
            actions.append(next_node)

            # 저장
            memory.remember(Experience(state=current_state,
                            state_tsr=current_state_tsr,
                            action=next_node,
                            reward=reward,
                            next_state=next_state,
                            next_state_tsr=next_state_tsr))
                    
            # state, current solution 업데이트
            current_state = next_state
            current_state_tsr = next_state_tsr
            solution = next_solution

            loss = None
            if len(memory) >= BATCH_SIZE and len(memory) >= 2000:
                experiences = memory.sample_batch(BATCH_SIZE)
                batch_states_tsrs = [e.state_tsr for e in experiences]
                batch_Ws = [e.state.W for e in experiences]
                batch_actions = [e.action for e in experiences]
                batch_targets = []
                for i, experience in enumerate(experiences):
                    target = experience.reward
                    if not is_state_final(experience.next_state):
                        _, best_reward = trainer.get_best_action(experience.next_state_tsr,
                                                                 experience.next_state)
                        target += 0.9 * best_reward
                    batch_targets.append(target)
                loss = trainer.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
                # print(f'Loss: {loss}')

            pbar.update(1)  # progress bar 업데이트

    length = total_distance(solution, W)
    path_lengths.append(length)
    print(f'Epoch: {epoch + 1}, Solution: {solution}')
    print(f'Epoch: {epoch + 1}, Path Length: {path_lengths[-1]}')

# 에포크에 따른 경로 길이 변화 시각화
plt.plot(range(1, num_epochs + 1), path_lengths)
plt.xlabel('Epoch')
plt.ylabel('Path Length')
plt.title('Change in Path Length over Epochs')
plt.grid(True)
plt.show()