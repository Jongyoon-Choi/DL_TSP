import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import time

from collections import namedtuple

from network import QTrainer, QNet

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 파라미터 정의
SEED = 1                     # A seed for the random number generator
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Graph
NR_NODES = 200               # Number of nodes N
EMBEDDING_DIMENSIONS = 5     # Embedding dimension D
EMBEDDING_ITERATIONS_T = 1   # Number of embedding iterations T

# Learning
NR_EPISODES = 100
MEMORY_CAPACITY = 10000
N_STEP_QL = 2                # Number of steps (n) in n-step Q-learning to wait before computing target reward estimate
BATCH_SIZE = 16

GAMMA = 0.9
INIT_LR = 5e-3
LR_DECAY_RATE = 1. - 2e-5    # learning rate decay

MIN_EPSILON = 0.1
EPSILON_DECAY_RATE = 6e-3    # epsilon decay

"""
State, action 관련 자료형, 함수, 클래스 정의
- State : 현재 state에 대한 정보를 저장하기 위한 자료형 
- Experience : state tensor들을 한번만 계산하기 위해 experience 인스턴스에 저장합니다.  
- state2tens : state를 5개의 차원으로 embedding하는 함수  
- 여러 experience를 저장해두기 위한 메모리 클래스  
"""
State = namedtuple('State', ('W', 'coords', 'partial_solution'))
Experience = namedtuple('Experience', ('state', 'state_tsr', 'action', 'reward', 'next_state', 'next_state_tsr'))


def state2tens(state):
    solution = set(state.partial_solution)
    sol_last_node = state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
    sol_first_node = state.partial_solution[0] if len(state.partial_solution) > 0 else -1
    coords = state.coords
    nr_nodes = coords.shape[0]

    xv = [[(1 if i in solution else 0),           # 해당 노드를 방문 했는지 여부
           (1 if i == sol_first_node else 0),     # 해당 노드가 시작 노드인지 여부 
           (1 if i == sol_last_node else 0),      # 해당 노드가 마지막 노드인지 여부
           coords[i,0],                           # 해당 노드의 x좌표
           coords[i,1]                            # 해당 노드의 y좌표
          ] for i in range(nr_nodes)]
    
    return torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=device)


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

# 그 외의 함수 정의
def total_distance(solution, W):
    if len(solution) < 2:
        return 0 
    
    total_dist = 0
    for i in range(len(solution) - 1):
        total_dist += W[solution[i], solution[i+1]].item()
        
    if len(solution) == W.shape[0]:
        total_dist += W[solution[-1], solution[0]].item()

    return total_dist

        
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


def get_distance_matrix(x, num_cities=998):
    x = torch.tensor(x)
    x1, x2 = x[:,0:1], x[:,1:2]
    d1 = x1 - (x1.T).repeat(num_cities,1)
    d2 = x2 - (x2.T).repeat(num_cities,1)
    distance_matrix = (d1**2 + d2**2)**0.5   # Euclidean Distance
    return distance_matrix.numpy()


def init_model(fname=None):
    Q_net = QNet(EMBEDDING_DIMENSIONS, T=EMBEDDING_ITERATIONS_T).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=INIT_LR)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_RATE)
    
    if fname is not None:
        checkpoint = torch.load(fname)
        Q_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    Q_trainer = QTrainer(Q_net, optimizer, lr_scheduler)
    return Q_trainer, Q_net, optimizer, lr_scheduler

# Training Loop
# TSP Data Load
coords = np.array(pd.read_csv('2024_AI_TSP.csv', header=None))
coords = coords[:NR_NODES, :NR_NODES]

# make distance matrix
W_np = get_distance_matrix(coords, num_cities = NR_NODES)

# init Trainer, Model
Q_trainer, Q_net, optimizer, lr_scheduler = init_model()

# generate memory
memory = Memory(MEMORY_CAPACITY)


losses = []
path_lengths = []
found_solutions = dict()
current_min_med_length = float('inf')

start_time = time.time()    # 시간 측정
for episode in range(NR_EPISODES):
    
    # tensor (distance matrix)
    W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)
    
    # start node = 0
    solution = [0]
    
    # current state
    current_state = State(partial_solution=solution, W=W, coords=coords)
    current_state_tsr = state2tens(current_state)
    

    # define state, state_tsrs(embedding), reward, action list
    states = [current_state]
    states_tsrs = [current_state_tsr] 
    rewards = []
    actions = []
    
    
    # current value of epsilon
    epsilon = max(MIN_EPSILON, (1-EPSILON_DECAY_RATE)**episode)
    

    while not is_state_final(current_state):
        
        # select next node
        if epsilon >= random.random():
            next_node = get_next_neighbor_random(current_state)
        else:
            next_node, est_reward = Q_trainer.get_best_action(current_state_tsr, current_state)
        

        # append next node to solution
        next_solution = solution + [next_node]

        # calulate reward
        reward = -(total_distance(next_solution, W) - total_distance(solution, W))
        
        
        next_state = State(partial_solution=next_solution, W=W, coords=coords)
        next_state_tsr = state2tens(next_state)
        
        states.append(next_state)
        states_tsrs.append(next_state_tsr)
        rewards.append(reward)
        actions.append(next_node)
        
        
        if len(solution) >= N_STEP_QL:
            memory.remember(Experience(state=states[-N_STEP_QL],
                                       state_tsr=states_tsrs[-N_STEP_QL],
                                       action=actions[-N_STEP_QL],
                                       reward=sum(rewards[-N_STEP_QL:]),
                                       next_state=next_state,
                                       next_state_tsr=next_state_tsr))
            
        if is_state_final(next_state):
            for n in range(1, N_STEP_QL):
                memory.remember(Experience(state=states[-n],
                                           state_tsr=states_tsrs[-n], 
                                           action=actions[-n], 
                                           reward=sum(rewards[-n:]), 
                                           next_state=next_state,
                                           next_state_tsr=next_state_tsr))
        
        
        current_state = next_state
        current_state_tsr = next_state_tsr
        solution = next_solution
        

        loss = None
        if len(memory) >= BATCH_SIZE:

            # sampling batch experience
            experiences = memory.sample_batch(BATCH_SIZE)
            
            batch_states_tsrs = [e.state_tsr for e in experiences]
            batch_Ws = [e.state.W for e in experiences]
            batch_actions = [e.action for e in experiences]
            batch_targets = []
            

            for i, experience in enumerate(experiences):
                target = experience.reward
                if not is_state_final(experience.next_state):
                    _, best_q_value = Q_trainer.get_best_action(experience.next_state_tsr, experience.next_state)
                    target += GAMMA * best_q_value
                batch_targets.append(target)
                
            loss = Q_trainer.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
            losses.append(loss)

    length = total_distance(solution, W)
    path_lengths.append(length)

    if episode % 10 == 0:
        print('Ep %d. Loss = %.3f, length = %.3f, epsilon = %.4f, lr = %.4f' % (
            episode, (-1 if loss is None else loss), length, epsilon,
            Q_trainer.optimizer.param_groups[0]['lr']))
        found_solutions[episode] = (W.clone(), coords.copy(), [n for n in solution])

end_time = time.time()

# Generate Solutions
solution = [0]
current_state = State(partial_solution=solution, W=W, coords=coords)
current_state_tsr = state2tens(current_state)

while not is_state_final(current_state):
    next_node, est_reward = Q_trainer.get_best_action(current_state_tsr, 
                                                    current_state)
    
    solution = solution + [next_node]
    current_state = State(partial_solution=solution, W=W, coords=coords)
    current_state_tsr = state2tens(current_state)

execution_time = end_time - start_time
minutes = execution_time // 60
seconds = execution_time % 60

print("Final solution : ", str(solution))
print("Final distance : ", total_distance(solution, W_np))
print("실행 시간: {} 분 {} 초".format(int(minutes), round(seconds, 2)))