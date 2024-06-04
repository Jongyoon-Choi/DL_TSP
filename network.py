import torch
import torch.nn as nn
import torch.nn.functional as F

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
        """ 주어진 state에 대해 최적의 greedy action을 선택하는 단계. 
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
        """ Batch단위의 (embedding of state, distance matrix, action, target_q_value)를 통해 Gradient를 통한 최적화를 수행하는 단계.
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