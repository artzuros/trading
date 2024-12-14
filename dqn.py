import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQNAgent:
    def __init__(self, state_dim, action_dim, epsilon=0.1, gamma=0.99, lr=1e-3, target_update=10, seed=42):
        self.set_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.target_update = target_update

        # Q-Networks
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = []
        self.batch_size = 64
        self.max_memory = 10000
        self.steps = 0

    def set_seed(self, seed: int):
        # Set Python random seed
        random.seed(seed)
        
        # Set numpy random seed
        np.random.seed(seed)
        
        # Set PyTorch random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic behavior on CUDA (if applicable)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute Q values
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.steps += 1
