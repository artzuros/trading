import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)  
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = np.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class StateNormalizer:
    def __init__(self, epsilon=4e-8):
        self.mean = None
        self.std = None
        self.epsilon = epsilon

    def fit(self, states):
        self.mean = np.mean(states, axis=0)
        self.std = np.std(states, axis=0) + self.epsilon

    def transform(self, state):
        if self.mean is None:
            return state
        return (state - self.mean) / self.std

    def inverse_transform(self, normalized_state):
        return normalized_state * self.std + self.mean

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bounds, gamma=0.99, lr=3e-4, tau=0.005,
                 min_trade_amount=1000, max_trade_amount=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim  
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        ).to(self.device)

        self.target_actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        ).to(self.device)
        
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Critic Network  
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

        self.target_critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(), 
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Noise process
        self.noise = OUNoise(action_dim)

        # Replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 256

        self.min_trade_amount = min_trade_amount
        self.max_trade_amount = max_trade_amount
        self.state_normalizer = StateNormalizer()

    def scale_action_to_trade_amount(self, action):
        """Scale normalized action (-1, 1) to actual trade amount"""
        scaled_action = action * self.max_trade_amount
        # Apply minimum trade threshold
        scaled_action = np.where(
            np.abs(scaled_action) < self.min_trade_amount,
            0,
            scaled_action
        )
        return np.clip(scaled_action, -self.max_trade_amount, self.max_trade_amount)

    def act(self, state, evaluate=False):
        # Normalize state
        normalized_state = self.state_normalizer.transform(state)
        normalized_state = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            normalized_action = self.actor(normalized_state).cpu().numpy().flatten()
            if not evaluate:
                normalized_action += self.noise.sample()
                normalized_action = np.clip(normalized_action, -1, 1)
            
            # Scale to actual trade amount
            real_action = self.scale_action_to_trade_amount(normalized_action)
            
        return real_action

    def store_transition(self, state, action, reward, next_state, done):
        # Normalize state and action before storing
        normalized_state = self.state_normalizer.transform(state)
        normalized_next_state = self.state_normalizer.transform(next_state)
        normalized_action = action / self.max_trade_amount  # Scale action back to [-1, 1]
        
        self.memory.append((normalized_state, normalized_action, reward, normalized_next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Update state normalizer periodically
        if len(self.memory) % 1000 == 0:
            states = np.vstack([t[0] for t in self.memory])
            self.state_normalizer.fit(states)

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = map(np.vstack, zip(*batch))
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(1 - done).to(self.device)

        # Update critic
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            target_q = reward + done * self.gamma * self.target_critic(
                torch.cat([next_state, next_action], dim=1))
            
        current_q = self.critic(torch.cat([state, action], dim=1))
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(torch.cat([state, self.actor(state)], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param + (1.0 - self.tau) * target_param)
            
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param + (1.0 - self.tau) * target_param)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])