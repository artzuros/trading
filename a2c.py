import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

def set_seed(seed: int):
    # Set Python random seed
    random.seed(seed)
    
    # Set numpy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you're using multiple GPUs
    
    # Ensure deterministic behavior on CUDA (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example of setting the seed to 42
set_seed(42)

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, continuous=False, entropy_coef=0.01, seed=42):
        self.set_seed(seed)  # Set the seed here
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.continuous = continuous
        self.entropy_coef = entropy_coef

        # Actor-Critic Networks with shared features
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        if continuous:
            self.actor_mean = nn.Linear(128, action_dim)
            self.actor_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(128, action_dim)
            
        self.critic = nn.Linear(128, 1)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def set_seed(self, seed: int):
        # Set Python random seed
        random.seed(seed)
        
        # Set numpy random seed
        np.random.seed(seed)
        
        # Set PyTorch random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you're using multiple GPUs
        
        # Ensure deterministic behavior on CUDA (if applicable)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def parameters(self):
        if self.continuous:
            return list(self.shared_layers.parameters()) + \
                   list(self.actor_mean.parameters()) + \
                   [self.actor_std] + \
                   list(self.critic.parameters())
        else:
            return list(self.shared_layers.parameters()) + \
                   list(self.actor.parameters()) + \
                   list(self.critic.parameters())

    def get_action_distribution(self, state_tensor):
        features = self.shared_layers(state_tensor)
        if self.continuous:
            mean = self.actor_mean(features)
            std = F.softplus(self.actor_std).expand_as(mean)
            return torch.distributions.Normal(mean, std)
        else:
            logits = self.actor(features)
            return torch.distributions.Categorical(logits=logits)

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist = self.get_action_distribution(state_tensor)
            if self.continuous:
                action = dist.sample()
                action = torch.clamp(action, -1, 1)  # Clip actions to valid range
                return action.squeeze(0).numpy()
            else:
                action = dist.sample()
                return action.item()

    def compute_returns(self, rewards, dones, next_state_values):
        returns = torch.zeros_like(rewards)
        future_return = next_state_values[-1] * (1 - dones[-1])
        
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + self.gamma * future_return * (1 - dones[t])
            returns[t] = future_return
        return returns

    def train(self, states, actions, rewards, next_states, dones):
        # Convert to tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        if self.continuous:
            actions = torch.FloatTensor(actions)
        else:
            actions = torch.LongTensor(actions)

        # Get shared features
        features = self.shared_layers(states)
        next_features = self.shared_layers(next_states)

        # Get values
        values = self.critic(features)
        next_values = self.critic(next_features).detach()

        # Compute returns and advantages
        returns = self.compute_returns(rewards, dones, next_values)
        advantages = returns - values.detach()

        # Get action distributions
        dist = self.get_action_distribution(states)
        
        # Compute policy loss
        if self.continuous:
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        else:
            log_probs = dist.log_prob(actions).unsqueeze(1)
        
        # Policy loss with entropy bonus
        entropy = dist.entropy().mean()
        policy_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropy

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }