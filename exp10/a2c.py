import torch
import torch.nn as nn
import torch.optim as optim

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, continuous=False):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.continuous = continuous

        # Actor-Critic Networks
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1) if not continuous else nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        if self.continuous:
            action = action_probs.squeeze(0).numpy()
        else:
            action = torch.argmax(action_probs, dim=1).item()
        return action

    def train(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        if self.continuous:
            actions = torch.FloatTensor(actions)
        else:
            actions = torch.LongTensor(actions)

        # Calculate values and advantages
        values = self.critic(states)
        with torch.no_grad():
            next_values = self.critic(next_states)
            target_values = rewards + self.gamma * next_values * (1 - dones)
        advantages = target_values - values

        # Actor loss
        action_probs = self.actor(states)
        if self.continuous:
            dist = torch.distributions.Normal(action_probs, torch.ones_like(action_probs) * 0.1)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        else:
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss
        critic_loss = nn.MSELoss()(values, target_values)

        # Optimize both losses
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
