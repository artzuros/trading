import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DDPGAgent:
    def __init__(self, input_dim, action_dim, hidden_dim=256, gamma=0.99, tau=0.005, lr=0.001):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.memory = []
        self.batch_size = 64
        
        self.actor = self._build_network(input_dim, action_dim, hidden_dim, output_activation=nn.Tanh())
        self.actor_target = self._build_network(input_dim, action_dim, hidden_dim, output_activation=nn.Tanh())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = self._build_network(input_dim + action_dim, 1, hidden_dim, output_activation=nn.Identity())
        self.critic_target = self._build_network(input_dim + action_dim, 1, hidden_dim, output_activation=nn.Identity())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Copy weights to target networks
        self._update_target_networks(tau=1.0)

    def _build_network(self, input_dim, output_dim, hidden_dim, output_activation):
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            output_activation  # Output activation directly
        ]
        return nn.Sequential(*layers)

    def _update_target_networks(self, tau=None):
        tau = tau or self.tau
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        action += noise * np.random.normal(size=self.action_dim)
        return np.clip(action, -1, 1)

    def get_action(self, state, noise=0.1):
        """Alias for select_action."""
        return self.select_action(state, noise)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Limit memory size
            self.memory.pop(0)

    def train_step(self, state, reward, next_state, done):
        action = self.select_action(state)
        next_action = self.select_action(next_state)
        self.store_transition(state, action, reward, next_state, done)
        self.update()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Update critic
        next_actions = self.actor_target(next_states)
        target_q_values = self.critic_target(torch.cat([next_states, next_actions], dim=1))
        target_q_values = rewards + (1 - dones) * self.gamma * target_q_values.detach()

        q_values = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = nn.MSELoss()(q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(torch.cat([states, self.actor(states)], dim=1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._update_target_networks()
