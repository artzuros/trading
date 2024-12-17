### DQN (Deep Q-Network)
**Description:**
DQN is an RL algorithm that uses a neural network to approximate the Q-value function. The Q-value function estimates the expected future rewards for each action in a given state. The agent selects actions based on these Q-values and uses an epsilon-greedy strategy to balance exploration and exploitation.

**Key Features:**
- **Experience Replay:** Stores past experiences and samples them to train the agent.
- **Target Networks:** Periodically updates the target network to stabilize training.

**Architecture in Codebase:**
- **Q-Network:** A neural network with two hidden layers of 128 units each, using ReLU activations.
- **Target Network:** A copy of the Q-Network, updated periodically.
- **Optimizer:** Adam optimizer.
- **Memory:** A replay buffer to store experiences.

### DDPG (Deep Deterministic Policy Gradient)
**Description:**
DDPG is an off-policy RL algorithm designed for continuous action spaces. It uses two neural networks: an actor network to output continuous actions and a critic network to evaluate the Q-value of the state-action pair.

**Key Features:**
- **Deterministic Policy:** Uses a deterministic policy to directly output actions.
- **Ornstein-Uhlenbeck Noise:** Adds noise to the action outputs to help with exploration.
- **Target Networks:** Uses target networks for both actor and critic to stabilize learning.

**Architecture in Codebase:**
- **Actor Network:** A neural network with two hidden layers of 256 units each, using LayerNorm and ReLU activations, and a Tanh activation for the output layer.
- **Critic Network:** A neural network with two hidden layers of 256 units each, using LayerNorm and ReLU activations, and a linear output layer.
- **Target Networks:** Copies of the actor and critic networks, updated periodically.
- **Optimizer:** Adam optimizer.
- **Memory:** A replay buffer to store experiences.
- **Noise:** Ornstein-Uhlenbeck process for exploration.

### A2C (Advantage Actor-Critic)
**Description:**
A2C is a synchronous variant of the Actor-Critic algorithm. The actor network selects actions, and the critic network evaluates the action by estimating the state value. The agent learns by maximizing the advantage function, which measures how much better an action is compared to the average action.

**Key Features:**
- **Synchronous Updates:** Uses multiple agents to update the policy at the same time, ensuring more stable and faster learning.
- **Entropy Regularization:** Encourages exploration and prevents premature convergence of the policy.

**Architecture in Codebase:**
- **Shared Layers:** Two hidden layers of 128 units each, using ReLU activations.
- **Actor Network:** For continuous actions, a mean output layer and a standard deviation parameter. For discrete actions, a linear output layer.
- **Critic Network:** A linear output layer to estimate state values.
- **Optimizer:** Adam optimizer.
- **Entropy Regularization:** Adds entropy to the policy loss to encourage exploration.