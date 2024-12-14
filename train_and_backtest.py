import argparse
import numpy as np
import torch
import random
from env import TradingEnv  # Your custom environment
from dqn import DQNAgent
from ddpg import DDPGAgent
from a2c import A2CAgent
from yfinance import download as yf_download

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy random module
    torch.manual_seed(seed)  # PyTorch CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU seed
    # For torch.backends.cudnn (if using CUDA):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(args, agent, env, num_episodes=500, max_steps=500):
    if args.model == "a2c":
        rewards_history = []
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for step in range(max_steps):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                state = next_state
                total_reward += reward
                if done:
                    break
            agent.train(states, actions, rewards, next_states, dones)
            rewards_history.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        return rewards_history
    else:
        rewards_history = []
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            for step in range(max_steps):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                total_reward += reward
                if done:
                    break
            rewards_history.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        return rewards_history


def backtest(agent, env, max_steps=500):
    """
    Backtest the trained agent.
    """
    state = env.reset()
    total_reward = 0
    actions = []
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        actions.append(action)
        if done:
            break
    print(f"Backtest Total Reward: {total_reward}")
    return total_reward, actions

def fetch_ohlc_data(ticker, start_date, end_date):
    """
    Fetch OHLC data using yfinance.
    """
    data = yf_download(ticker, start=start_date, end=end_date, interval="1d")
    data = data[['Open', 'High', 'Low', 'Close']].dropna()
    return data

def main(args):
    # Set the random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Fetch OHLC data
    ohlc_data = fetch_ohlc_data(args.ticker, args.start_date, args.end_date)
    env = TradingEnv(ohlc_data, action_space_type="discrete" if args.discrete else "continuous")

    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if args.discrete else env.action_space.shape[0]
    if args.model == "dqn":
        agent = DQNAgent(state_dim, action_dim, epsilon=args.epsilon, seed=args.seed)
    elif args.model == "ddpg":
        agent = DDPGAgent(state_dim, action_dim, action_bounds=1.0, seed=args.seed)
    elif args.model == "a2c":
        agent = A2CAgent(state_dim, action_dim, continuous=not args.discrete, seed=args.seed)
    else:
        raise ValueError(f"Invalid model type: {args.model}")

    # Train the agent
    print(f"Training {args.model.upper()}...")
    train_rewards = train_model(args, agent, env, num_episodes=args.episodes)

    # Backtest the agent
    print(f"Backtesting {args.model.upper()}...")
    total_reward, actions = backtest(agent, env)

    # Save results
    np.save(f"{args.model}_train_rewards.npy", train_rewards)
    np.save(f"{args.model}_backtest_actions.npy", actions)
    torch.save(agent, f"{args.model}_agent.pt")
    print(f"Results and model saved for {args.model.upper()}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and backtest RL agents for stock trading.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL).")
    parser.add_argument("--start_date", type=str, default="2020-01-01", help="Start date for data (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default="2023-01-01", help="End date for data (YYYY-MM-DD).")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ddpg", "a2c"], help="Model type.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for exploration (DQN only).")
    parser.add_argument("--discrete", action="store_true", help="Use discrete action space.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(args)
