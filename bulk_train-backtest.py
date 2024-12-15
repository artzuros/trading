import argparse
import numpy as np
import torch
import random
from env import TradingEnv  # Your custom environment
from dqn import DQNAgent
from ddpg import DDPGAgent
from a2c import A2CAgent
from yfinance import download as yf_download
import os
import pandas as pd
import matplotlib.pyplot as plt

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

# def train_model(args, agent, env, num_episodes=50, max_steps=500):
#     rewards_history = []
#     for episode in range(num_episodes):
#         state = env.reset()
#         total_reward = 0
#         for step in range(max_steps):
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             agent.store_transition(state, action, reward, next_state, done)
#             agent.train()
#             state = next_state
#             total_reward += reward
#             if done:
#                 break
#         rewards_history.append(total_reward)
#         print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
#     return rewards_history

def train_model(args, agent, env, num_episodes=20, max_steps=500):
    if args['model'] == "a2c":
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

def save_results(model_name, ticker, seed, rewards, actions, backtest_reward):
    """
    Save results into files.
    """
    results_dir = f"results/{model_name}/{ticker}/{seed}"
    os.makedirs(results_dir, exist_ok=True)
    np.save(f"{results_dir}/train_rewards.npy", rewards)
    np.save(f"{results_dir}/backtest_actions.npy", actions)
    np.save(f"{results_dir}/backtest_reward.npy", backtest_reward)

def compare_results(models, tickers, seeds, num_episodes=50):
    """
    Compare the performance of the models on different tickers and seeds.
    """
    all_results = {}
    
    for model in models:
        all_results[model] = {}

        for ticker in tickers:
            all_results[model][ticker] = {'rewards': [], 'backtest_rewards': []}
            
            for seed in seeds:
                print(f"\nTraining {model.upper()} for {ticker} with seed {seed}...")
                # Set seed for reproducibility
                set_seed(seed)
                
                # Fetch OHLC data
                ohlc_data = fetch_ohlc_data(ticker, "2020-01-01", "2023-01-01")
                env = TradingEnv(ohlc_data, action_space_type="discrete" if model == "dqn" else "continuous")

                # Initialize agent
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.n if model == "dqn" else env.action_space.shape[0]
                
                if model == "dqn":
                    agent = DQNAgent(state_dim, action_dim, epsilon=0.1, seed=seed)
                elif model == "ddpg":
                    agent = DDPGAgent(state_dim, action_dim, action_bounds=1.0, seed=seed)
                elif model == "a2c":
                    agent = A2CAgent(state_dim, action_dim, continuous=True, seed=seed)
                else:
                    raise ValueError(f"Invalid model type: {model}")

                # Train the agent
                train_rewards = train_model({'model': model, 'episodes': num_episodes, 'seed': seed}, agent, env, num_episodes)
                
                # Backtest the agent
                total_backtest_reward, actions = backtest(agent, env)
                
                # Save the results
                save_results(model, ticker, seed, train_rewards, actions, total_backtest_reward)
                
                # Store the results for comparison
                all_results[model][ticker]['rewards'].append(np.mean(train_rewards))
                all_results[model][ticker]['backtest_rewards'].append(total_backtest_reward)

    # Convert results into a DataFrame for easy comparison
    results_df = pd.DataFrame(columns=['Model', 'Ticker', 'Seed', 'Train Reward', 'Backtest Reward'])

    for model in models:
        for ticker in tickers:
            for seed in seeds:
                train_reward = np.mean(all_results[model][ticker]['rewards'])
                backtest_reward = all_results[model][ticker]['backtest_rewards'][seeds.index(seed)]
                results_df = results_df._append({
                    'Model': model,
                    'Ticker': ticker,
                    'Seed': seed,
                    'Train Reward': train_reward,
                    'Backtest Reward': backtest_reward
                }, ignore_index=True)

    # Save the comparison results
    results_df.to_csv("results/comparison.csv", index=False)

    # Plot the comparison
    for model in models:
        plt.figure(figsize=(10, 5))
        plt.title(f"{model.upper()} - Performance Comparison")
        for ticker in tickers:
            avg_train_rewards = [np.mean(all_results[model][ticker]['rewards'])]
            avg_backtest_rewards = [np.mean(all_results[model][ticker]['backtest_rewards'])]
            plt.plot([ticker] * len(avg_train_rewards), avg_train_rewards, 'o-', label=f"{ticker} Train Rewards")
            plt.plot([ticker] * len(avg_backtest_rewards), avg_backtest_rewards, 'x-', label=f"{ticker} Backtest Rewards")
        
        plt.xlabel('Ticker')
        plt.ylabel('Reward')
        plt.legend(loc="best")
        plt.savefig(f"results/{model}_performance_comparison.png")
        plt.show()

def main(args):
    models = ['a2c', 'dqn', 'ddpg']
    tickers = args.tickers.split(',')
    seeds = [int(seed) for seed in args.seeds.split(',')]
    
    # Compare results across models, tickers, and seeds
    compare_results(models, tickers, seeds, num_episodes=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and backtest RL agents for stock trading.")
    parser.add_argument("--model", type=str, default="dqn,ddpg", help="Comma-separated list of model types.")
    parser.add_argument("--tickers", type=str, default="AAPL,GOOG,MSFT", help="Comma-separated list of stock tickers.")
    parser.add_argument("--seeds", type=str, default="42,55,457", help="Comma-separated list of random seeds.")
    args = parser.parse_args()
    
    main(args)
