import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from gymnasium import spaces, Env
from stable_baselines3 import PPO, DQN
import os
import shutil
import webbrowser

# Define the custom strategy
class periodicStrategy(Strategy):
    def init(self):
        print(f"Start with equity={self.equity:.2f}")

    def next(self, action: int | None = None):
        print(f"Action={action} Equity={self.equity:.2f} Date={self.data.index[-1]}")
        if action:
            if action == 1:
                self.buy()
            elif action == 2:
                self.position.close()

    def observation(self):
        closes = self.data.Close[-20:]
        closes = (closes - closes.min()) / (closes.max() - closes.min())
        return [closes]


# Define the custom environment
class CustomEnv(Env):
    """Custom Environment that follows gym interface."""    
    def __init__(self, bt: Backtest):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 20), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # Actions: 0 = do nothing, 1 = buy, 2 = sell
        self.bt = bt
        
    def reward_calculation(self):
        if self.previous_equity < self.bt._step_strategy.equity:
            return +1
        return -1
        
    def check_done(self):
        if self.bt._step_time + 2 > len(self.bt._data):
            self.render()
            return True
        return False
        
    def step(self, action):
        obs = self.bt._step_strategy.observation()
        reward = self.reward_calculation()
        done = self.check_done()
        info = {}
        self.bt.next(action=action)
        return obs, reward, False, done, info

    def reset_backtesting(self):
        self.bt.initialize()
        self.bt.next()
        while True:
            obs = self.bt._step_strategy.observation()
            if np.shape(obs) == (1, 20):  # Wait for the first valid observation
                break
            self.bt.next()
                
    def reset(self, seed=None):
        self.previous_equity = 10
        self.reset_backtesting()
        return self.bt._step_strategy.observation(), {}

    def render(self, mode='human'):
        result = self.bt.next(done=True)
        self.bt.plot(results=result, open_browser=False)
        
    def close(self):
        pass

# Get stock data from Yahoo Finance
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df


# Streamlit frontend
st.title('Stock Trading with Reinforcement Learning')

# User inputs
ticker = st.text_input("Enter Stock Ticker", "AAPL")
period = st.selectbox("Select Time Period", ["1y", "3y", "5y"])
training_timesteps = st.number_input("Training Timesteps", min_value=100, max_value=10000, step=1000, value=1000)

# Fetch the stock data
st.subheader("Stock Data")
stock_data = get_stock_data(ticker, period)

# Show stock data in a table
col1, col2 = st.columns([3, 1])
with col1:
    st.write(stock_data.head())

# Plot stock data graph
with col2:
    st.subheader("Stock Data Plot")
    fig, ax = plt.subplots(figsize=(10, 5))  # Landscape mode
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker} Stock Price")
    ax.legend(loc='best')
    st.pyplot(fig)


# Set up the backtest
bt = Backtest(stock_data, periodicStrategy, cash=10000)

# Set up the environment
env = CustomEnv(bt)

# Training the RL agent (PPO or DQN)
train_button = st.button("Train PPO Agent")

if train_button:
    with st.spinner("Training the model..."):
        # Train PPO agent
        # ppo_model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./logs_ppo/")
        # ppo_model.learn(total_timesteps=training_timesteps, log_interval=1)

        dqn_model = DQN("MlpPolicy", env, verbose=0, tensorboard_log="./logs_dqn/")
        dqn_model.learn(total_timesteps=10000, log_interval=1)
        
        bt = Backtest(stock_data, periodicStrategy, cash=10000)
        # Run the backtest
        result = bt.run()
        st.success("Backtest completed!")

        # Plot the backtest results and show it
        bt.plot(open_browser=True)

        st.success(f"Training completed with {training_timesteps} timesteps!")
